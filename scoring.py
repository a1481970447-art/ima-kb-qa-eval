"""D4 产出：双层评分器（规则层 + LLM-as-judge）。

对应 7 天计划 D4：规则指标 + judge prompt 调通。
- 规则层（自动、零成本）：检索命中率 / 拒答准确率 / 端到端延迟
- 模型层（LLM-as-judge，独立 prompt）：忠实度 / 正确性 / 完整性
- 裁判每条指标跑 3 次取多数，agreement < 3/3 的分歧样本单独记录

设计依据（引用进评测报告）：
- D15（MT-Bench, arXiv:2306.05685）：LLM-as-judge 的位置/冗长/自我增强三类偏差
  → 裁判 prompt 显式声明"长度不影响打分"，且裁判与被测模型共用 hy3
  属自我增强偏差风险，靠 majority vote + 独立评分维度缓解
- D14（RAGAS, arXiv:2309.15217）：faithfulness = 回答论断可由检索上下文支撑
- D16（Seven Failure Points）：检索层与生成层分离的归因依据

用法：
    python scoring.py demo                     # 6 条验收样本（四类题型 × 全部 5 指标）
    python scoring.py demo --ids F-01,J-01     # 自选样本
    python scoring.py demo --no-llm            # 只跑规则层（不调 judge，秒出）
    D5 全量跑批直接 import：from scoring import score_item
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from llm_client import chat  # noqa: E402
from rag_pipeline import EVAL_SET, RagPipeline  # noqa: E402

JUDGE_RUNS = 3          # 裁判每指标运行次数（majority vote）
JUDGE_TEMPERATURE = 0.7 # 采样温度高于生成（0.2），测的是裁判稳定性而非确定性复读
VERDICT_SCORE = {"supported": 2, "partial": 1, "unsupported": 0,
                 "correct": 2, "incorrect": 0,
                 "complete": 2, "missing": 0}
DEMO_IDS = ["F-01", "F-02", "R-10", "R-15", "J-01", "M-05"]

# ---------------- 规则层 ----------------
# 拒答话术检测：先匹配 SYSTEM_PROMPT 内置话术，再覆盖常见改写
REFUSAL_PATTERNS = [
    r"知识库中没有找到", r"没有找到与该问题相关的可靠内容",
    r"(无法|没能|未能)(从|在)(知识库|资料|参考|库中|文中)找到",
    r"(知识库|资料|参考|库中|文中)(里)?(没有|并未|不含|未收录|未记载)",
    r"没有(任何)?(相关|对应|可靠)?(的)?(记载|记录|信息|内容|数据)",
    r"无法(给出|确定|回答|提供)(确切的)?(答案|数字|数据|信息)?",
]


def detect_refusal(answer: str) -> bool:
    return any(re.search(p, answer) for p in REFUSAL_PATTERNS)


def rule_metrics(item: dict, record: dict) -> dict:
    """规则层三指标，全部零 LLM 调用。

    - retrieval_hit：金标文档进入 retrieved（top-k 全量 / top-3 两档口径，
      rank 记金标文档的最高排名，供 D5 归因用：rank>3 的"命中"仍是弱命中）
    - refusal_accuracy：拒答题应拒答 & 非拒答题不应错误拒答
    - latency：端到端延迟直接取 record
    """
    gold_docs = set(item["gold_doc_id"])
    doc_ids = [c["doc_id"] for c in record["retrieved"]]
    if gold_docs:
        ranks = [i + 1 for i, d in enumerate(doc_ids) if d in gold_docs]
        rank = min(ranks) if ranks else None
        hit, hit_top3 = rank is not None, rank is not None and rank <= 3
    else:  # 拒答题无金标
        rank, hit, hit_top3 = None, None, None

    expect_refusal = item["type"] == "refusal"
    refused = detect_refusal(record["answer"])
    refusal_ok = refused if expect_refusal else not refused

    return {
        "retrieval_hit": hit,
        "retrieval_hit_top3": hit_top3,
        "retrieval_rank": rank,
        "refusal_expected": expect_refusal,
        "refusal_detected": refused,
        "refusal_accuracy": refusal_ok,
        "latency_retrieve_s": record["latency_retrieve_s"],
        "latency_generate_s": record["latency_generate_s"],
        "latency_total_s": record["latency_total_s"],
    }


# ---------------- 模型层（LLM-as-judge） ----------------
JUDGE_SYSTEM = (
    "你是 RAG 评测体系的裁判（LLM-as-judge）。你的任务不是回答问题，"
    "而是对【待评回答】按指定维度打分。严格遵守：\n"
    "1. 只依据给出的评测材料判断，不使用你自己关于 ima、腾讯或任何产品的知识。\n"
    "2. 回答长短不影响打分：冗长的回答不加分，简洁的回答不扣分。\n"
    "3. 待评回答的措辞不需要与金标一致，只判断语义与要点。\n"
    "4. 只输出一行 JSON（不要思考过程、不要代码块标记）：\n"
    '   {"verdict": "<见各维度说明>", "reason": "<30字内中文理由>"}'
)

METRIC_PROMPTS = {
    "faithfulness": (
        "【评分维度】引用忠实度：待评回答中的事实论断是否都能由【检索上下文】支撑。\n"
        "verdict 取值：\n"
        '- "supported"：论断基本都有上下文支撑（允许合理概括）\n'
        '- "partial"：部分论断无出处，但主体有支撑\n'
        '- "unsupported"：大量内容无依据、编造或与上下文冲突\n\n'
        "【检索上下文】\n{context}\n\n【问题】{question}\n\n【待评回答】\n{answer}"
    ),
    "factuality": (
        "【评分维度】事实正确性：待评回答与【金标答案】相比，关键事实是否正确。\n"
        "verdict 取值：\n"
        '- "correct"：关键事实（数字/日期/名称/结论）与金标一致\n'
        '- "partial"：部分正确或不够精确，但无关键错误\n'
        '- "incorrect"：存在关键事实错误（错误数字/日期/张冠李戴/与金标矛盾）\n\n'
        "【金标答案（已溯源到文档原文）】\n{gold}\n\n"
        "【金标原文出处】{quote}\n\n【问题】{question}\n\n【待评回答】\n{answer}"
    ),
    "completeness": (
        "【评分维度】完整性：金标答案的要点是否被待评回答覆盖。\n"
        "verdict 取值：\n"
        '- "complete"：要点基本覆盖（允许次要细节缺失）\n'
        '- "partial"：覆盖了部分要点，遗漏明显\n'
        '- "missing"：只覆盖少数要点或全部遗漏\n\n'
        "【金标答案（要点清单）】\n{gold}\n\n【问题】{question}\n\n【待评回答】\n{answer}"
    ),
}

# 指标 → 该跑哪些题（与 eval_set 的 metrics 字段对齐）
JUDGE_METRICS = ("faithfulness", "factuality", "completeness")


def _render_prompt(metric: str, item: dict, record: dict) -> str:
    ctx = "\n\n".join(
        f"[{c['chunk_id']}（{c['section']}）] {c['text']}"
        for c in record["retrieved"]  # 与生成上下文同源（完整 top-k）：
        # D4 验收时曾只喂 top-5，导致回答引用 top-8 内合法 chunk 被误判"无依据"——
        # 忠实度判定基准必须与被测生成过程完全一致
    )
    return METRIC_PROMPTS[metric].format(
        context=ctx, question=item["question"], answer=record["answer"],
        gold=item["gold_answer"], quote=item["gold_source_quote"][:300],
    )


def parse_verdict(raw: str) -> dict:
    """从裁判回复中提取 JSON（容忍思考模型的包裹文本）。"""
    m = re.search(r"\{[^{}]*\}", raw, re.S)
    if not m:
        return {"verdict": "parse_error", "reason": raw[:80]}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {"verdict": "parse_error", "reason": raw[:80]}


def judge_once(metric: str, item: dict, record: dict) -> dict:
    raw = chat(_render_prompt(metric, item, record), system=JUDGE_SYSTEM,
               max_tokens=4096, temperature=JUDGE_TEMPERATURE)
    out = parse_verdict(raw)
    out["score"] = VERDICT_SCORE.get(out.get("verdict"))
    return out


def judge_majority(metric: str, item: dict, record: dict, n: int = JUDGE_RUNS) -> dict:
    """同一指标跑 n 次取多数。分歧（agreement < n）单独记录，供人工仲裁。"""
    runs = [judge_once(metric, item, record) for _ in range(n)]
    verdicts = [r["verdict"] for r in runs]
    top, cnt = Counter(verdicts).most_common(1)[0]
    return {
        "metric": metric,
        "runs": runs,
        "majority_verdict": top,
        "score": VERDICT_SCORE.get(top),
        "agreement": f"{cnt}/{n}",
        "stable": cnt == n,
        "disagreement": verdicts if cnt < n else [],
    }


def judge_metrics(item: dict, record: dict) -> dict:
    """按 eval_set 的 metrics 字段决定跑哪些裁判指标。"""
    out = {}
    for m in JUDGE_METRICS:
        if m in item.get("metrics", []):
            out[m] = judge_majority(m, item, record)
    return out


# ---------------- 汇总 ----------------
def score_item(item: dict, record: dict, use_llm: bool = True) -> dict:
    """D5 全量跑批的入口：一条评测记录 → 完整双层评分。"""
    result = {"id": item["id"], "type": item["type"], "subtype": item["subtype"],
              "rule": rule_metrics(item, record)}
    if use_llm:
        result["judge"] = judge_metrics(item, record)
        scores = [j["score"] for j in result["judge"].values() if j["score"] is not None]
        if scores:
            result["judge_avg"] = round(statistics.mean(scores), 2)
    return result


def load_eval_set() -> list[dict]:
    return json.loads(EVAL_SET.read_text(encoding="utf-8"))["eval_set"]


# ---------------- CLI ----------------
def cmd_demo(ids: list[str] | None, use_llm: bool) -> None:
    eval_set = {e["id"]: e for e in load_eval_set()}
    picks = ids or DEMO_IDS
    pipe = RagPipeline()
    results = []
    t0 = time.perf_counter()

    for qid in picks:
        item = eval_set[qid]
        print(f"\n{'═' * 62}\n[{qid}]（{item['type']}｜{item['subtype']}）{item['question']}")
        record = pipe.answer(item["question"])
        scored = score_item(item, record, use_llm=use_llm)
        scored["record"] = {"answer": record["answer"],
                            "retrieved_doc_ids": [c["doc_id"] for c in record["retrieved"]]}
        results.append(scored)

        r = scored["rule"]
        hit_desc = (f"hit_top3={r['retrieval_hit_top3']} rank={r['retrieval_rank']}"
                    if r["retrieval_hit"] is not None else "拒答题（无金标）")
        print(f"  规则层：{hit_desc} ｜ 拒答={r['refusal_accuracy']} ｜ "
              f"延迟 {r['latency_total_s']}s")
        for m, j in scored.get("judge", {}).items():
            flag = "✅" if j["stable"] else "⚠️ 分歧"
            print(f"  judge.{m}：{j['majority_verdict']}（{j['agreement']} 一致）"
                  f" {flag} — {j['runs'][0]['reason']}")

    # 验收汇总
    n_judged = [s for s in results if s.get("judge")]
    disagreements = [
        {"id": s["id"], "metric": m, "verdicts": j["disagreement"]}
        for s in n_judged for m, j in s["judge"].items() if not j["stable"]
    ]
    print(f"\n{'═' * 62}\nD4 验收汇总（{len(results)} 条 · "
          f"{time.perf_counter() - t0:.0f}s）")
    print(f"  跑通：{len(results)}/{len(picks)} 条端到端完成")
    if n_judged:
        ok = sum(1 for s in n_judged for j in s["judge"].values() if j["agreement"] != "0/3")
        total = sum(len(s["judge"]) for s in n_judged)
        maj = sum(1 for s in n_judged for j in s["judge"].values()
                  if int(j["agreement"][0]) >= 2)
        print(f"  裁判一致性：{maj}/{total} 个指标达到 majority（≥2/3）"
              f"{'✅' if maj == total else '⚠️'}")
        print(f"  完全一致（3/3）占比："
              f"{sum(1 for s in n_judged for j in s['judge'].values() if j['stable'])}/{total}")
    if disagreements:
        print(f"  分歧样本（需人工仲裁，D5 复核）：{len(disagreements)} 个")
        for d in disagreements:
            print(f"    - {d['id']}.{d['metric']}: {d['verdicts']}")

    out = Path(ROOT / "d4_demo_results.json")
    out.write_text(json.dumps(
        {"meta": {"day": "D4", "date": "2026-08-28", "sample_ids": picks,
                  "judge_runs": JUDGE_RUNS, "judge_temperature": JUDGE_TEMPERATURE},
         "results": results,
         "disagreements": disagreements},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已保存 → {out.name}")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="双层评分器（D4）")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_demo = sub.add_parser("demo", help="验收样本演示（默认 6 条四类题型）")
    p_demo.add_argument("--ids", type=str, help="逗号分隔的题目 id")
    p_demo.add_argument("--no-llm", action="store_true", help="只跑规则层")
    args = ap.parse_args()

    if args.cmd == "demo":
        ids = args.ids.split(",") if args.ids else None
        cmd_demo(ids, use_llm=not args.no_llm)


if __name__ == "__main__":
    main()
