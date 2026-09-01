"""D5 产出：全量跑批 + badcase 三层自动归因。

对应 7 天计划 D5（差异化核心）：
- 50 条评测集批量跑批，输出 JSON / CSV 全量结果
- 错误自动归因到三层：检索层（没召回/弱命中）/ 生成层（召回了但答错）/ 数据层（库里脏数据污染）
- judge 分歧样本 + 边界 case 导出人工复核清单，仲裁结论记录在 D5_跑批与归因说明.md

归因决策树（详见 D5_跑批与归因说明.md）：
    A. 拒答题（J）：应拒答却乱答 → generation 层（拒答边界执行失败）
    B. 非拒答题错误：
       1. 错误拒答（该答却拒答）：金标在 top-3 → generation；否则 → retrieval
       2. 金标未进 top-3（含未召回/弱命中 rank>3）→ retrieval
       3. multi_doc 题金标未全部进 top-3 → retrieval（部分召回）
       4. 金标在 top-3、答案错误：
          - 数据层信号命中（回答忠实于上下文但与金标矛盾 ｜ 陷阱冲突文档被召回）
            → data 层
          - 否则 → generation 层
    C. 无 fatal 错误 → correct（partial 判定记为弱问题，进复核清单）

用法：
    python run_eval.py run                # 全量跑批（断点续跑，checkpoint 增量保存）
    python run_eval.py run --limit 2      # 冒烟
    python run_eval.py run --fresh        # 清空 checkpoint 重跑
    python run_eval.py stats              # 基于 checkpoint 重算统计（不调 LLM）
    python run_eval.py review             # 导出人工复核清单 d5_review_list.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import llm_client  # noqa: E402

# ---- 给 chat 调用加重试补丁（网络抖动 / 限流），不动 D4 已验收的 scoring.py ----
_orig_chat = llm_client.chat


def _chat_retry(prompt: str, **kw) -> str:
    for attempt in range(4):
        try:
            return _orig_chat(prompt, **kw)
        except Exception:
            if attempt == 3:
                raise
            time.sleep(5 * (attempt + 1))


llm_client.chat = _chat_retry

from rag_pipeline import RagPipeline  # noqa: E402
from scoring import (  # noqa: E402
    JUDGE_METRICS,
    JUDGE_RUNS,
    VERDICT_SCORE,
    judge_once,
    load_eval_set,
    rule_metrics,
)

CHECKPOINT = ROOT / "d5_checkpoint.json"
RESULTS_JSON = ROOT / "d5_results.json"
RESULTS_CSV = ROOT / "d5_results.csv"
REVIEW_CSV = ROOT / "d5_review_list.csv"
ARBITRATION = ROOT / "d5_arbitration.json"
JUDGE_WORKERS = 4

# 数据层陷阱文档提取：trap_note 中「数据层陷阱：Dxx」/「数据层错误文档 Dxx」
CONFLICT_RES = [
    re.compile(r"数据层陷阱[：:]\s*(D\d{2})"),
    re.compile(r"数据层(?:错误|过期|污染)[^。；]{0,24}?(D\d{2})"),
]


def conflict_docs(item: dict) -> list[str]:
    gold = set(item.get("gold_doc_id") or [])
    hits: set[str] = set()
    note = item.get("trap_note", "")
    for rx in CONFLICT_RES:
        hits |= {m.group(1) for m in rx.finditer(note)}
    return sorted(hits - gold)


# ---------------- 并发 judge（复用 scoring.judge_once，majority 逻辑同 D4） ----------------
def judge_metric_concurrent(pool: ThreadPoolExecutor, metric: str,
                            item: dict, record: dict) -> dict:
    futures = [pool.submit(judge_once, metric, item, record) for _ in range(JUDGE_RUNS)]
    runs = [f.result() for f in futures]
    for idx, r in enumerate(runs):  # parse_error 补跑（最多 2 次）
        tries = 0
        while r.get("verdict") == "parse_error" and tries < 2:
            tries += 1
            r = judge_once(metric, item, record)
        runs[idx] = r
    verdicts = [r["verdict"] for r in runs]
    top, cnt = Counter(verdicts).most_common(1)[0]
    return {
        "metric": metric,
        "runs": runs,
        "majority_verdict": top,
        "score": VERDICT_SCORE.get(top),
        "agreement": f"{cnt}/{len(verdicts)}",
        "stable": cnt == len(verdicts),
        "disagreement": verdicts if cnt < len(verdicts) else [],
    }


def score_item_concurrent(pool: ThreadPoolExecutor, item: dict, record: dict) -> dict:
    """与 scoring.score_item 输出结构一致（rule + judge + judge_avg），D6 看板可直接复用。"""
    result = {"id": item["id"], "type": item["type"], "subtype": item["subtype"],
              "rule": rule_metrics(item, record)}
    judge = {}
    metrics = [m for m in JUDGE_METRICS if m in item.get("metrics", [])]
    if metrics:
        judge = {m: judge_metric_concurrent(pool, m, item, record) for m in metrics}
    result["judge"] = judge
    scores = [j["score"] for j in judge.values() if j["score"] is not None]
    if scores:
        result["judge_avg"] = round(statistics.mean(scores), 2)
    return result


# ---------------- 三层自动归因 ----------------
def _v(judge: dict, metric: str):
    return judge.get(metric, {}).get("majority_verdict")


def attribute(item: dict, scored: dict, record: dict) -> dict:
    """返回 {layer, reason, errors, needs_review}。layer ∈ correct/retrieval/generation/data。"""
    rule, judge = scored["rule"], scored.get("judge", {})
    gold = set(item.get("gold_doc_id") or [])
    docs_all = [c["doc_id"] for c in record["retrieved"]]
    docs_top3 = docs_all[:3]

    # ---- 错误判定（fatal）----
    errors: list[str] = []
    if not rule["refusal_accuracy"]:
        errors.append("错误拒答" if not rule["refusal_expected"] else "应拒答未拒答")
    if _v(judge, "factuality") == "incorrect":
        errors.append("事实错误")
    if _v(judge, "faithfulness") == "unsupported":
        errors.append("无依据编造")
    if _v(judge, "completeness") == "missing":
        errors.append("要点遗漏")

    needs_review = any(_v(judge, m) == "partial" for m in ("factuality", "faithfulness",
                                                            "completeness"))
    if not errors:
        layer = "correct"
        reason = ""
    elif rule["refusal_expected"]:
        # A. 拒答题：应拒答却没拒答 → 生成层没守住边界；陷阱文档进 top-3 记为诱因
        layer = "generation"
        trap_in = sorted(set(conflict_docs(item)) & set(docs_top3))
        reason = ("拒答边界执行失败：库外问题未拒答"
                  + (f"（陷阱文档 {trap_in} 高位召回为诱因，属检索+数据复合）" if trap_in else ""))
    elif not rule["refusal_accuracy"] and not rule["refusal_expected"]:
        # B1. 错误拒答（该答却拒答）
        if rule["retrieval_hit_top3"]:
            layer, reason = "generation", "金标已入 top-3 仍错误拒答"
        else:
            layer, reason = "retrieval", "金标未召回，模型无据可答而拒答"
    elif rule["retrieval_hit_top3"] is False:
        # B2. 金标未进 top-3（未召回 / 弱命中）
        if rule["retrieval_hit"]:
            layer = "retrieval"
            reason = f"金标弱命中（rank={rule['retrieval_rank']}，未进 top-3）"
        else:
            layer, reason = "retrieval", "金标文档未召回"
    elif item["type"] == "multi_doc" and (gold - set(docs_top3)):
        # B3. 多文档题：金标未全部进入 top-3 → 部分召回
        layer = "retrieval"
        reason = f"多文档题金标未全部进 top-3（缺 {sorted(gold - set(docs_top3))}）"
    else:
        # B4. 金标在 top-3，答案错 → 数据层 or 生成层
        fact_bad = _v(judge, "factuality") == "incorrect"
        faithful = _v(judge, "faithfulness")
        conf_in = sorted(set(conflict_docs(item)) & set(docs_all))
        data_signal = []
        if fact_bad and faithful == "supported":
            data_signal.append("回答忠实于检索上下文但与金标矛盾（上下文被污染）")
        if fact_bad and conf_in:
            data_signal.append(f"陷阱冲突文档 {conf_in} 被召回")
        if data_signal:
            layer, reason = "data", "；".join(data_signal)
        else:
            layer = "generation"
            reason = "召回上下文正确但生成错误（编造/误读/要点丢失）"

    return {"layer": layer, "reason": reason, "errors": errors,
            "needs_review": needs_review or layer == "data"}  # 数据层判定全部进人工复核


# ---------------- 跑批主流程 ----------------
def load_checkpoint() -> dict:
    if CHECKPOINT.exists():
        return json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    return {}


def save_checkpoint(ckpt: dict) -> None:
    CHECKPOINT.write_text(json.dumps(ckpt, ensure_ascii=False), encoding="utf-8")


def run_one(pipe: RagPipeline, pool: ThreadPoolExecutor, item: dict) -> dict:
    """单条：生成 → 评分 → 归因。带 3 次整体重试。"""
    last_err = None
    for attempt in range(3):
        try:
            record = pipe.answer(item["question"])
            scored = score_item_concurrent(pool, item, record)
            scored["record"] = {
                "answer": record["answer"],
                "retrieved": [{"chunk_id": c["chunk_id"], "doc_id": c["doc_id"],
                               "section": c["section"], "score": c["score"]}
                              for c in record["retrieved"]],
                "latency_total_s": record["latency_total_s"],
            }
            scored["attribution"] = attribute(item, scored, record)
            return scored
        except Exception as e:  # noqa: BLE001
            last_err = e
            print(f"    ⚠️ 第 {attempt + 1} 次失败：{e!r}，重试…")
            time.sleep(10 * (attempt + 1))
    return {"id": item["id"], "type": item["type"], "subtype": item["subtype"],
            "failed": True, "error": repr(last_err)}


def cmd_run(limit: int | None, fresh: bool) -> None:
    eval_set = load_eval_set()
    ckpt = {} if fresh else load_checkpoint()
    if fresh and CHECKPOINT.exists():
        CHECKPOINT.unlink()
    pipe = RagPipeline()
    pool = ThreadPoolExecutor(max_workers=JUDGE_WORKERS)

    todo = [it for it in eval_set if it["id"] not in ckpt]
    if limit:
        todo = todo[:limit]
    print(f"全量 {len(eval_set)} 条 ｜ checkpoint 已有 {len(ckpt)} 条 ｜ 本次待跑 {len(todo)} 条")
    t_start = time.perf_counter()

    for i, item in enumerate(todo, 1):
        t0 = time.perf_counter()
        scored = run_one(pipe, pool, item)
        ckpt[item["id"]] = scored
        save_checkpoint(ckpt)
        if scored.get("failed"):
            print(f"[{i}/{len(todo)}] {item['id']} ❌ FAILED（checkpoint 已存，末尾统一重试）")
            continue
        a = scored["attribution"]
        r = scored["rule"]
        flag = "✅" if a["layer"] == "correct" else "❌"
        hit = (f"rank={r['retrieval_rank']}" if r["retrieval_hit_top3"]
               else f"hit={r['retrieval_hit']}（未进top3）" if r["retrieval_hit"] is not None
               else "拒答题")
        dis = sum(1 for j in scored.get("judge", {}).values() if not j["stable"])
        print(f"[{i}/{len(todo)}] {item['id']} {item['type']:<9} {flag} "
              f"{a['layer']:<10} {hit:<16} judge={scored.get('judge_avg', '-')} "
              f"分歧={dis}  {time.perf_counter() - t0:.0f}s")

    # 失败条目末尾统一重试一轮
    failed = [k for k, v in ckpt.items() if v.get("failed")]
    for qid in failed:
        item = next(e for e in eval_set if e["id"] == qid)
        print(f"↻ 重试失败条目 {qid}")
        scored = run_one(pipe, pool, item)
        ckpt[qid] = scored
        save_checkpoint(ckpt)
    still_failed = [k for k, v in ckpt.items() if v.get("failed")]

    print(f"\n跑批完成：{len(ckpt)}/{len(eval_set)} 条，"
          f"耗时 {(time.perf_counter() - t_start) / 60:.1f} min"
          + (f"，仍失败 {still_failed}" if still_failed else ""))
    pool.shutdown(wait=False)
    finalize(ckpt)


# ---------------- 汇总：统计 + 导出 ----------------
def build_rows(ckpt: dict, arbitration: dict | None = None) -> list[dict]:
    eval_set = {e["id"]: e for e in load_eval_set()}
    rows = []
    for qid in sorted(ckpt):
        s = ckpt[qid]
        if s.get("failed"):
            rows.append({"id": qid, "failed": True})
            continue
        item, r, j = eval_set[qid], s["rule"], s.get("judge", {})
        a = s["attribution"]
        arb = (arbitration or {}).get(qid, {})
        rows.append({
            "id": qid, "type": s["type"], "subtype": s["subtype"],
            "difficulty": item.get("difficulty", ""), "origin": item.get("origin", ""),
            "question": item["question"],
            "gold_doc_id": "|".join(item["gold_doc_id"]),
            "retrieval_hit_top3": r["retrieval_hit_top3"],
            "retrieval_rank": r["retrieval_rank"],
            "refusal_accuracy": r["refusal_accuracy"],
            "faithfulness": _v(j, "faithfulness") or "-",
            "factuality": _v(j, "factuality") or "-",
            "completeness": _v(j, "completeness") or "-",
            "judge_avg": s.get("judge_avg", "-"),
            "latency_total_s": r["latency_total_s"],
            "wrong": bool(a["errors"]),
            "attribution_layer": a["layer"],
            "attribution_layer_final": arb.get("final_layer", a["layer"]),
            "attribution_reason": a["reason"],
            "attribution_note": arb.get("note", ""),
            "errors": "|".join(a["errors"]),
            "judge_disagreements": sum(1 for x in j.values() if not x["stable"]),
            "needs_review": a["needs_review"],
        })
    return rows


def compute_stats(rows: list[dict]) -> dict:
    ok_rows = [r for r in rows if not r.get("failed")]
    n = len(ok_rows)
    errors = [r for r in ok_rows if r["wrong"]]
    attr = Counter(r.get("attribution_layer_final", r["attribution_layer"])
                   for r in errors)
    by_type = {}
    for t in ("fact", "reasoning", "refusal", "multi_doc"):
        sub = [r for r in ok_rows if r["type"] == t]
        by_type[t] = {
            "n": len(sub),
            "correct": sum(1 for r in sub if not r["wrong"]),
            "error_rate": round(sum(1 for r in sub if r["wrong"]) / len(sub), 3)
            if sub else None,
            "judge_avg": round(statistics.mean(
                [r["judge_avg"] for r in sub if r["judge_avg"] != "-"]), 2)
            if any(r["judge_avg"] != "-" for r in sub) else None,
        }
    non_refusal = [r for r in ok_rows if r["type"] != "refusal"]
    refusal_rows = [r for r in ok_rows if r["type"] == "refusal"]
    lats = [r["latency_total_s"] for r in ok_rows]

    def _rate(num: int, den: int):
        return round(num / den, 3) if den else None

    stats = {
        "n": n,
        "n_correct": n - len(errors),
        "n_error": len(errors),
        "error_rate": round(len(errors) / n, 3),
        "attribution_counts": dict(attr),
        "attribution_pct_of_errors": {k: round(v / len(errors), 3) for k, v in attr.items()}
        if errors else {},
        "attribution_coverage": f"{sum(attr.values())}/{len(errors)}",
        "by_type": by_type,
        "metrics": {
            "retrieval_hit_top3_rate": _rate(sum(
                1 for r in non_refusal if r["retrieval_hit_top3"]), len(non_refusal)),
            "refusal_accuracy_rate": _rate(sum(
                1 for r in refusal_rows if r["refusal_accuracy"]), len(refusal_rows)),
            "factuality_correct_rate": _rate(sum(
                1 for r in ok_rows if r["factuality"] == "correct"),
                sum(1 for r in ok_rows if r["factuality"] != "-")),
            "factuality_partial_rate": _rate(sum(
                1 for r in ok_rows if r["factuality"] == "partial"),
                sum(1 for r in ok_rows if r["factuality"] != "-")),
            "faithfulness_supported_rate": _rate(sum(
                1 for r in ok_rows if r["faithfulness"] == "supported"),
                sum(1 for r in ok_rows if r["faithfulness"] != "-")),
            "completeness_complete_rate": _rate(sum(
                1 for r in ok_rows if r["completeness"] == "complete"),
                sum(1 for r in ok_rows if r["completeness"] != "-")),
        },
        "judge_disagreement_items": sum(1 for r in ok_rows if r["judge_disagreements"]),
        "latency_total_s_mean": round(statistics.mean(lats), 2) if lats else None,
    }
    return stats


def finalize(ckpt: dict | None = None) -> None:
    if ckpt is None:
        ckpt = load_checkpoint()
    arbitration = {}
    if ARBITRATION.exists():
        arbitration = json.loads(ARBITRATION.read_text(encoding="utf-8"))
    rows = build_rows(ckpt, arbitration)
    stats = compute_stats(rows)

    RESULTS_JSON.write_text(json.dumps(
        {"meta": {"day": "D5", "date": time.strftime("%Y-%m-%d"),
                  "pipeline": "rag_pipeline.py（D3）", "scoring": "scoring.py（D4）",
                  "judge_runs": JUDGE_RUNS, "top_k": 8,
                  "arbitration": "人工复核仲裁见 d5_arbitration.json 与 D5_跑批与归因说明.md",
                  "attribution_tree": "retrieval / generation / data（见 D5_跑批与归因说明.md）"},
         "stats": stats,
         "results": [dict(ckpt[r["id"]],
                          attribution_final=r.get("attribution_layer_final"))
                     for r in rows]},
        ensure_ascii=False, indent=2), encoding="utf-8")

    cols = ["id", "type", "subtype", "difficulty", "origin", "question", "gold_doc_id",
            "retrieval_hit_top3", "retrieval_rank", "refusal_accuracy", "faithfulness",
            "factuality", "completeness", "judge_avg", "latency_total_s", "wrong",
            "attribution_layer", "attribution_layer_final", "attribution_reason",
            "attribution_note", "errors", "judge_disagreements", "needs_review"]
    with open(RESULTS_CSV, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    print_stats(stats, rows)


def print_stats(stats: dict, rows: list[dict]) -> None:
    print("\n" + "═" * 62)
    print(f"D5 跑批统计（{stats['n']} 条）")
    print(f"  正确 {stats['n_correct']} ｜ 错误 {stats['n_error']}"
          f"（错误率 {stats['error_rate']:.0%}）")
    n_err = stats["n_error"]
    covered = sum(stats["attribution_counts"].values())
    cov = (f"归因覆盖：{covered}/{n_err}"
           + (" ✅ 100%（当前无错误样本）" if n_err == 0 else
              " ✅ 100%" if covered == n_err else " ⚠️ 存在未归因错误"))
    print(cov)
    print("  各层错误占比：")
    for k, v in stats["attribution_counts"].items():
        print(f"    {k:<11} {v} 条（占错误 {stats['attribution_pct_of_errors'][k]:.0%}）")
    print("  分题型错误率：" + " ｜ ".join(
        f"{t} {d['error_rate']:.0%}" for t, d in stats["by_type"].items()
        if d["error_rate"] is not None))
    print("  关键指标：" + " ｜ ".join(
        f"{k}={v}" for k, v in stats["metrics"].items() if v is not None))
    print(f"  judge 分歧条目：{stats['judge_disagreement_items']}"
          f" ｜ 平均延迟 {stats['latency_total_s_mean']}s")


def cmd_review() -> None:
    """导出人工复核清单：judge 分歧 + partial 弱问题 + 数据层判定 + 边界 case。"""
    ckpt = load_checkpoint()
    eval_set = {e["id"]: e for e in load_eval_set()}
    rows = []
    for qid in sorted(ckpt):
        s = ckpt[qid]
        if s.get("failed"):
            continue
        j = s.get("judge", {})
        a = s["attribution"]
        dis = [f"{m}:{'/'.join(x['disagreement'])}" for m, x in j.items() if not x["stable"]]
        partial = [m for m in ("factuality", "faithfulness", "completeness")
                   if _v(j, m) == "partial"]
        review_reasons = []
        if dis:
            review_reasons.append(f"裁判分歧[{';'.join(dis)}]")
        if partial:
            review_reasons.append(f"partial弱问题{partial}")
        if a["layer"] == "data":
            review_reasons.append("数据层判定复核")
        if a["layer"] == "correct" and a["errors"]:
            review_reasons.append("errors非空但判correct")
        if a["layer"] == "generation" and s["rule"]["retrieval_hit_top3"] is False:
            review_reasons.append("检索/生成边界")
        if review_reasons:
            rows.append({
                "id": qid, "type": s["type"], "subtype": s["subtype"],
                "question": eval_set[qid]["question"],
                "gold_answer": eval_set[qid]["gold_answer"],
                "answer": s["record"]["answer"][:400],
                "attribution_layer": a["layer"],
                "attribution_reason": a["reason"],
                "review_reason": "；".join(review_reasons),
                "manual_verdict": "",  # 人工仲裁后回填
            })
    with open(REVIEW_CSV, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else
                          ["id", "review_reason"])
        w.writeheader()
        w.writerows(rows)
    print(f"人工复核清单：{len(rows)} 条 → {REVIEW_CSV.name}")


def cmd_arbitrate() -> None:
    """应用人工仲裁（d5_arbitration.json）→ 重算最终统计并导出 d5_results.*。

    d5_arbitration.json 每项：{qid: {"final_layer": ..., "note": ...}}
    未出现在仲裁文件中的条目维持自动归因结果。
    """
    if not ARBITRATION.exists():
        sys.exit(f"未找到 {ARBITRATION.name}，先完成人工复核")
    finalize()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="D5 全量跑批 + 三层归因")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_run = sub.add_parser("run", help="全量跑批（checkpoint 续跑）")
    p_run.add_argument("--limit", type=int, help="只跑前 N 条（冒烟）")
    p_run.add_argument("--fresh", action="store_true", help="清空 checkpoint 重跑")
    sub.add_parser("stats", help="重算统计并导出（不调 LLM）")
    sub.add_parser("review", help="导出人工复核清单 CSV")
    sub.add_parser("arbitrate", help="应用人工仲裁并导出最终结果")
    args = ap.parse_args()

    if args.cmd == "run":
        cmd_run(args.limit, args.fresh)
    elif args.cmd == "stats":
        finalize()
    elif args.cmd == "review":
        cmd_review()
    elif args.cmd == "arbitrate":
        cmd_arbitrate()


if __name__ == "__main__":
    main()
