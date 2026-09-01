# -*- coding: utf-8 -*-
"""
D6 · ima 知识库问答质量评测 —— Streamlit 看板
一页看板：总分 / 分层得分 / 分数分布 / badcase 列表（按归因标签筛选）

运行：
    streamlit run dashboard.py

数据来源（均为 D5 产出，只读）：
    d5_results.json      全量结果 + 明细（回答、召回 chunks）
    d5_arbitration.json  人工复核仲裁结论
    eval_set.json        金标评测集（金标答案、出处摘录）
"""
import json
from pathlib import Path

import pandas as pd
import streamlit as st
import altair as alt

BASE = Path(__file__).resolve().parent

# ---------- 数据加载（st.cache_resource 只读磁盘，无副作用） ----------
@st.cache_data(show_spinner=False)
def load_data():
    with open(BASE / "d5_results.json", encoding="utf-8") as f:
        results = json.load(f)
    with open(BASE / "d5_arbitration.json", encoding="utf-8") as f:
        arbitration = json.load(f)
    with open(BASE / "eval_set.json", encoding="utf-8") as f:
        eval_set = json.load(f)
    gold = {item["id"]: item for item in eval_set["eval_set"]}
    return results, arbitration, gold


results, arbitration, gold = load_data()
stats = results["stats"]
rows = results["results"]

# ---------- 页面 ----------
st.set_page_config(page_title="ima 评测工作台", page_icon="📊", layout="wide")
st.title("📊 ima 知识库问答质量评测工作台")
st.caption(
    f"金标评测集 {stats['n']} 条 · LLM 裁判每条 3 次取多数 · 检索 top_k={results['meta']['top_k']} · "
    f"跑批日期 {results['meta']['date']} · 归因覆盖率 {stats['attribution_coverage']}"
)

# ---------- 1. KPI 总分 ----------
st.subheader("总分")
k1, k2, k3, k4 = st.columns(4)
k1.metric("端到端正确率", f"{1 - stats['error_rate']:.0%}", f"{stats['n_correct']}/{stats['n']} 条正确")
k2.metric("错误样本", f"{stats['n_error']} 条", "100% 完成三层归因")
k3.metric("裁判分歧率", f"{stats['judge_disagreement_items']} 条", "5 处分歧全部人工仲裁")
k4.metric("平均端到端延迟", f"{stats['latency_total_s_mean']:.1f} s", "检索 + 生成")

st.divider()

# ---------- 2. 分层指标（规则层自动 + 模型层裁判） ----------
st.subheader("分层指标")
m = stats["metrics"]


def _fmt(v):
    return "—" if v is None else f"{v:.1%}"


metric_rows = [
    ("检索层", "检索命中率（金标文档进 top-3）", _fmt(m["retrieval_hit_top3_rate"]), "规则，自动"),
    ("边界能力", "拒答准确率（库外问题不乱编）", _fmt(m["refusal_accuracy_rate"]), "规则，自动"),
    ("生成层", "事实正确性（factuality=correct）", _fmt(m["factuality_correct_rate"]), "LLM-as-judge ×3"),
    ("生成层", "事实部分正确率（partial）", _fmt(m["factuality_partial_rate"]), "LLM-as-judge ×3"),
    ("生成层", "引用忠实度（faithfulness=supported）", _fmt(m["faithfulness_supported_rate"]), "LLM-as-judge ×3"),
    ("生成层", "完整性（completeness=complete）", _fmt(m["completeness_complete_rate"]), "LLM-as-judge ×3"),
]
st.dataframe(
    pd.DataFrame(metric_rows, columns=["层级", "指标", "得分", "测量方式"]),
    width="stretch",
    hide_index=True,
)

# ---------- 3. 图表区：分题型得分 / 分数分布 / 归因占比 ----------
c1, c2, c3 = st.columns(3, gap="large")

with c1:
    st.markdown("**分题型错误率**")
    by_type = pd.DataFrame(
        [
            {"题型": t, "条数": d["n"], "错误率": d["error_rate"], "裁判均分": d["judge_avg"]}
            for t, d in stats["by_type"].items()
        ]
    )
    type_name = {"fact": "事实型", "reasoning": "推理型", "refusal": "拒答型", "multi_doc": "多文档关联"}
    by_type["题型"] = by_type["题型"].map(type_name)
    st.dataframe(by_type, hide_index=True, width="stretch")
    st.bar_chart(by_type.set_index("题型")["错误率"], y_label="错误率")
    st.caption("多文档关联型错误率 60% —— 知识库问答的“多源整合”是最弱项")

with c2:
    st.markdown("**裁判分数分布（40 条非拒答题）**")
    judged = [r for r in rows if r.get("judge_avg") is not None]
    dist = pd.DataFrame(
        [{"分数": r["judge_avg"]} for r in judged]
    )
    dist_chart = (
        alt.Chart(dist)
        .mark_bar(cornerRadiusEnd=3)
        .encode(
            x=alt.X("分数:O", sort=[0, 0.5, 1, 1.5, 2], title="judge_avg（0=错 … 2=全对）"),
            y=alt.Y("count()", title="条数"),
            color=alt.value("#2C6FBB"),
        )
        .properties(height=220)
    )
    st.altair_chart(dist_chart, width="stretch")
    st.caption("双峰分布：全对 2 分占主体，0 分长尾即 badcase 池")

with c3:
    st.markdown("**9 条错误的归因占比**")
    attr = stats["attribution_counts"]
    layer_name = {"retrieval": "检索层", "generation": "生成层", "data": "数据层"}
    attr_df = pd.DataFrame(
        [{"归因层": layer_name[k], "条数": v, "颜色": c}
         for (k, v), c in zip(attr.items(), ["#2C6FBB", "#E8893B", "#C44E4E"])]
    )
    pie = (
        alt.Chart(attr_df)
        .mark_arc(innerRadius=55, outerRadius=90)
        .encode(
            theta="条数",
            color=alt.Color("归因层", scale=alt.Scale(
                domain=["检索层", "生成层", "数据层"], range=["#2C6FBB", "#E8893B", "#C44E4E"])),
            tooltip=["归因层", "条数"],
        )
        .properties(height=220)
    )
    st.altair_chart(pie, width="stretch")
    st.caption(
        "错误分布：检索层 44% / 生成层 44% / 数据层 12%"
        "——多文档题 M-04 证明数据层脏数据会真实触发，对应产品解法是知识库健康度引导"
    )

st.divider()

# ---------- 4. Badcase 列表（核心交互：按归因标签筛选） ----------
st.subheader("Badcase 列表 · 按归因标签筛选")

f1, f2 = st.columns([1, 2])
with f1:
    layer_filter = st.multiselect(
        "归因层（人工仲裁后终判）",
        options=["retrieval", "generation", "data"],
        default=[],
        format_func=lambda x: {"retrieval": "检索层", "generation": "生成层", "data": "数据层"}[x],
        help="默认展示全部 9 条错误；选标签则只看某一层",
    )
with f2:
    type_filter = st.multiselect(
        "题型",
        options=["fact", "reasoning", "multi_doc"],
        default=[],
        format_func=lambda x: {"fact": "事实型", "reasoning": "推理型", "multi_doc": "多文档关联"}[x],
    )

layer_color = {"retrieval": "#2C6FBB", "generation": "#E8893B", "data": "#C44E4E"}

bad = [r for r in rows if r["attribution_final"] != "correct"]
if layer_filter:
    bad = [r for r in bad if r["attribution_final"] in layer_filter]
if type_filter:
    bad = [r for r in bad if r["type"] in type_filter]

st.markdown(f"共 **{len(bad)}** 条 badcase")

for r in bad:
    layer = r["attribution_final"]
    arb = arbitration.get(r["id"], {})
    g = gold.get(r["id"], {})
    title = f"`{r['id']}` {g.get('question', '')}"
    header = st.container(border=True)
    with header:
        tcol1, tcol2, tcol3 = st.columns([6, 1.2, 1.2])
        tcol1.markdown(f"**{title}**")
        tcol2.markdown(
            f"<span style='background:{layer_color[layer]}22;color:{layer_color[layer]};"
            f"padding:2px 10px;border-radius:10px;font-size:13px'>"
            f"{ {'retrieval': '检索层', 'generation': '生成层', 'data': '数据层'}[layer] }</span>",
            unsafe_allow_html=True,
        )
        tcol3.markdown(f"`{ {'fact': '事实型', 'reasoning': '推理型', 'refusal': '拒答型', 'multi_doc': '多文档'}[r['type']] }`")

        auto_layer = r["attribution"]["layer"]
        if auto_layer != layer:
            st.markdown(
                f"🧑‍⚖️ **人工改判：{auto_layer} → {layer}**（自动归因树误判，复核后修正）"
            )

        colA, colB = st.columns(2)
        with colA:
            st.markdown("**🧭 归因结论**")
            st.markdown(f"- 自动归因：{r['attribution']['reason'] or '—'}")
            if arb.get("note"):
                st.markdown(f"- 人工复核：{arb['note']}")
            if r["attribution"].get("errors"):
                st.markdown(f"- 错误类型：`{'` `'.join(r['attribution']['errors'])}`")
        with colB:
            st.markdown("**📐 规则指标**")
            rule = r["rule"]
            st.markdown(
                f"- 金标文档 top-3 命中：{'✅' if rule.get('retrieval_hit_top3') else '❌'}"
                f"（rank = {rule.get('retrieval_rank') if rule.get('retrieval_rank') is not None else '未召回'}）"
            )
            st.markdown(
                f"- 拒答判定：{'✅ 正确' if rule.get('refusal_accuracy') else '❌ 错误拒答'}"
            )
            st.markdown(f"- 端到端延迟：{rule.get('latency_total_s', '—')} s")

        with st.expander("🔍 查看实际回答 vs 金标答案 vs 召回明细"):
            st.markdown("**实际回答**")
            st.info(r["record"]["answer"])
            st.markdown("**金标答案**")
            st.success(g.get("gold_answer", "—"))
            if g.get("gold_source_quote"):
                st.caption(f"金标出处：{g.get('gold_doc_id')} · {g.get('source_section', '')}")
            st.markdown("**召回 chunks（top-5）**")
            rec = r["record"].get("retrieved", [])[:5]
            st.table(
                pd.DataFrame(
                    [
                        {"chunk": c["chunk_id"], "章节": c["section"], "相似度": round(c["score"], 4)}
                        for c in rec
                    ]
                )
            )

st.divider()

# ---------- 5. 声明 ----------
with st.container(border=True):
    st.markdown(
        "**📌 声明与已知局限**\n\n"
        "1. 本看板基于**复现架构**（手写 RAG pipeline，贴近 ima 可观察行为），结论用于方法论演示，"
        "而非对 ima 真实实现的定量指控。\n"
        "2. 归因层级为人工仲裁后终判（d5_arbitration.json）；2 条样本发生改判，说明纯自动归因树"
        "存在 22% 的误判率——这本身就是「评测需要人机协同」的证据。\n"
        "3. 已知口径局限（F-07 教训）：检索命中率按**文档级**统计会高估检索质量，"
        "章节级口径待下一版评测集补充标注。\n"
        "4. LLM 裁判每条跑 3 次取多数，分歧样本 5 处全部人工仲裁，无一例翻转。"
    )
