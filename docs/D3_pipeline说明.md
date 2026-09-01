# D3 交付说明：RAG 复现 pipeline（rag_pipeline.py）

> 对应 7 天计划 D3 ｜ 产出：`rag_pipeline.py` + `rag_index/`（20 篇 / 997 chunks 向量索引）｜ 验收：单条问答端到端跑通 ✅
> 执行日期：2026-08-27 ｜ 上游：`eval_set.json`（D2）· `corpus/` D01–D20 ｜ 下游：D4 scoring.py · D5 归因

## 一、验收结果

| 验收项 | 标准 | 结果 |
|---|---|---|
| 端到端跑通 | 解析→切块→embedding→检索→生成 单条问答完整链路 | ✅（延迟 ~10–24s/条，检索 <4s，其余为 hy3 思考生成） |
| 零框架依赖 | 手写最简 pipeline，无 LangChain | ✅ 仅 openai SDK + numpy + pypdf |
| 腾讯叙事 | 模型/向量均走混元 TokenHub | ✅ hy3（生成）+ kinfra-text-embedding-0.6b（向量化） |
| top-k 可配置 | CLI `-k` / 环境变量 `RAG_TOP_K` | ✅ 已演示 k=8 与 k=4 |
| eval_set 抽测 | 4 类题型各抽 1 条 | ✅ F-01/F-03/R-10 检索命中 + J-01 正确拒答 |
| 索引可增量 | 文档不变不重算向量 | ✅ 二次 build 直接命中缓存 |

## 二、链路设计（贴近 ima 可观察行为的三个决策）

```
corpus/ D01–D20 ──► 解析 ──► 切块 ──► embedding ──► 向量检索 ──► 生成
  md: 按标题切节     标题前缀     混元 embed      cosine top-k    hy3 + 引用约束
  pdf: pypdf 按页    +滑窗 600/120  (批32,缓存)   (k 可配置)      (拒答规则内置)
```

1. **front-matter 物理隔离**：解析时剥离 YAML 头——直接落实 D1 教训 4（内部元数据被索引进检索池污染答案，A-14）。
2. **chunk 携带「文档·章节」标题前缀 + 每篇一个摘要 chunk**：复现 D1 观察到的 ima 行为——高亮多为"文档摘要/节标题"而非答案段落。代价与收益同源：标题词可命中（F-03 金标 chunk 排第 1），但也复现了"仅标题匹配挤占排序"的弱点。
3. **生成 prompt 内置拒答规则**（"资料不足时明确回答没找到"）：为 D2 拒答题（J 类 10 条）提供公平的拒答机会，拒答准确率才能测出生成层真实水平而非 prompt 缺失。

## 三、抽测亮点（D5 归因的预演）

- **F-01（错源高排完整复现）**：错误帖 D08 排检索第一（0.654），权威 D02 第三——与 D1 实测 A-01 一致。生成层最终答对，但这正是"检索层埋雷、生成层兜底"的典型样本：D5 按 trap_note 归因时，检索层命中率该题应记 hit（金标进 top-3），归因焦点在信源排序。
- **F-03（对照组正向）**：金标 chunk 直接排第 1（0.711），滑窗切块把"细节埋深"文档的算力条款完整切出——对照组不翻车，命中率差异才可归因。
- **R-10（计算题归因分离器）**：检索命中 D11 金标 + 回答给出计算过程——检索/生成两层表现分离，D5 优先看这类题。
- **J-01（拒答陷阱）**：D19 的 chunk size 表格如 D2 预埋被召回（top-3 全是 D19/D16），生成层仍正确拒答——"有相关内容 ≠ 有答案"的边界成立。

## 四、对 D4/D5 的接口约定

```python
from rag_pipeline import RagPipeline
pipe = RagPipeline()
record = pipe.answer(question, top_k=8)
# record = {question, answer, retrieved:[{chunk_id, doc_id, section, score, text}],
#           top_k, latency_retrieve_s, latency_generate_s, latency_total_s}
```

- D4 规则层：`retrieved` 中 `doc_id ∈ gold_doc_id` 判检索命中；`latency_*` 即端到端延迟指标；拒答题只看 answer 是否拒答。
- D4 裁判层：answer 原文 + eval_set 的 gold_answer / gold_source_quote 进 judge prompt。
- D5 归因：检索命中 + 答错 → 生成层；检索未命中 → 检索层；命中陷阱文档（trap_note 标注的 D08/D10/D11/D12/D05）→ 数据层。

## 五、已知限制（写进评测报告的声明）

| 项 | 说明 |
|---|---|
| 复现 ≠ ima 实现 | 本 pipeline 是贴近 ima 可观察行为的复现架构，结论用于方法论演示，非对 ima 的定量指控 |
| PDF 双栏乱序 | pypdf 对英文论文双栏提取顺序有乱序，与 D1 观察到的 ima 英文 PDF 高亮质量问题同源，跨语言题（R 类 6 条）结果解释时需注意 |
| hy3 思考延迟 | 生成 7–24s/条，50 条全量跑批约 15–20 分钟，D5 需容忍或并发 |

## 六、复现命令

```bash
python rag_pipeline.py build                    # 建索引（增量缓存，997 chunks）
python rag_pipeline.py ask "ima 是什么时候上线的？"   # 单条问答
python rag_pipeline.py ask -k 4 --json "..."     # top-k 可配置 + 结构化输出
python rag_pipeline.py smoke                    # eval_set 抽测 4 条
```

---
*D3 · 2026-08-27 · 上游：eval_set.json · 下游：D4 scoring.py + judge prompt*
