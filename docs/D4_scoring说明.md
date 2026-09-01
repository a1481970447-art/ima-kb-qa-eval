# D4 交付说明：双层评分器（scoring.py）

> 对应 7 天计划 D4 ｜ 产出：`scoring.py`（`demo` 子命令，运行后生成 d4_demo_results.json 演示样本） ｜ 验收：6 条样本跑通、裁判一致性 ≥ 2/3 ✅（实际 6/6 全部 3/3 完全一致）
> 执行日期：2026-08-28 ｜ 上游：`rag_pipeline.py`（D3）· `eval_set.json`（D2）｜ 下游：D5 全量跑批 + 归因

## 一、验收结果

| 验收项 | 标准 | 结果 |
|---|---|---|
| 样本跑通 | ≥ 5 条端到端（RAG 生成 + 双层评分） | ✅ 6 条（四类题型全覆盖，274s） |
| 裁判一致性 | majority ≥ 2/3 | ✅ 6/6 个指标全部 3/3 完全一致（超过标准） |
| 规则层零 LLM 调用 | 命中率/拒答/延迟纯规则计算 | ✅ 正则 + 集合运算，可离线复算 |
| judge prompt 独立 | 裁判与生成 prompt 分离 | ✅ 独立 system + 三维度专用 user prompt |
| 分歧记录 | agreement < 3 时单独留档 | ✅ 机制就绪（本次 demo 无分歧样本） |

验收样本（覆盖全部 5 个指标 × 4 类题型）：

| 样本 | 题型 | 考核指标 | 规则层 | 裁判层 |
|---|---|---|---|---|
| F-01 | fact·数据冲突 | hit + factuality | 命中 top3 rank=2（错误源 D08 仍排第 1） | factuality=correct 3/3 |
| F-02 | fact·细节埋深 | hit + factuality + completeness | 命中 rank=1 | correct 3/3 + complete 3/3 |
| R-10 | reasoning·计算 | hit + factuality | 命中 rank=1 | correct 3/3 |
| R-15 | reasoning·观点引用 | hit + faithfulness | 命中 rank=1 | supported 3/3 |
| J-01 | refusal·内部参数陷阱 | refusal_accuracy | **正确拒答**（top3 全是 D19/D16 论文，未中招） | —（规则层） |
| M-05 | multi_doc·跨文验证 | hit + factuality | 命中 rank=1 | correct 3/3 |

## 二、指标体系实现（对应方案第三节）

### 规则层（自动，零成本）

| 指标 | 实现 |
|---|---|
| 检索命中率 | 金标文档进入 retrieved 的 doc_id 集合；双口径 `retrieval_hit`（top-k）/ `retrieval_hit_top3`，附 `retrieval_rank`（金标最高排名，D5 归因用：rank>3 的命中是弱命中） |
| 拒答准确率 | 双向检测：拒答题应拒答（6 组正则匹配拒答话术）＆ 非拒答题不应错误拒答 |
| 端到端延迟 | 直接取 record 的 `latency_retrieve/generate/total_s` |

### 模型层（LLM-as-judge，独立 prompt）

| 维度 | 判什么 | verdict → 分值 |
|---|---|---|
| faithfulness | 回答论断是否由检索上下文支撑（防幻觉，RAGAS 口径） | supported/partial/unsupported → 2/1/0 |
| factuality | 关键事实 vs 金标答案（数字/日期/名称/结论） | correct/partial/incorrect → 2/1/0 |
| completeness | 金标要点覆盖率 | complete/partial/missing → 2/1/0 |

- 每指标跑 **3 次取多数**（temperature 0.7，高于生成用的 0.2——测的是裁判稳定性而非确定性复读），agreement<3/3 记入 `disagreements` 留待 D5 人工仲裁（方案第七节风险对策落地）
- 每条题按 eval_set 的 `metrics` 字段裁剪要跑的裁判维度，不无差别全跑

### 裁判 prompt 的反偏差设计（D15 理论落地）

1. **冗长偏差**：system 显式声明"回答长短不影响打分"
2. **自我增强偏差**：裁判与被测模型同为 hy3——majority vote 缓解，且这是复现 ima 同模型自评的现实约束（报告须声明）
3. **知识泄漏**：system 声明"不使用你自己的产品知识，只依据评测材料"
4. **输出约束**：只输出一行 JSON（verdict/reason），正则提取，容错思考模型包裹文本

## 三、验收过程中的一个设计 bug 与修复（本身是 demo 素材）

初版 faithfulness 裁判只喂 top-5 chunks，而生成用的是完整 top-k——R-15 的回答合法引用了 top-8 内的 D12#013，却被误判"引用未提供的内容"得 partial。修复：**忠实度判定基准必须与被测生成过程上下文完全一致**（改为全量 retrieved）。修复后 R-15 = supported 3/3。

> 这正是 LLM-as-judge 工程化的典型陷阱：裁判证据面 ≠ 被测证据面 → 系统性低估忠实度。写进报告的"评测体系构建踩坑"小节。

## 四、验收样本里的归因预演（喂给 D5）

- **F-01**：错误源 D08（"今年 2 月上线"错误帖）仍排检索第 1、金标 D03/D02 排 2/3——D1 的 A-01 badcase 在复现 pipeline 中完整复现。生成层最终答对 → "检索层埋雷、生成层兜住"，D5 记检索弱命中（rank>1 的信源排序问题），归因焦点在数据层信源质量。
- **F-02**：D1 实测金标 chunk 未召回（A-03），本 pipeline rank=1 命中 D02#015——切块策略（滑窗 600/120 + 章节前缀）修复了部分细节埋深问题，构成与 ima 实测的对照差异。
- **J-01**：陷阱文档 D19 的 chunk size 表占据 top3，生成层仍正确拒答并说明"库里有什么"——"有相关内容 ≠ 有答案"的边界行为成立，拒答准确率测的是生成层而非检索层。
- **R-10**：检索命中金标 D11 + 计算过程正确——检索/生成两层分离的天然归因样本。

## 五、对 D5 的接口约定

```python
from rag_pipeline import RagPipeline
from scoring import score_item
pipe = RagPipeline()
record = pipe.answer(item["question"])        # D3 结构化输出
scored = score_item(item, record)             # → {rule: {...}, judge: {...}, judge_avg}
# 分歧样本自动落 disagreements 字段，D5 人工仲裁后写入最终归因
```

成本口径：6 条验收消耗 RAG 生成 6 次 + 裁判 18 次调用，约 4.5 分钟。外推 D5 全量 50 条 ≈ 150 次裁判调用 + 50 次生成，预计 30–40 分钟、API 费用 < 10 元（在方案"总成本 < 50 元"预算内）。

## 六、已知限制

| 项 | 说明 |
|---|---|
| 裁判与被测同模型 | hy3 自评存在自我增强偏差风险（D15），majority vote 缓解但未消除；报告声明，并保留 10% 人工抽检 |
| 拒答正则的覆盖面 | 6 组正则覆盖 D3 内置话术 + 常见改写；极不规范表述可能漏检，D5 对 refusal 题保留人工复核 |
| partial/incorrect 的分值粒度 | 0/1/2 三档粗粒度——有意为之：demo 卖点是归因体系不是精细细分数 |

## 七、复现命令

```bash
python scoring.py demo                     # 6 条验收（默认 F-01,F-02,R-10,R-15,J-01,M-05）
python scoring.py demo --ids F-01,J-01     # 自选样本
python scoring.py demo --no-llm            # 只跑规则层（快速自检）
python scoring.py demo --ids R-15          # 单条复验（修复 faithfulness 后）
```

---
*D4 · 2026-08-28 · 上游：rag_pipeline.py + eval_set.json · 下游：D5 全量跑批 + badcase 归因*
