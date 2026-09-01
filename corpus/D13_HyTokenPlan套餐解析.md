---
doc_id: D13
title: Hy Token Plan 上线：28 元起、覆盖混元 Hy3 preview 的 4 档套餐解析
source: https://cloud.tencent.com/developer/article/2676264
publisher: 腾讯云开发者社区（作者 gavin1024，2026-05-29 发布）
retrieved: 2026-08-31
format: markdown（网页正文提取）
language: zh
quality_note: 数字密集（套餐/参数/价格），适合事实与计算型金标题
---

# Hy Token Plan 上线：28 元起、覆盖混元 Hy3 preview 的 4 档套餐解析

**摘要**：腾讯云 TokenHub 上新 Hy Token Plan，28 元/月起即可调用混元 Hy3 preview，4 档套餐覆盖从尝鲜到重度开发，原生 256K 上下文，按月预付费、Token 统一抵扣。

## 一、Hy Token Plan 是什么：一个套餐直通最新混元

Hy Token Plan 是腾讯云大模型服务平台 TokenHub 在 2026 年 4 月上线的混元专属订阅套餐。它的设计初衷很明确——把腾讯自研的最新一代混元大模型 Hy3 preview，以更易接受的月度预付费形式交付给个人开发者。

Hy3 preview 是混元当前最智能的语言模型：采用 295B 参数、21B 激活的 MoE 架构，原生支持 256K 上下文，具备深度思考（交错式思考）、结构化输出、Function Calling 与 Cache 缓存能力。最大输入 192K、最大输出 128K，能在单次对话中处理长文档、完整代码库与复杂工作流上下文。把这样规格的模型放进固定月费套餐，意味着开发者不需要再为每一次推理逐 Token 估价。

## 二、4 档套餐档位与定价全景

Hy Token Plan 当前提供 4 个档位，月度 Token 限额从 3500 万一路覆盖到 6.5 亿，对应人群从首次尝鲜到把 AI 当核心生产力的重度用户：

| 套餐档位 | 月度 Token 限额 | 活动单价 | 覆盖模型 | 适配人群 |
|---|---|---|---|---|
| 体验套餐（Lite） | 3500 万 Tokens | 28 元/月 | Hy3 preview | 首次体验，约 70 轮问答 |
| 基础套餐（Standard） | 1 亿 Tokens | 78 元/月 | Hy3 preview | 日常使用，约 200 轮问答 |
| 进阶套餐（Pro） | 3.2 亿 Tokens | 238 元/月 | Hy3 preview | 每天高频使用 AI 的开发者 |
| 专业套餐（Max） | 6.5 亿 Tokens | 468 元/月 | Hy3 preview | 重度生产力用户 |

### 2.1 价格梯度怎么看

Lite 28 元的入场门槛，把"想试试 Hy3 preview"的成本压到一杯咖啡的价位；Standard 78 元覆盖一个开发者一天数小时的 IDE 内 AI 协作；Pro 238 元的 3.2 亿 Tokens 配额已经能支持密集的 Agent 编排与长上下文阅读；Max 468 元则是面向把混元嵌入日常主力工作流的重度用户。

值得注意的是，Hy Token Plan 同档位价格普遍低于通用 Token Plan：以 Lite 为例，Hy 档 28 元对比通用档 39 元；Max 档 468 元对比通用档 599 元。这一价差来自混元自研模型的成本优势，也意味着如果你的工作流主要围绕 Hy3 preview，选 Hy Token Plan 比通用包更划算。

### 2.2 套餐价相比按量计费的差距

按 Hy3 preview 当前的在线推理价格，输入分段从 1.2 元/百万 Tokens 起、输出分段从 4 元/百万 Tokens 起，缓存命中输入低至 0.4 元/百万 Tokens。把 Pro 档 3.2 亿 Tokens / 238 元换算成百万 Tokens 单价约 0.74 元，相比直接调用文本生成服务低 50% 以上——这正是腾讯云官方对 Token Plan 价格优势的描述：同等用量，省超一半。

## 三、谁该选 Hy Token Plan：4 类典型场景

### 3.1 长文档处理与代码库理解

Hy3 preview 原生 256K 上下文，配合 192K 最大输入，能一次性吃下整本技术手册、几个微服务的完整代码或一份完整的产品需求文档。Pro 档 3.2 亿 Tokens 月度配额，约等于每月反复阅读分析数百份长文档而不必担心 Token 见底。

### 3.2 Agent 工作负载与多轮深度推理

Hy3 preview 支持深度思考（交错式思考）+ Function Calling，是 Agent 编排场景里的理想底座。Standard 78 元档大致可以支撑约 200 轮带工具调用的问答对话，足够覆盖一个开发者日常的 AI Pair Programming 节奏。

### 3.3 在 IDE 中重度使用 AI 编程工具

Hy Token Plan 与通用 Token Plan 共用同一套 API Key 与调用地址。这意味着在 CodeBuddy Code、OpenCode、Cline、Cursor、Claude Code 等编程工具里，把 Base URL 配置为 `https://api.lkeap.cloud.tencent.com/plan/v3` 或 Anthropic 协议地址，就能在 Hy3 preview 与其他主流模型之间自由切换，由 Model ID 决定从哪个套餐扣减。

### 3.4 高频 Function Calling 与结构化输出

Hy3 preview 同时支持结构化输出与 Function Calling，对需要严格 JSON Schema 输出、或频繁调用外部工具的场景非常友好；Cache 缓存又把重复 system prompt 与对话前缀的成本进一步压低。

## 四、开通与使用：从下单到首次调用

### 4.1 三步完成接入

a. 在 Token Plan 活动页选定档位、完成下单

b. 进入控制台 API Key 管理页面创建 API Key（Token Plan 个人版仅生成 1 个 API Key）

c. 在你常用的 AI 工具中替换 Base URL 与 API Key 即可调用

### 4.2 几条必须知道的规则

- **限购 1+1**：每个主账号最多同时持有 2 个 Token Plan，即 1 个通用 Token Plan + 1 个 Hy Token Plan，同一系列只能持有 1 个档位
- **统一抵扣**：缓存命中输入、缓存未命中输入、输出 Token 都从套餐包内统一扣减，不需要按类型分别核算
- **套餐有效期**：以自然月为单位，自购买当日起算；剩余 Token 不结转到下个月，过期后 API Key 同步失效
- **不支持降配**：可升配到更高档，不可降配；通用与 Hy 两条线均不支持退款
- **使用边界**：仅限在 AI 工具中使用，不可用于自动化脚本或非交互式批量调用——批量任务请走在线推理或批量任务计费

## 五、用前先 0 元试一遍：新人免费体验包

如果你还没决定先选哪一档，可以先把腾讯云账号开通后免费拿到的体验额度跑一遍：每个主账号一次性获得 Hy3 preview 100 万 Tokens 的免费体验额度，有效期 90 天，覆盖几乎所有主力模型；体验过程不需要预付费即可在控制台直接试用。把真实的工作流跑一轮，再回头判断 Lite、Standard、Pro、Max 哪一档更贴合自己的月度消耗节奏，决策会清晰得多。

## 六、和通用 Token Plan 怎么搭配

如果你的开发场景同时涉及混元 Hy3 preview 与 GLM、Kimi、MiniMax 等主流国产模型，最优组合是：1 个 Hy Token Plan + 1 个通用 Token Plan。两者共用同一套 API Key 与同一套调用地址，由请求里指定的 Model ID 决定从哪个套餐扣减；既能享受 Hy 档的混元低价，又能保留多模型工具链的灵活度。

## 七、写在最后

Hy Token Plan 的价值，不只是把 Hy3 preview 拉进了 28 元的入场价，更是给重度依赖混元长上下文与深度思考能力的开发者一个稳定的月度预算锚点。无论你正在做 Agent 编排、代码库重构，还是把混元嵌入文档与办公自动化场景，4 档定价都给出了清晰的升级阶梯。

现在就到 Token Plan 活动页 https://cloud.tencent.com/act/pro/tokenplan 选择适合你节奏的档位，或先到 TokenHub 控制台 https://console.cloud.tencent.com/tokenhub/ 领取新人免费体验额度，把 Hy3 preview 跑进自己的工作流里。
