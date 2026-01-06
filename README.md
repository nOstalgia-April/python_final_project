# 基于用户评论的商品推荐策略与情感分析（Digital Music 5-core）

## 1. 项目目标
1) 识别具有“口碑陷阱”的商品，并尽可能分析其共同问题  
2) 识别口碑最好的商品，并尽可能分析其受欢迎的主要特征  
3) 基于简单关联规则，为特定商品提供 2 个交叉销售推荐  

## 2. 数据说明（以 `Digital_Music_5.json` 为例）
典型字段：
- 商品：`asin`（商品唯一标识）、`style`（如 `Format:`）
- 用户：`reviewerID`、`reviewerName`
- 评分与文本：`overall`、`summary`、`reviewText`
- 时间：`unixReviewTime`（秒级时间戳，适合排序/时间窗统计）、`reviewTime`（可读字符串）
- 可信度/权重：`verified`（验证购买）、`vote`（有用票，可作为评论权重）

数据特点与约束：
- Digital Music 通常没有显式的“关联商品”（如 also_bought/also_viewed），第 3 项需要用“用户-商品共现”自行构造关联信号。

## 3. 总体架构（Pipeline）
### 3.1 数据层（Ingest & Clean）
- 读取 JSONL（逐行 JSON）→ 字段标准化（`asin/reviewerID/overall/reviewText/unixReviewTime`）→ 去重与缺失处理
-（可选）按 `asin` 关联元数据（若你有 metadata 文件）补全 `title/brand/category/price` 等特征

### 3.2 NLP/情感层（Sentiment & Topics）
- 文本清洗：大小写/标点/停用词、分词（中英文按各自规则）
- 情感得分：建议融合
  - 星级情感（强信号）：`overall` 直接映射正/负/中性
  - 文本情感（补充信号）：对 `reviewText/summary` 做情感分类或情感词典打分
- 主题/方面抽取：直接使用 Ask AI（大模型）对评论文本做归纳总结，输出“差评集中问题/好评卖点”
  - 输入：按 `asin` 聚合后的评论（可按星级分组：1–2 星差评、4–5 星好评），并限制每次输入的评论条数/总字数
  - 输出：Top 问题/卖点清单（可要求按“主题-证据句-出现频率/占比”结构化返回），用于后续商品画像与共性分析

### 3.3 商品画像层（Product Profiling，按 `asin` 聚合）
核心统计：
- 口碑：均分、样本量、方差/标准差、低分率（1–2 星占比）
- 趋势：按时间窗（如月/季度）统计均分与低分率变化，识别“近期变差”
- 可信度：`verified` 占比；`vote` 加权后的低分率/负向主题占比
文本画像：
- 正/负向高频主题（“受欢迎特征”“共同问题”）
- 典型评论样例（可选：按 `vote` 或情感强度挑选 Top-N）

## 4. 业务产出对应（3 个目标）
### 4.1 口碑陷阱识别（目标 1）
推荐判定思路（满足越多越可疑）：
- “表面高分”但分歧大：均分高 + 方差大/低分率不低
- 时间趋势恶化：近期低分率上升或均分下降
- 差评主题集中：负向主题/关键词（如“下载/兼容/授权/曲目缺失/音质压缩”等）在差评中高度集中
输出：
- 口碑陷阱商品清单（`asin` + 指标）+ 共同问题 Top 主题/关键词

### 4.2 口碑最好商品识别（目标 2）
推荐判定思路：
- 高均分 + 稳定：低方差、低低分率、趋势稳定或上升
- 好评卖点清晰：正向主题集中（少数主题覆盖大多数好评）
输出：
- Top-N 口碑最佳商品榜单 + 主要受欢迎特征（主题/关键词），并可按 `style.Format:` 做分组对比

#### “正向主题集中”如何落地（基于 Ask AI，三步法）
为控制成本并保证可复现，采用“先筛候选商品池 → AI 生成主题 → AI 按主题归类计数”的流程：

1) **候选商品池筛选（先用数值指标缩小范围）**
   - 先按 `asin` 聚合，设置最小评论数门槛（例如 `n_reviews >= 30/50`），避免样本太小导致主题不稳定
   - 从通过门槛的商品中选 Top-N 候选（例如 20/50 个）：
     - 高均分、低低分率、低方差（稳定）
     -（可选）趋势不变差：近期窗口均分不低于历史
   - 只对候选池调用 AI（避免对上万 `asin` 全量跑）

2) **Ask AI 生成主题（每个 asin 一套主题，但固定数量 K）**
   - 输入：同一 `asin` 的好评评论集合（建议 4–5 星），固定抽样条数 `N`（如 N=50/100），避免不同商品输入规模不一致
   - 输出：固定 `K` 个主题（例如 K=5 或 7）+ `Other`，每个主题给“短标签 + 定义 + 证据句”
   - 约束：相近主题需合并，避免同义拆分导致“看起来不集中”

3) **Ask AI 按主题归类并计数（用于计算“集中度”）**
   - 输入：步骤 2 的主题列表 + 同一批 N 条好评
   - 输出：每条评论只能归入 K+Other 中的一个主题，并汇总 `count` 与 `share=count/N`
   - 由 `share` 计算集中度（任选其一或组合）：
     - Top-3 覆盖率：`Top3_share = share1 + share2 + share3`（越高越集中）
     - 有效主题数：`N_eff = 1 / Σ(p_i^2)`（越小越集中）
     - 熵：`H = -Σ p_i log p_i`（越低越集中）

一致性建议（可选）：
- 同一 `asin` 变更抽样/顺序重复步骤 3，若 Top 主题占比波动过大，说明样本 N 太小或主题粒度过细，需要调大 N 或调小 K。

### 4.3 交叉销售推荐（目标 3：关联规则）
#### A. 构造 basket（解决“没有关联商品字段”的问题）
用评论行为构造“篮子”：
- **按用户聚合**：每个 `reviewerID` 的所有 `asin` 集合 = 1 个 basket
-（可选更像“同一段购买/兴趣”）**按时间窗切分 session**：对同一 `reviewerID`，按 `unixReviewTime` 在 30/90 天内分桶形成多个 basket

#### B. 规则挖掘与指标
从 basket 中挖规则 `A -> B`（用户拥有/评价过 A 时，也常出现 B）。本项目采用 **Apriori**：
- Apriori 核心：先找满足最小支持度的**频繁项集**，再由频繁项集生成规则并计算指标；利用“子集若不频繁，则其超集必不频繁”的先验性质做剪枝，减少候选。

指标定义（用于筛选与排序）：
- `support(A∩B)`：A 与 B 同时出现的频率（过滤偶然共现）
- `confidence(A->B) = P(B|A)`：命中率/可靠性
- `lift(A->B) = P(B|A) / P(B)`：去除“B 本身很热门”带来的虚高，`lift>1` 表示更强的真实关联

#### B.1 Apriori 的对齐实现方案（从业务到算法）
1) **数据对齐（basket 生成）**
   - 输入：每条评论的 `reviewerID`、`asin`、`unixReviewTime`
   - 生成：`basket_id -> {asin}`，其中 `basket_id` 可取
     - `reviewerID`（用户级 basket），或
     - `reviewerID + time_window`（会话级 basket，更贴近“同一段时间的交叉购买/兴趣”）
   - 清洗：同一 basket 内 `asin` 去重；过滤超大 basket（可选，避免“重度用户”支配共现）
2) **频繁项集挖掘（Apriori）**
   - 设定 `min_support_count`（或 `min_support`）过滤低频组合
   - 重点输出 **2-项集** `{A,B}`（项目交叉销售通常足够；也更稳定）
3) **规则生成**
   - 从 `{A,B}` 生成两条方向规则：`A -> B` 与 `B -> A`
   - 计算 `support/confidence/lift` 并保留满足阈值的规则
4) **推荐落地（每个目标商品给 2 个）**
   - 对指定商品 `A`：收集所有 `A -> B` 规则，按 `lift`（主）+ `confidence`（次）+ `support`（再次）排序，取 Top-2 的 `B`

#### B.2 参数建议（可在报告中说明为“经验阈值 + 数据驱动校准”）
- `min_support_count`：建议用“绝对次数”起步（例如 ≥5/10），再根据样本量调参
- `min_confidence`：避免低命中（例如 ≥0.2 或 ≥0.3）
- `lift`：建议保留 `>1` 的规则（排除纯热门效应）

#### C. 如何给“指定商品”推荐 2 个交叉销售
对目标商品 `A`：
1) 取所有前件包含 `A` 的候选规则  
2) 先过滤：`support` 不过小、`confidence` 不过低、`lift > 1`  
3) 排序：优先 `lift`，其次 `confidence`，再看 `support`  
4) 取后件里 Top-2 商品作为交叉销售推荐  

兜底策略（避免长尾商品没有规则）：
- 用“同一用户内的共现次数 / 余弦相似度”的 item-item 共现 Top-2 做推荐
-（若有 metadata）用同艺人/同流派/关键词相似 Top-2，再用口碑指标过滤

## 5. 最终交付物建议
- `口碑陷阱清单.csv`：`asin` + 指标 + 负向主题 Top-K
- `口碑最佳榜单.csv`：`asin` + 指标 + 正向主题 Top-K
- `交叉销售推荐.csv`：`target_asin` → `rec_asin_1/rec_asin_2` + `lift/confidence/support`

---

# Stack Overflow 调查（2023/2024/2025）：AI 工具趋势、态度变化、影响评估与用户画像

## 1. 数据与问卷来源（按年份）
- 2023：`stack-overflow-developer-survey-2023/survey_results_public.csv` + `stack-overflow-developer-survey-2023/survey_results_schema.csv` + `stack-overflow-developer-survey-2023/so_survey_2023.pdf`
- 2024：`stack-overflow-developer-survey-2024/survey_results_public.csv` + `stack-overflow-developer-survey-2024/survey_results_schema.csv` + `stack-overflow-developer-survey-2024/2024 Developer Survey.pdf`
- 2025：`stack-overflow-developer-survey-2025/survey_results_public.csv` + `stack-overflow-developer-survey-2025/survey_results_schema.csv` + `stack-overflow-developer-survey-2025/2025_Developer_Survey_Tool.pdf`

说明：
- 以每年的 `survey_results_schema.csv` 为"题目字典"，用 `qname → question` 确认题意并做跨年对齐；用 `survey_results_public.csv` 的列名/取值实现指数计算。
- `NA` 代表未回答/不适用（常见原因：未使用 AI 工具的人不会继续回答部分 AI 细题）。

## 2. 业务架构（Business → Analytics Pipeline）
1) **问卷对齐层**：按年读取 schema，建立"跨年题目映射表"（同名但题意变化的列必须用 `question` 校验）  
2) **指数构建层**：从"跨年可比题"优先构造核心指数；对仅部分年份存在的题构造扩展指数（标注适用年份）  
3) **趋势/变化分析层**：按年输出指数分布、均值/中位数、分组差异（开发者角色/工作年限/远程办公等）  
4) **影响评估层**：以 AI 使用强度/覆盖度为自变量，分析对"效率/学习/满意度"等的关联（分层对比 + 回归控制）  
5) **画像分群层**：用"AI 使用强度+态度/信任+场景覆盖+摩擦/挑战（有则用）"聚类；再用人口统计/职业变量做画像解释

## 3. 指数设计：问题来源（按年份）与计算方法
优先跨年可比：下面标注为 **[2023/2024/2025]** 的指数可用于三年趋势；仅部分年份存在的标注为 **[2024–2025]**、**[2025 only]**。

### 3.1 AI 使用趋势（Adoption & Coverage）
**(I1) AI 采纳强度指数 `AI_Adoption`（0–100）[2023/2024/2025]**
- 题目来源
  - 2023 `AISelect`：`Do you currently use AI tools in your development process? *`
  - 2024 `AISelect`：`Do you currently use AI tools in your development process? *`
  - 2025 `AISelect`：`Do you currently use AI tools in your development process?`
- 列名（public.csv）：`AISelect`
- 计算（建议映射，保证跨年可比）
  - 2023/2024：`Yes=100`；`No, but I plan to soon=30`；`No, and I don't plan to=0`
  - 2025：`Yes, I use AI tools daily=100`；`Yes, I use AI tools weekly=75`；`Yes, I use AI tools monthly or infrequently=40`；`No, and I don't plan to=0`

**(I2) AI 工作流覆盖指数（当前使用）`AI_WorkflowCoverage_Current`（0–100）[2023/2024/2025]**
- 题目来源
  - 2023 `AITool`：`Which parts of your development workflow are you currently using AI tools for and which are you interested in using AI tools for over the next year? Please select all that apply.`
  - 2024 `AITool`：同上（题意一致）
  - 2025 `AITool`：`Which parts of your development workflow are you currently integrating into AI or using AI tools to accomplish or plan to use AI to accomplish over the next 3 - 5 years? Please select one for each scenario.`
- 列名（public.csv）
  - 2023：`AIToolCurrently Using`（多选；`;` 分隔）
  - 2024：`AIToolCurrently Using`（多选；`;` 分隔）
  - 2025：`AIToolCurrently mostly AI` + `AIToolCurrently partially AI`（分别是场景列表；`;` 分隔）
- 跨年可比口径（优先取三年交集场景，样本扫描得到的交集为 8 个）
  - `Writing code`、`Testing code`、`Debugging and getting help`、`Committing and reviewing code`、`Deployment and monitoring`、`Documenting code`、`Learning about a codebase`、`Project planning`
- 计算
  - 2023/2024：覆盖数 = `AIToolCurrently Using` 中命中上述 8 场景的个数；指数 = 覆盖数 / 8 * 100
  - 2025：覆盖数 =（`AIToolCurrently mostly AI` 命中数 * 1.0）+（`AIToolCurrently partially AI` 命中数 * 0.5）；指数 = 覆盖数 / 8 * 100

**(I3) AI 工具使用广度指数 `AI_ToolBreadth`（0–100）[2023–2025]**
- 题目来源
  - 2023 `AISearch`：`Which AI-powered search tools did you use regularly over the past year, and which do you want to work with over the next year?`
  - 2023 `AIDev`：`Which AI-powered developer tools did you use regularly over the past year, and which do you want to work with over the next year?`
  - 2024 `AISearchDev`：`Which AI-powered search and developer tools did you use regularly over the past year, and which do you want to work with over the next year?`
  - 2025 `DevEnvs`：`Which development environments and AI-enabled code editing tools did you use regularly over the past year, and which do you want to work with over the next year?`
  - 2025 `AIModels`：`Which LLM models for AI tools have you used for development work in the past year, and which would you like to use next year?`
- 列名（public.csv）
  - 2023：`AISearchHaveWorkedWith` + `AIDevHaveWorkedWith`（去重后的工具个数）
  - 2024：`AISearchDevHaveWorkedWith`（工具个数）
  - 2025：`DevEnvsHaveWorkedWith` + `AIModelsHaveWorkedWith`（去重后的工具个数）
- 计算：提供两种计算方法
  - 绝对值方法：直接统计使用的工具数量
  - 相对值方法：工具个数 /（该年工具选项总数）* 100
  - 2023年工具选项总数：约20种（AISearch 12种 + AIDev 8种）
  - 2024年工具选项总数：约12种（AISearchDev 12种）
  - 2025年工具选项总数：约45种（DevEnvs 25种 + AIModels 20种）

### 3.2 开发者态度变化（Attitude & Trust）
**(I4) AI 态度指数 `AI_Attitude`（0–100）[2023/2024/2025]**
- 题目来源
  - 2023 `AISent`：`How favorable is your stance on using AI tools as part of your development workflow?`
  - 2024 `AISent`：同上
  - 2025 `AISent`：同上
- 列名（public.csv）：`AISent`
- 计算（Likert 映射）：`Very favorable=100`、`Favorable=75`、`Indifferent=50`、`Unsure=40`、`Unfavorable=25`、`Very unfavorable=0`

**(I5) AI 信任指数（输出准确性信任）`AI_Trust`（0–100）[2023/2024/2025]**
- 题目来源（注意：同一题意在不同年份列名不同，必须按题干对齐）
  - 2023 `AIBen`：`How much do you trust the accuracy of the output from AI tools as part of your development workflow?`
  - 2024 `AIAcc`：同上
  - 2025 `AIAcc`：同上
- 列名（public.csv）：2023 `AIBen`；2024/2025 `AIAcc`
- 计算（Likert 映射）：`Highly trust=100`、`Somewhat trust=75`、`Neither trust nor distrust=50`、`Somewhat distrust=25`、`Highly distrust=0`

**(I6) AI 威胁感知指数 `AI_Threat`（0–100）[2024–2025]**
- 题目来源
  - 2024 `AIThreat`：`Do you believe AI is a threat to your current job?`
  - 2025 `AIThreat`：同上
- 列名（public.csv）：`AIThreat`
- 计算：`Yes=100`、`I'm not sure=50`、`No=0`

### 3.3 AI 对效率与学习方式的影响（Benefits / Impact）
**(I7) 期望收益指数（效率/学习/质量/协作）`AI_ExpectedBenefits_*`（0–100）[2023–2024]**
- 题目来源（题意一致、列名跨年不同）
  - 2023 `AIAcc`：`For the AI tools you use... what are the MOST important benefits you are hoping to achieve?`
  - 2024 `AIBen`：同上
- 列名（public.csv）：2023 `AIAcc`；2024 `AIBen`（多选；`;` 分隔）
- 跨年可比口径（两年答案选项的交集，样本扫描得到 5 项）
  - `Increase productivity`、`Greater efficiency`、`Speed up learning`、`Improve accuracy in coding`、`Improve collaboration`
- 构建（示例）
  - `AI_ExpectedBenefits_Efficiency` = 选中 `{Increase productivity, Greater efficiency}` 的覆盖比例（0/1/2 → /2 * 100）
  - `AI_ExpectedBenefits_Learning` = 是否选中 `Speed up learning`（0/100）
  - `AI_ExpectedBenefits_Quality` = 是否选中 `Improve accuracy in coding`（0/100）
  - `AI_ExpectedBenefits_Collab` = 是否选中 `Improve collaboration`（0/100）

**(I8) AI 复杂任务胜任指数 `AI_ComplexHandling`（0–100）[2024–2025]**
- 题目来源
  - 2024 `AIComplex`：`How well do the AI tools you use... handle complex tasks?`
  - 2025 `AIComplex`：同上
- 列名（public.csv）：`AIComplex`
- 计算：按选项从"Very well"到"Very poor"做 0–100 映射（含"不使用/不知道"可单列为缺失或低分）

**(I9) AI Agent 实际影响指数（效率/学习/质量/协作）`AI_AgentImpact_*`（0–100）[2025 only]**
- 题目来源
  - 2025 `AIAgentImpact`：`To what extent do you agree with the following statements regarding the impact of AI agents on your work as a developer?`
  - 该题包含多条陈述（例如：`AI agents have increased my productivity.`、`...accelerated my learning...` 等）
- 列名（public.csv）：以同意程度拆成多列（如 `AIAgentImpactStrongly agree`、`AIAgentImpactSomewhat agree` 等），每列存放"被选中的陈述列表"
- 计算：对每条陈述做强度打分（`Strongly agree=1`、`Somewhat agree=0.5`、`Neutral=0`、`Somewhat disagree=-0.5`、`Strongly disagree=-1`），再按主题聚合并归一化到 0–100

### 3.4 工作满意度影响（Satisfaction）
**(I10) 工作满意度指数 `JobSatisfaction`（0–100）[2024–2025]**
- 题目来源
  - 2024 `JobSat`：`How satisfied are you in your current professional developer role?`
  - 2025 `JobSat`：同上
- 列名（public.csv）：`JobSat`（0–10 量表）
- 计算：`JobSat / 10 * 100`
- 分析建议：用 `AI_Adoption`/`AI_WorkflowCoverage_Current` 分桶比较均值；并用回归控制（工作年限、岗位、远程等）降低混杂影响

补充说明：
- 2023 年 public.csv 中没有 `JobSat` 量表列，因此"满意度的跨年变化"建议以 2024–2025 为研究窗口。

### 3.5 学习方式与学习投入（Learning）
**(I11) AI 学习相关使用指数 `AI_UseForLearning`（0–100）[2023/2024/2025]**
- 列名（public.csv）：同 (I2)
- 跨年可比口径（优先用一致场景）
  - 2023/2024：是否在 `AIToolCurrently Using` 中选择 `Learning about a codebase`
  - 2025：是否在 `AIToolCurrently mostly AI` 或 `AIToolCurrently partially AI` 中选择 `Learning about a codebase`（可把 `Learning new concepts or technologies` 作为扩展项单独统计）
- 计算：选择则 100，否则 0（或把"mostly=100、partially=50"做强度化）

**(I12) AI 学习投入指数 `AI_LearnEngagement`（0–100）[2025 only]**
- 题目来源
  - 2025 `LearnCodeAI`：`Did you spend time in the last year learning AI programming or AI-enabled tooling on your own or at work?`
  - 2025 `AILearnHow`：`How did you learn to code for AI in the past year? Select all that apply.`
- 列名（public.csv）：`LearnCodeAI`、`AILearnHow`
- 计算（示例）：`LearnCodeAI` 映射为 0/100（学过=100，不学=0），再叠加 `AILearnHow` 的渠道多样性（渠道数/该题总选项数 * 100）做加权平均

## 4. 实现与代码

所有上述指数均已通过 `task2.py` 文件实现，该文件能够处理2023、2024和2025三年的Stack Overflow开发者调查数据，并计算出所有定义的指数。计算结果将保存为CSV文件，供后续分析使用。

主要功能：
- 自动加载各年份的调查数据
- 根据各年份问卷结构差异正确计算各项指数
- 支持跨年可比性分析
- 保留绝对值和相对值两种计算方法（如AI工具广度指数）
- 输出结果包含所有定义的指数，便于后续趋势分析和用户画像构建
- 数据预处理：针对聚类分析前的数据进行预处理


使用方法：
```bash
cd final\ project
python task1.py
```

执行完成后，结果将保存在 `final project/results/` 目录下，每年生成一个CSV文件，包含所有受访者的各项指数计算结果。

## 5. 用户画像（Personas）：类目从哪里来、怎么丰富
### 5.1 画像维度（建议优先从"AI 行为 + 态度 + 体验结果"构建）
- AI 行为：`AI_Adoption`、`AI_WorkflowCoverage_Current`、`AI_ToolBreadth`
- AI 态度/信任：`AI_Attitude`、`AI_Trust`、`AI_Threat`
- AI 使用体验（有则用）：2024 `AIChallenges`；2025 `SOFriction`、`AIFrustration`、`AIAgents`、`AIAgentChange`、`AI_AgentImpact_*`
- 结果变量：`JobSatisfaction`（2024/2025）

### 5.2 分群方法与"富化解释"
- 分群输入：上述指数（统一 0–100）+ 若干关键哑变量（是否使用 agents、是否有高摩擦等）
- 分群方法：混合型数据可用 Gower 距离 + 层次聚类（或 k-prototypes）；输出 4–8 类
- 富化解释（画像"像人"）：对每类补充开发者背景变量的分布差异（如 `MainBranch`、`Employment`、`WorkExp/YearsCodePro`、`RemoteWork`、`Country`、常用技术栈等），并给出每类的"代表性特征 Top-N"

### 5.3 使用 AI（大模型）自动"写画像描述"（可行性与落地方式）
技术上可行，且适合把聚类结果转成可读的"人群画像文案"。关键前提：**把 AI 的输入限制为结构化的聚合统计/差异点**，而不是把整份原始问卷逐行喂给模型。

推荐工作流：
1) **先算人群特征表**（每个 cluster 一行）
   - 基础：`cluster_id`、样本量 `n`、占比 `pct`
   - 核心指数：`AI_Adoption`、`AI_WorkflowCoverage_Current`、`AI_Attitude`、`AI_Trust`、`AI_Threat`、`JobSatisfaction` 的均值/中位数/分位数
   - 关键行为：高摩擦占比（由 `SOFriction` 映射）、agents 使用占比（`AIAgents`）
   - "代表性特征 Top-N"：对比全体（或其他 cluster）后，差异最大的变量（百分点差/均值差）
2) **把"数字 + 结论约束"喂给 AI 写文案**
   - 输入建议：每个 cluster 的 JSON（或表格）+ 统一写作模板（见下）
   - 输出要求：只基于提供的数据描述，不得杜撰；每段必须引用 2–4 个关键数值（如"采用强度中位数/高摩擦占比/满意度"）
3) **人工与规则校验**
   - 校验 AI 文案里的数值是否与表一致；出现"无依据推断"则删改
   - 对敏感/身份类推断（地区、行业刻板印象）明确禁止

写作模板（给 AI 的约束提示）建议包含：
- 输出结构：`一句话标签` + `典型特征(3条)` + `与全体差异(2条)` + `潜在需求/痛点(1条)`  
- 禁止项：不得新增未提供字段；不得解释因果（只能描述关联）；不得使用带偏见的推断

## 6. 实现与代码 (Task 21 和 Task 22)

### Task 21: 聚类分析与用户画像构建

Task 21 实现了完整的聚类分析流程，主要包括以下步骤：

#### 6.1 数据预处理与缺失值处理
- **加载数据**: 从 [processed_data_for_clustering_task2.csv](file:///c:/Users/11752/Documents/GitHub/python_final_project/final%20project/results/processed_data_for_clustering_task2.csv) 加载数据
- **缺失值处理**: 
  - `AI_Adoption` 使用中位数填充
  - `AI_Attitude` 和 `AI_Trust` 使用 KNN 插值
  - 分类变量使用 'Unknown' 填充
- **异常值处理**: 使用 IQR + MAD 修正 Z 值法截断异常值，限定在 5%-95% 范围内

#### 6.2 聚类分析实现
- **分年抽样**: 按年份分别随机抽样（每组最多 6000 条记录）
- **聚类变量选择**: `AI_Adoption`, `AI_Attitude`, `AI_UseForLearning`, `AI_Trust`
- **距离计算**: 使用 Gower 距离计算样本间相似性（适用于混合数据类型）
- **聚类算法**: 使用层次聚类 (AgglomerativeClustering)，链接方法为 'average'
- **最优聚类数确定**: 使用轮廓系数法测试 2-10 个聚类，选择轮廓系数最高的聚类数
- **结果保存**: 将聚类结果合并保存至 `clustered_data_task21.csv`

#### 6.3 背景变量分析
- **可视化**: 生成各聚类中背景变量（如 `MainBranch`, `Employment`, `WorkExp`, `RemoteWork`, `Country`, `DevType`）的分布图
- **分布差异分析**: 计算每个聚类中背景变量的分布差异
- **Top-N 特征生成**: 计算每类与总体差异最大的特征，生成 Top-5 代表性特征

使用方法：
```bash
cd final\ project
python task21.py
```

执行完成后，结果将保存在 `final project/results/` 目录下，包括聚类结果、可视化图片和背景变量分析结果。

### Task 22: AI 人群画像文案生成

Task 22 实现了基于聚类结果的 AI 人群画像文案自动生成，主要包括以下步骤：

#### 6.4 聚类结果处理与占比计算
- **加载聚类数据**: 从 task21 生成的聚类结果文件读取数据
- **计算占比**: 计算并输出各聚类的样本数量和占比
- **统计摘要**: 生成聚类的统计摘要（均值、中位数等）

#### 6.5 AI 画像文案生成
- **API 调用**: 使用 DeepSeek API 进行 AI 画像文案生成
- **输入数据**: 将聚类统计信息、AI 相关指标和背景变量信息作为输入
- **系统提示**: 设置系统提示以约束 AI 输出格式和内容
- **结构化输出**: 按照指定结构生成画像文案：
  - 一句话标签（基于 AI 行为与态度特征）
  - 典型特征 (3 条，重点分析 AI 行为与态度指标)
  - 与全体差异 (2 条，对比 AI 相关指标)
  - 潜在需求/痛点 (1 条)

#### 6.6 结果保存与输出
- **JSON 格式**: 将 AI 生成的画像保存为 JSON 格式
- **文本格式**: 生成可读的文本文件，便于阅读和分析
- **结果文件**: 包括 `ai_generated_portraits.json` 和 `ai_generated_portraits.txt`

使用方法：
```bash
cd final\ project
python task22.py
```

执行完成后，AI 生成的人群画像将保存在 `final project/results/` 目录下，包括 JSON 和文本格式的画像结果。

### 6.7 数据流程总结

1. **Task 1**: 计算各项 AI 指数并保存
2. **Task 2**: 数据预处理，生成用于聚类的 CSV 文件
3. **Task 21**: 执行聚类分析、异常值处理、抽样、聚类计算、背景变量分析
4. **Task 22**: 基于聚类结果使用 AI 生成人群画像文案

这一完整流程实现了从原始调查数据到最终人群画像的端到端分析，充分体现了 README 中提到的用户画像构建目标。