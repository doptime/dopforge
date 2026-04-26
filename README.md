# dopforge

> 用 SimpleTES 风格的 C×L×K 三轴搜索 + 多维 Pareto 评分,让 LLM 在你描述的目标上
> 自己探索方法论实现路径,产出一组互不支配的精英解。
>
> 底层修改引擎:[doptime/dopharness](https://github.com/doptime/dopharness)。

## 目录

1. [它解决什么问题](#它解决什么问题)
2. [核心思想](#核心思想)
3. [包结构](#包结构)
4. [快速开始](#快速开始)
5. [dopharness 接口契约](#dopharness-接口契约)
6. [设计理由](#设计理由)
7. [已知改进项](#已知改进项)
8. [v1 消融](#v1-消融)

---

## 它解决什么问题

dopharness 让 LLM 改代码不再幻觉,但单次修改的方向仍然由 LLM 拍脑袋决定。
对于"造一个最佳认字游戏"这类**没有绝对评估标准**的开放式任务,单次拍脑袋远远不够——
你需要让系统**并行探索许多方向**、**用多维评估筛掉显然差的**、把帕累托前沿留给人挑。

dopforge 把这件事做了。它不告诉 LLM 怎么做,只告诉它"什么算好",然后开 C×L×K 路并行
让它自己造、自己改、自己进化,最终给你一组互不支配的精英实现。

## 核心思想

| 来源 | 思想 | dopforge 怎么用 |
|---|---|---|
| **SimpleTES** | C×L×K 三轴搜索 | 全局并行 C 路 + 局部精修 L 轮 + 海选 K 个挑赢家 |
| **SimpleTES** | 轨迹级"奖励在终点" | `Lineage.BestEver` 跟踪 Pareto 历史峰,中途低分不致命,后代翻盘仍前沿 |
| **dopharness** | 三态网关 + 外科修改 | 用作 Mutator;每个候选解一份 cp -a 隔离的 WorkDir |
| **dopharness** | 4 层记忆 | L0/L2/L3 跨 lineage 共享,L4 per-candidate 随父代继承 |
| **dopharness** | markdown chunk 化 | LLM 节级编辑 README/spec 文档,不必整篇重写 |
| **Pareto 前沿** | 多目标决策 | 全程不强行加权;最终输出非支配集合,人工挑赢家 |
| **早停** | 基线最优时停 | `StallAfter{N}` 检测 BestEver 不再扩张 ⇒ 提前释放 lineage 预算 |

## 包结构

```
dopforge/
├── README.md                          ← 本文
├── goal.go                            ← Goal: 描述"是什么/怎么算好",不写"怎么做"
├── candidate.go                       ← Candidate + Score + Workspace
├── evaluate.go                        ← Pipeline + ShellGate + LLMJudgeStage
├── search.go                          ← Lineage + Pareto + Plateau + Trajectory + 主循环
├── forge.go                           ← Forge 顶层门面 + SharedMemory
├── cmd/dopforge/main.go               ← 开箱即用的入口程序(接 doptime/llm)
└── examples/recognition_game/
    └── goal.json                      ← 样例任务:认字游戏
```

`cmd/dopforge/main.go` 是真正的入口,接 doptime/llm 跑得起来。换 LLM SDK 时
只改这一个文件的 `buildCallers()`,其他不动。

依赖:`github.com/doptime/dopharness`、`github.com/doptime/llm`。

---

## 快速开始

```bash
# 1. 写一份 goal.json(或者用 examples/recognition_game/goal.json 试水)
cp examples/recognition_game/goal.json ./goal.json

# 2. 准备种子目录(没有就让 main 自动放一个最小 stub)
mkdir -p ./seed

# 3. 跑一个最小预算的搜索验证管道
go run ./cmd/dopforge \
    -goal ./goal.json \
    -seed ./seed \
    -work ./.forge_work \
    -C 2 -L 2 -K 2

# C=2 L=2 K=2 = 8 次 mutate,大约消耗 8 次主模型调用 + 8 次 judge 调用 +
# 16 次 triage 调用。看 ./.forge_work/trajectory.json 确认管道通了再放大。

# 4. 通了之后放大到正式预算
go run ./cmd/dopforge -goal ./goal.json -C 4 -L 6 -K 3
```

输出:`./.forge_work/L*/c*****/` 每个目录是一份候选实现。命令行最后会打印
帕累托前沿的几条候选 + 它们的多维评分。**人工挑一个**。

---

## dopharness 接口契约

> 集成 dopforge 不需要看 dopharness 源码;以下是稳定 API 摘要。

### 心智模型

dopharness 给 LLM 提供**幻觉最小化的代码修改环境**:项目按 AST 切成 chunk,
小模型先做三态裁定(FULL/SKELETON/IGNORE),大模型只能通过有语法校验的 ToolCall
改代码,失败回滚、最多重试 3 轮。

集成方关心的是它的 IO:**喂一段自然语言任务 → 拿到一份 RunReport,工程文件就地改完**。

### 顶层入口

```go
// 包路径 github.com/doptime/dopharness/harness
type Harness struct{ /* opaque */ }

func New(cfg Config) (*Harness, error)
func (h *Harness) Index(ctx) (*index.Report, error)
func (h *Harness) BuildContext(userPrompt) (*BuildContextResult, error)
func (h *Harness) AsLLMTools(builder tools.ToolBuilder) []any
func (h *Harness) Run(ctx, userPrompt) (*RunReport, error)
func (h *Harness) Memory() *memory.Memory
```

**生命周期**:`New → Index → AsLLMTools(builder) → Run`(可重复)。
`Run` 必须在 `AsLLMTools` 之后调用;两者与 `Index` 互斥。

### 三类 Caller(集成方实现)

dopharness 不绑定任何 LLM 客户端库,集成方桥接以下函数:

```go
// gateway 包
type TriageCaller func(p TriagePromptParams, sink func(*TriageDecisionPayload)) error
type ExpandCaller func(p ExpandPromptParams,  sink func(*ExpandDecisionPayload))  error

// harness 包
type MainCaller func(systemPrompt, userPrompt string, tools []any) error
```

dopforge 在 `cmd/dopforge/main.go` 给出了接 doptime/llm 的完整桩;换 SDK 复用
那一文件的结构即可。

### 工具桥接

```go
type ToolBuilder interface {
    Build(name, desc string, handler any) any  // 返回 LLM SDK 的 Tool 对象
}
```

dopharness 默认暴露 7 个工具(modify_chunk / delete_chunk / add_chunk /
create_file / delete_file / read_chunk / search_chunks_by_name)。
markdown 文件用相同的工具集编辑,无需特殊处理。

### Memory 4 层

```go
type Memory struct {
    L0 Layer  // Meta Rules     (system prompt, 永久规则)
    L1 Layer  // Insight Index  (gateway 产出的三态 chunk)
    L2 Layer  // Global Facts   (项目级稳定事实)
    L3 Layer  // Task Skills    (可复用 SOP, 从目录加载 .md)
    L4 Layer  // Session Records(历史会话)
}
```

L0/L2/L3/L4 都是导出字段,集成方可以**直接覆盖**(dopforge 的 SharedMemory 就这么做)。
L1 由 gateway 自动产出,不要碰。

### 关键不变量

1. **就地修改**:`Run` 成功后 `ProjectRoot` 下源码已改写,无 staging
2. **失败回滚**:`Run` 失败时部分修改仍在;dopforge 通过 cp -a 副本天然解决
3. **chunk store 同步**:`Run` 成功后 `.dopharness/` 与磁盘一致;下次 Index 增量
4. **隔离**:dopharness 不写 ProjectRoot 之外的文件

---

## 设计理由

四个反常识的决定,逐一解释:

**1. 不强行返回标量奖励。** 多目标评估天然是启发式的,折成单分等于把启发式偏见
注入选择压力。dopforge 全程多维向量,只在 PickBestInBatch 内 tiebreak 时用一次
加权,且仅作用于 batch 内,不影响全局前沿。

**2. 不在 lineage 之间做产量归一化。** 你可能直觉想奖励"产量稳定"的 lineage,
但这会惩罚"长期低分突然翻盘"的探索路径——而那种路径恰好是 SimpleTES 论文里 21 个
突破中最常见的来源。让 lineage 之间互不打架,只在全局 Pareto 前沿这一层汇合。

**3. 不强制 Elo / 排名机制。** 之前版本设计过多维 Elo + 锚点 + 毕业制,跑了一轮
分析发现:LLMJudgeStage 已经直接产多维 [0,1] 分,Pareto 直接消费这些分就够了。
Elo 只是"对法官打分稳定性的兜底",在多数场景下是过度防御。砍掉。

**4. 早停优于全跑完。** 一条 lineage 在第 3 代就已达到它的最优形态后,后续 K 个孩子
怎么 mutate 都比不过,这种情况下继续跑就是浪费。`StallAfter{N: 3}` 检测 BestEver
连续 N 代不扩张就停,把预算让给还在前进的 lineage。

---

## 已知改进项

跑得起来,但有两处明显低效。等遇到瓶颈了再改:

### 改进 1:法官评估应该用 chunk diff,不应该整文件喂

**现状**:`LLMJudgeStage` 通过 `FilesToInclude` 把候选解 WorkDir 整目录读出来,
每个文件 4KB 截断后塞给法官 prompt。这有两个代价——
法官每代都重看 80% 不变的内容(评分噪声盖过实际改动信号),且 token 占用上不去。

**应该改成**:
- 父代评估时:整文件喂(基线评分)
- 子代评估时:只喂 dopharness chunk store 中**与父代不同的 chunk**(diff 评分)

dopharness 的 chunk store 已经按文件 + ID 索引,能直接 diff `parent.WorkDir/.dopharness/chunks.json`
和 `child.WorkDir/.dopharness/chunks.json`。**markdown chunk 化在这里立刻发挥价值**——
README 改了一个 H2 节,只把那个 chunk 喂给法官,而不是整个 README。

代码改造点:`evaluate.go` 的 `LLMJudgeStage.Run` 加一个 `Parent` 参数,
`buildJudgePrompt` 加 diff 段。约 50 行。

### 改进 2:种子代不应该跑 C 次相同的评估

**现状**:`search.go` gen=0 阶段每个 lineage 各 NewCandidate 一个 seed,各自
Materialize(各自 cp 一份 SeedDir),各自 Pipeline.Evaluate。**C 次评估的结果完全相同**,
浪费 C-1 次 LLM 调用。

**应该改成**:
- 跑 1 次 seed 评估
- 让所有 lineage 的 history[0] 共享这个 seed candidate
- 注意:`Workspace.Cleanup` 此时不能动 seed.WorkDir(否则其他 lineage 失去拷贝源)

代码改造点:`search.go` gen=0 块,约 30 行。

---

## v1 消融

从早期设计到当前版本砍掉的:

| 砍掉的组件 | 行数 | 砍掉理由 |
|---|---:|---|
| 多维 Elo + 锚点 + 毕业制 | ~390 | LLMJudge 已直接产多维分,Pareto 直接消费就够 |
| SyntheticPlayer Stage 桩 | ~50 | 高度业务相关;用户实现 Stage 接口自己加 |
| AbsoluteFloor + CombinedPolicy | ~60 | 策略组合糖,实战很少用 |
| Trajectory 完整版(含 child snapshot) | ~50 | 简化为只记 picks + frontier 已够 debug |

**剩下的都是承重墙**。再砍要伤设计了。

未来扩展点(蓄意没做):

- **trajectory-level post-training**:SimpleTES 第二大贡献。dopforge 不训练模型,
  但 trajectory.json 的 schema 已预留;可以单独起 forge_train 项目消费它。
- **跨 lineage 受精**:让 lineage A 的 winner 被 lineage B 拷贝继续演化。
  evolutionary 算法常见做法,但会复杂化轨迹分析,v1 不做。
- **K 自适应**:K 在 lineage 进步快时减小、停滞时增大。可以做但收益不确定。

## 许可

MIT(待定)。
