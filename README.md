# dopforge

> 用 SimpleTES 风格的 C×L×K 评估驱动搜索,迭代 Markdown 方法论文档。
>
> 底层修改引擎是 `dopharness`:它提供受控的 chunk 索引、三态上下文筛选和工具化编辑。
> 从工程心智看,dopharness 接近 GenericAgent 的 Agent/ToolCall 循环,再叠加 AST/Markdown
> chunk 化与失败回滚;dopforge 则把它放进评估驱动的搜索外循环。

## 项目定位

dopforge 不是某个具体“认字游戏”或“如何学习”的项目。它是一个**方法论迭代工具**:

1. 用户给出一个方法论探索目标,例如“如何学习”“如何做产品调研”“如何做论文复现”。
2. dopforge 维护多条并行 lineage,每条 lineage 迭代修改 Markdown 方法论文档。
3. 每一代生成 K 个候选版本,通过硬门槛 + LLM 多维 judge 评价。
4. 搜索内部用一个 weighted scalar proxy 做 SimpleTES 风格局部选择。
5. 最终仍输出 Pareto Frontier,保留多目标互不支配的方法论版本供人类挑选。

这个分层很重要:

- **框架意图**:方法论迭代引擎,写在代码、README 和默认 prompt 中。
- **用户任务意图**:某次具体探索的目标、维度、硬门槛和运行目录,写在 `examples/<task>/goal.json` 中。

## 与 SimpleTES 的对应关系

SimpleTES 的核心不是“让模型想更久”,而是把 test-time compute 花在
`propose -> evaluate -> refine` 的评估驱动循环上。dopforge 的对应如下:

| SimpleTES 概念 | dopforge 对应 |
|---|---|
| Global width `C` | `Budget.C`: 并行 lineage 数 |
| Refinement depth `L` | `Budget.L`: 每条 lineage 的迭代代数 |
| Local samples `K` | `Budget.K`: 每代从同一父代生成 K 个候选 |
| Proposal / refinement | `dopharness.Run` 通过工具修改 Markdown chunk |
| Evaluator | `Pipeline`: hard gates + `LLMJudgeStage` |
| Scalar score | `Score.Value(dimensions)`: 多维 LLM 分数的 weighted scalar proxy |
| Local selection | `PickBestInBatch`: 在通过硬门槛的 K 个候选中选 scalar proxy 最高者 |
| Best trajectory history | `Trajectory`: 记录每代 picked candidate、scalar proxy、soft vector 和 notes |
| Diversity / final choice | `GlobalFrontier`: 合并各 lineage 的 Pareto 前沿 |

与论文最大的差异是:论文中的很多任务有可靠的程序化标量评价,而方法论质量只能依赖
LLM 多维启发式 judge。为了避免偏离 SimpleTES 的搜索动力,当前版本做了折中:

- 搜索内部使用 `Score.Value` 作为稳定 scalar proxy,对齐 SimpleTES 的局部选择假设。
- 最终输出仍使用 Pareto Frontier,避免把方法论质量过早压成单一分数。
- hard gates 尽量保持确定性,用于防止 reward hacking,例如禁止生成 Go module、run.sh、Dockerfile 等工程脚手架。

## 包结构

```text
dopforge/
├── README.md
├── goal.go                         # Goal、Dimension、HardGate
├── candidate.go                    # Candidate、Score、Workspace
├── evaluate.go                     # Pipeline、ShellGate、LLMJudgeStage
├── search.go                       # C×L×K 主循环、scalar proxy、Pareto、Trajectory
├── forge.go                        # Forge 门面、dopharness Mutator、SharedMemory
├── main/
│   └── main.go                     # CLI 入口,接 doptime/llm
└── examples/
    └── learning_methodology/
        └── goal.json               # 示例任务:如何学习的方法论
```

注意:项目根目录不再保留 `goal.json`,`main/goal.json` 也已删除。所有具体任务配置都放到
`examples/<task>/goal.json`,避免框架意图和任务意图混淆。

## 配置文件

配置文件现在分成 `run` 和 `goal` 两层:

```json
{
  "run": {
    "name": "learning-methodology",
    "work_dir": "./runs/learning-methodology",
    "seed_dir": "./seeds/learning-methodology",
    "artifact": "METHOD.md"
  },
  "goal": {
    "description": "...",
    "llm_judge_rubric": "...",
    "dimensions": [
      { "name": "clarity", "higher_is_better": true, "weight": 1.0 }
    ],
    "hard_gates": [
      { "name": "artifact_exists", "cmd": ["bash", "-c", "test -f METHOD.md"], "timeout": "5s" }
    ]
  }
}
```

`run.work_dir` 用来集中存放全部中间结果,包括候选目录、共享 skills 和 trajectory:

```text
runs/<name>/
├── L0/c00001/...
├── L1/c00002/...
├── .shared/skills/...
└── trajectory.json
```

命令行参数可以覆盖配置文件:

```bash
go run ./main -config ./examples/learning_methodology/goal.json

go run ./main \
  -config ./examples/learning_methodology/goal.json \
  -work ./runs/learning-methodology-exp2 \
  -C 4 -L 6 -K 3
```

优先级:

```text
命令行参数 > 配置文件 run 字段 > 默认值
```

为了兼容旧版本,`-goal` 仍可作为 `-config` 的别名,但新代码和文档都推荐使用 `-config`。

## 运行方式

```bash
# 最小预算验证流程
go run ./main -config ./examples/learning_methodology/goal.json -C 2 -L 2 -K 2

# 放大探索
go run ./main -config ./examples/learning_methodology/goal.json -C 4 -L 6 -K 3
```

如果 `run.seed_dir` 为空或不存在 Markdown 文件,CLI 会创建一个最小 `METHOD.md` 种子。
它不会创建 `go.mod`、`main.go`、`run.sh` 或任何沙盒运行环境。

## dopharness 接口契约

集成方只需要关心稳定入口:

```go
// 包路径 github.com/doptime/dopharness/harness
type Harness struct{ /* opaque */ }

func New(cfg Config) (*Harness, error)
func (h *Harness) Index(ctx) (*index.Report, error)
func (h *Harness) AsLLMTools(builder tools.ToolBuilder) []any
func (h *Harness) Run(ctx, userPrompt) (*RunReport, error)
func (h *Harness) Memory() *memory.Memory
```

生命周期:

```text
New -> Index -> AsLLMTools(builder) -> Run
```

三类 LLM caller:

```go
// gateway 包
type TriageCaller func(p TriagePromptParams, sink func(*TriageDecisionPayload)) error
type ExpandCaller func(p ExpandPromptParams, sink func(*ExpandDecisionPayload)) error

// harness 包
type MainCaller func(systemPrompt, userPrompt string, tools []any) error
```

工具桥接:

```go
type ToolBuilder interface {
    Build(name, desc string, handler any) any
}
```

dopharness 默认暴露 `modify_chunk / delete_chunk / add_chunk / create_file / delete_file /
read_chunk / search_chunks_by_name`。dopforge 的 prompt 会约束这些工具只用于 Markdown 方法论文档。

## 设计原则

### 1. 搜索内部需要 scalar proxy

SimpleTES 的局部选择依赖可比较分数。方法论任务没有天然程序化评分,但如果完全只用 Pareto,
K 内选择会过于松散,停滞判断也会不稳定。因此当前版本将多维 LLM judge 分数通过
`Score.Value(dimensions)` 转成 weighted scalar proxy,用于局部选择和 trajectory。

### 2. 最终结果保留 Pareto Frontier

方法论质量天然多目标:清晰、可执行、深度、适配性、Markdown 质量不一定同向。最终仍输出
Pareto 前沿,避免唯一标量把候选过早压扁。

### 3. Hard gates 用来防 reward hacking

LLM judge 是启发式评价,容易被格式、话术或无关工程脚手架欺骗。因此默认示例用确定性 shell
检查约束产物:

- 必须有 `METHOD.md`
- 必须有基本 Markdown 结构
- 顶层只能有 Markdown 文件
- 不能生成 `go.mod/main.go/run.sh/package.json/Dockerfile`

### 4. 运行目录属于任务配置

多项目探索时,中间结果必须隔离。`run.work_dir` 进入配置文件后,每个探索项目都可以拥有独立
`runs/<name>` 目录,不会互相覆盖。

## 许可

MIT(待定)。
