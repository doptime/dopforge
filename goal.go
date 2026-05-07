// Package dopforge 是 dopharness 之上的方法论搜索引擎。
//
// 核心思想:你描述“想得到什么方法论”和“怎么算好”(Goal),不描述具体写法;
// dopforge 用 SimpleTES 风格的 C×L×K 三轴搜索 + 多维 Pareto 评分驱动 LLM
// 对 Markdown 方法论文档进行多路线迭代,最终给你一组互不支配的精英版本。
//
// 文件分工:
//
//	goal.go      —— 描述任务:意图、维度、硬门槛
//	candidate.go —— Candidate + Score + Workspace
//	evaluate.go  —— Pipeline + ShellGate + LLMJudgeStage
//	search.go    —— C×L×K 主循环 + Pareto 前沿 + 停滞检测 + 轨迹记录
//	forge.go     —— Forge 顶层门面 + 跨 lineage 共享记忆
package dopforge

// Goal 是用户唯一必须仔细写的东西。
//
// 设计哲学:在这里描述“是什么”和“怎么算好”,不描述“怎么写”。
// 后者交给搜索引擎探索。对于方法论任务,Description 应明确输出物、读者、
// 禁区和风格边界,例如:“输出单个 METHOD.md;不要生成代码项目或运行脚本”。
type Goal struct {
	// Description 是给 LLM 看的自然语言意图。
	// 应当包含:主题、目标读者、输出形态、风格偏好、显式禁区。
	Description string

	// HardGates 是一组硬门槛。任何一条不通过,候选解不会进入 Pareto 前沿。
	// 对方法论文档任务,典型门槛是:METHOD.md 存在、非空、没有 go.mod/run.sh 等工程文件。
	HardGates []HardGate

	// Dimensions 声明多维评分轴及方向。评估器返回的 Soft map 必须覆盖这里的所有维度。
	// 搜索本身不依赖 Weight,Weight 只在同 batch 多个非支配候选并列时做 tiebreak。
	Dimensions []Dimension

	// LLMJudgeRubric 给 LLM 法官看的评分细则。空则不启用 LLM 软评分。
	LLMJudgeRubric string
}

// Dimension 是一个评分轴。
type Dimension struct {
	Name           string
	HigherIsBetter bool
	Weight         float64 // tiebreak 用,可空
}

// HardGate 是一条硬门槛检查器。
//
// 实现方拿到候选解的 workDir,自己决定怎么验证。返回 nil 即通过。
// 对文档任务,建议尽量使用轻量、确定性的 shell 检查,而不是运行沙盒。
type HardGate interface {
	Name() string
	Check(workDir string) error
}

// HardGateFunc 把一个普通函数包成 HardGate。
type HardGateFunc struct {
	N string
	F func(workDir string) error
}

func (g HardGateFunc) Name() string               { return g.N }
func (g HardGateFunc) Check(workDir string) error { return g.F(workDir) }
