// Package dopforge 是 dopharness 之上的方法论搜索引擎。
//
// 核心思想:你描述"想造什么"和"怎么算好"(Goal),不描述实现路径;dopforge 用
// SimpleTES 风格的 C×L×K 三轴搜索 + 多维 Pareto 评分驱动 LLM 自己探索具体实现,
// 最终给你一组互不支配的精英解。
//
// 五个文件、单一包:
//   goal.go      —— 描述任务:意图、维度、硬门槛
//   candidate.go —— 候选解的身份、分数、独立工作目录管理
//   evaluate.go  —— 四道闸评估器:硬门槛 + LLM 多维法官
//   search.go    —— C×L×K 主循环 + Pareto 前沿 + 停滞检测 + 轨迹记录
//   forge.go     —— 顶层门面 + 跨 lineage 共享记忆(接 dopharness 4 层)
//
// 见 README.md 了解 dopharness 的接口契约和使用步骤。
package dopforge

// Goal 是用户唯一必须仔细写的东西。
//
// 设计哲学:在这里描述"是什么"和"怎么算好",**不**描述"怎么做"。后者完全交给搜索
// 引擎自己探索。如果你忍不住想在 Description 里写"先实现 X 然后 Y",删掉那段——
// 它会把整个搜索锁死在你预想的路径上。
type Goal struct {
	// Description 是给 LLM 看的自然语言意图。
	// 应当包含:目的、风格偏好、参考样例、显式禁区,但不写实现路径。
	Description string

	// HardGates 是一组硬门槛。任何一条不通过,候选解直接被丢弃。
	// 例:能编译、能启动、单局 30 秒内结束、不读外部网络。
	HardGates []HardGate

	// Dimensions 声明了多维评分轴及方向。评估器返回的 Soft map 必须覆盖这里
	// 的所有维度。搜索本身不依赖 Weight,Weight 只在 PickBestInBatch 多个非
	// 支配候选并列时用作 tiebreak。
	Dimensions []Dimension

	// LLMJudgeRubric 给 LLM 法官看的评分细则。空则不启用法官单点评分。
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
// 实现方拿到候选解的 workDir,自己决定怎么验证。返回 nil 即视作通过。
// 典型:在 workDir 里 go build / npm test / docker run --rm,看退出码。
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
