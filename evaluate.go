package dopforge

import (
	"context"
	"encoding/json"
	"fmt"
	"os/exec"
	"strings"
	"time"
)

// ============================================================
// 评估层:Pipeline + Stage
// ============================================================
//
// 评估的核心论点是:**不强行返回标量奖励**,因为多目标评估天然是启发式的,
// 折成单分等于把启发式偏见注入选择压力。
//
// 替代方案:
//   1) HardGate     —— 硬门槛(布尔),不通过即淘汰
//   2) Soft Stages  —— 多维数值(向量),Pareto 直接消费
//
// 这两道闸的输出共同填充 Score。后续搜索层的 Pareto 比较只用 Score 中的
// 多维数值,**任何一处都不再合并维度**,直到最终 PickBestInBatch 在前沿
// 多个候选间需要 tiebreak 时才用 Goal.Dimension.Weight 做加权排序。

type Stage interface {
	Name() string
	Run(ctx context.Context, c *Candidate) StageOutput
}

// StageOutput 是单个 Stage 的产出。
//
// 任意 Stage 都可以贡献多个 Soft 维度。HardErr != nil 表示该 Stage 拒绝候选
// (即便它不是 HardGate 本身也可以拒绝,例如评估器内部数据完整性检查失败)。
type StageOutput struct {
	HardErr error
	Soft    map[string]float64
	Notes   string
}

// Pipeline 串联多个 Stage,产出最终 Score。
//
// 顺序:HardStages 先于 SoftStages。但即便 Hard 全失败,Soft 也仍会跑——
// 这样 LLM 拿到的 Notes 里既有"为什么不通过"也有"如果通过会怎样",反馈带宽更高,
// 加速下一轮迭代。
type Pipeline struct {
	HardStages []Stage
	SoftStages []Stage
	Goal       *Goal // 用于在 Soft 维度缺失时填默认值
}

// Evaluate 实现完整的四道闸评估。
func (p *Pipeline) Evaluate(ctx context.Context, c *Candidate) (*Score, error) {
	score := &Score{Soft: make(map[string]float64)}
	var notes []string

	for _, st := range p.HardStages {
		out := st.Run(ctx, c)
		score.Hard = append(score.Hard, GateResult{Name: st.Name(), Err: out.HardErr})
		mergeSoft(score.Soft, out.Soft)
		if out.Notes != "" {
			notes = append(notes, fmt.Sprintf("[%s] %s", st.Name(), out.Notes))
		}
	}
	for _, st := range p.SoftStages {
		out := st.Run(ctx, c)
		mergeSoft(score.Soft, out.Soft)
		if out.Notes != "" {
			notes = append(notes, fmt.Sprintf("[%s] %s", st.Name(), out.Notes))
		}
	}
	// 维度兜底:Goal 声明的每一维都必须在 Soft 里
	if p.Goal != nil {
		for _, d := range p.Goal.Dimensions {
			if _, ok := score.Soft[d.Name]; !ok {
				if d.HigherIsBetter {
					score.Soft[d.Name] = 0
				} else {
					score.Soft[d.Name] = 1e9
				}
			}
		}
	}
	score.Notes = strings.Join(notes, "\n")
	return score, nil
}

func mergeSoft(dst, src map[string]float64) {
	for k, v := range src {
		dst[k] = v
	}
}

// ============================================================
// 把 goal.HardGate 包成 Stage
// ============================================================

type hardGateStage struct{ g HardGate }

// WrapHardGates 把 Goal 的硬门槛全部包成 Stage 列表。
func WrapHardGates(gs []HardGate) []Stage {
	out := make([]Stage, 0, len(gs))
	for _, g := range gs {
		out = append(out, &hardGateStage{g: g})
	}
	return out
}

func (s *hardGateStage) Name() string { return s.g.Name() }
func (s *hardGateStage) Run(ctx context.Context, c *Candidate) StageOutput {
	if err := s.g.Check(c.WorkDir); err != nil {
		return StageOutput{HardErr: err, Notes: fmt.Sprintf("FAIL: %v", err)}
	}
	return StageOutput{}
}

// ============================================================
// ShellGate:开箱即用的硬门槛实现
// ============================================================

// ShellGate 跑一条 shell 命令看退出码。常见用法:`go build ./...`、`npm test`、
// `./run.sh --self-test`。Cmd 在候选解的 WorkDir 下执行(cwd = workDir)。
type ShellGate struct {
	GateName string
	Cmd      []string
	Timeout  time.Duration
}

func (g ShellGate) Name() string { return g.GateName }

func (g ShellGate) Check(workDir string) error {
	if len(g.Cmd) == 0 {
		return fmt.Errorf("ShellGate: empty Cmd")
	}
	to := g.Timeout
	if to == 0 {
		to = 60 * time.Second
	}
	ctx, cancel := context.WithTimeout(context.Background(), to)
	defer cancel()
	cmd := exec.CommandContext(ctx, g.Cmd[0], g.Cmd[1:]...)
	cmd.Dir = workDir
	out, err := cmd.CombinedOutput()
	if err != nil {
		s := string(out)
		if len(s) > 800 {
			s = s[:400] + "...\n[truncated]\n..." + s[len(s)-400:]
		}
		return fmt.Errorf("%s exit %v: %s", strings.Join(g.Cmd, " "), err, s)
	}
	return nil
}

// ============================================================
// LLMJudgeStage:多维 LLM 法官
// ============================================================
//
// 关键设计:
//   - 法官是一个独立 LLM,**与 mutate 用的不是同一只**。弱评估器评强生成,
//     在搜索过程中天然形成对抗,不会被生成器的盲点带偏。
//   - 输出是结构化 JSON:每一维一个 [0,1] 的分,加一段 critique。
//   - 维度对齐 Goal.Dimensions 严格检查;漏维度 = 解析失败 = 该轮分数缺失。

// JudgeCaller 是用户提供的 LLM 单点调用函数。
//
// 实现拿到完整 prompt,必须返回 LLM 回复文本(我们会从中提取 ```json 块)。
type JudgeCaller func(ctx context.Context, prompt string) (string, error)

// LLMJudgeStage 是产出 Goal.Dimensions 上多维分 + Notes 的 SoftStage。
type LLMJudgeStage struct {
	Goal           *Goal
	Caller         JudgeCaller
	FilesToInclude func(workDir string) (map[string]string, error)
}

func (s *LLMJudgeStage) Name() string { return "llm_judge" }

func (s *LLMJudgeStage) Run(ctx context.Context, c *Candidate) StageOutput {
	if s.Caller == nil || s.Goal == nil || s.Goal.LLMJudgeRubric == "" {
		return StageOutput{}
	}
	files, err := s.FilesToInclude(c.WorkDir)
	if err != nil {
		return StageOutput{Notes: fmt.Sprintf("judge: collect files: %v", err)}
	}
	prompt := buildJudgePrompt(s.Goal, files)
	reply, err := s.Caller(ctx, prompt)
	if err != nil {
		return StageOutput{Notes: fmt.Sprintf("judge: call err: %v", err)}
	}
	parsed, err := parseJudgeReply(reply, s.Goal.Dimensions)
	if err != nil {
		return StageOutput{Notes: fmt.Sprintf("judge: parse err: %v\nraw: %s", err, truncate(reply, 600))}
	}
	return StageOutput{Soft: parsed.Scores, Notes: parsed.Critique}
}

type judgeReply struct {
	Scores   map[string]float64 `json:"scores"`
	Critique string             `json:"critique"`
}

func parseJudgeReply(text string, dims []Dimension) (*judgeReply, error) {
	js := extractJSON(text)
	if js == "" {
		return nil, fmt.Errorf("no json block found")
	}
	var r judgeReply
	if err := json.Unmarshal([]byte(js), &r); err != nil {
		return nil, fmt.Errorf("unmarshal: %w", err)
	}
	for _, d := range dims {
		if _, ok := r.Scores[d.Name]; !ok {
			return nil, fmt.Errorf("missing dimension %q in judge reply", d.Name)
		}
	}
	return &r, nil
}

func extractJSON(text string) string {
	if i := strings.Index(text, "```json"); i >= 0 {
		rest := text[i+len("```json"):]
		if j := strings.Index(rest, "```"); j > 0 {
			return strings.TrimSpace(rest[:j])
		}
	}
	start := strings.Index(text, "{")
	end := strings.LastIndex(text, "}")
	if start >= 0 && end > start {
		return text[start : end+1]
	}
	return ""
}

func buildJudgePrompt(g *Goal, files map[string]string) string {
	var sb strings.Builder
	sb.WriteString("You are a strict, principled judge of a methodology implementation.\n\n")
	sb.WriteString("# Goal\n")
	sb.WriteString(g.Description)
	sb.WriteString("\n\n# Rubric\n")
	sb.WriteString(g.LLMJudgeRubric)
	sb.WriteString("\n\n# Dimensions\n")
	for _, d := range g.Dimensions {
		dir := "higher is better"
		if !d.HigherIsBetter {
			dir = "lower is better"
		}
		sb.WriteString(fmt.Sprintf("- %s (%s)\n", d.Name, dir))
	}
	sb.WriteString("\n# Candidate files\n")
	for path, content := range files {
		sb.WriteString(fmt.Sprintf("\n## %s\n```\n%s\n```\n", path, truncate(content, 4000)))
	}
	sb.WriteString(`
# Your task
Output ONLY a single JSON object inside a ` + "```json" + ` fenced block:

` + "```json" + `
{
  "scores": { "<dim_name>": <0.0-1.0>, ... },
  "critique": "concrete, specific feedback. point at file:line. say what to change next."
}
` + "```" + `

The critique will be fed to the next iteration. Be useful, not polite.
For "lower is better" dims, output the raw value; the search engine handles direction.
`)
	return sb.String()
}

func truncate(s string, max int) string {
	if len(s) <= max {
		return s
	}
	return s[:max/2] + "\n... [truncated] ...\n" + s[len(s)-max/2:]
}

// ============================================================
// 扩展:你想加 SyntheticPlayerStage 这类自定义 Stage,实现 Stage 接口即可
// 然后塞进 Pipeline.SoftStages 切片。dopforge 不内置——它是高度业务相关的。
// ============================================================
