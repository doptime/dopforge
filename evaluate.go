package dopforge

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"
)

// ============================================================
// 评估层:Pipeline + Stage
// ============================================================
//
// 评估的核心论点是:不强行返回标量奖励。方法论质量天然是多目标启发式判断,
// 折成单分会把偏见注入选择压力。
//
// 替代方案:
//   1) HardGate     —— 硬门槛(布尔),不通过即无法进入 Pareto 前沿
//   2) Soft Stages  —— 多维数值(向量),Pareto 直接消费

type Stage interface {
	Name() string
	Run(ctx context.Context, c *Candidate) StageOutput
}

// StageOutput 是单个 Stage 的产出。
// 任意 Stage 都可以贡献多个 Soft 维度。HardErr != nil 表示该 Stage 拒绝候选。
type StageOutput struct {
	HardErr error
	Soft    map[string]float64
	Notes   string
}

// Pipeline 串联多个 Stage,产出最终 Score。
//
// 顺序:HardStages 先于 SoftStages。但即便 Hard 失败,Soft 也仍会跑——这样 LLM
// 拿到的 Notes 里既有“为什么不通过”,也有“如果通过该怎么变好”。
type Pipeline struct {
	HardStages []Stage
	SoftStages []Stage
	Goal       *Goal // 用于在 Soft 维度缺失时填默认值
}

// Evaluate 实现完整评估。
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

// ShellGate 跑一条 shell 命令看退出码。Cmd 在候选解的 WorkDir 下执行。
//
// 对方法论文档任务,建议使用确定性文件检查,例如:
//
//	test -f METHOD.md
//	test ! -f go.mod && test ! -f run.sh
//	grep -q '^##' METHOD.md
//
// 不建议在默认方法论任务里运行沙盒、编译或启动服务。
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
// ContentGate:纯 Go 内容硬门槛
// ============================================================
//
// ContentGate 在候选 workDir 下读取一个 Markdown artifact,然后用纯 Go 函数
// 检查内容层面的强约束(关键词必现、长度区间、bullet/段落比上限等)。
// 比 ShellGate 更便携、更可控,且把内容 sanity 校验从 LLM judge 抽到硬门槛,
// 显著降低 judge 噪声对搜索的影响。

type ContentGate struct {
	GateName string
	// Artifact 是 workDir 下的相对路径;空则使用 "METHOD.md"。
	Artifact string
	// CheckFn 拿到读取后的全文,返回 nil 即通过。
	CheckFn func(content string) error
}

func (g ContentGate) Name() string { return g.GateName }

func (g ContentGate) Check(workDir string) error {
	if g.CheckFn == nil {
		return fmt.Errorf("ContentGate %q: CheckFn is nil", g.GateName)
	}
	name := g.Artifact
	if name == "" {
		name = "METHOD.md"
	}
	data, err := os.ReadFile(filepath.Join(workDir, name))
	if err != nil {
		return fmt.Errorf("read %s: %w", name, err)
	}
	return g.CheckFn(string(data))
}

// ============================================================
// LLMJudgeStage:多维 LLM 法官
// ============================================================
//
// 关键设计:
//   - 法官是一个独立 LLM,与 mutate 用的不是同一只
//   - 输出是结构化 JSON:每一维一个 [0,1] 的分,加一段 critique
//   - 维度对齐 Goal.Dimensions 严格检查;漏维度 = 解析失败 = 该轮分数缺失

// JudgeCaller 是用户提供的 LLM 单点调用函数。
// 实现拿到完整 prompt,必须返回 LLM 回复文本(我们会从中提取 JSON 块)。
type JudgeCaller func(ctx context.Context, prompt string) (string, error)

// LLMJudgeStage 是产出 Goal.Dimensions 上多维分 + Notes 的 SoftStage。
//
// Replicas 控制每个候选的 judge 重复采样次数;< 1 视为 1。N>=3 时,每一维取
// N 次返回值的中位数(对 LLM judge 噪声鲁棒),Notes 选择离中位数标量最近的
// 那次 critique,避免 outlier critique 误导下一轮 mutate。
type LLMJudgeStage struct {
	Goal           *Goal
	Caller         JudgeCaller
	FilesToInclude func(workDir string) (map[string]string, error)
	Replicas       int
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

	n := s.Replicas
	if n < 1 {
		n = 1
	}

	type slot struct {
		parsed *judgeReply
		err    error
	}
	slots := make([]slot, n)
	var wg sync.WaitGroup
	for i := 0; i < n; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			reply, err := s.Caller(ctx, prompt)
			if err != nil {
				slots[i] = slot{err: fmt.Errorf("call: %w", err)}
				return
			}
			parsed, err := parseJudgeReply(reply, s.Goal.Dimensions)
			if err != nil {
				slots[i] = slot{err: fmt.Errorf("parse: %w; raw: %s", err, truncate(reply, 400))}
				return
			}
			slots[i] = slot{parsed: parsed}
		}(i)
	}
	wg.Wait()

	var ok []*judgeReply
	var errs []string
	for _, sl := range slots {
		if sl.err != nil {
			errs = append(errs, sl.err.Error())
			continue
		}
		ok = append(ok, sl.parsed)
	}
	if len(ok) == 0 {
		return StageOutput{Notes: "judge: all replicas failed: " + strings.Join(errs, " | ")}
	}

	// 每一维取 N 次的中位数
	soft := make(map[string]float64, len(s.Goal.Dimensions))
	for _, d := range s.Goal.Dimensions {
		vals := make([]float64, 0, len(ok))
		for _, r := range ok {
			vals = append(vals, r.Scores[d.Name])
		}
		soft[d.Name] = median(vals)
	}

	// Notes 选离中位数标量最近的那次 critique
	scalars := make([]float64, len(ok))
	for i, r := range ok {
		scalars[i] = (&Score{Soft: r.Scores}).Value(s.Goal.Dimensions)
	}
	medScalar := median(scalars)
	bestIdx, bestDiff := 0, 1e18
	for i, sc := range scalars {
		d := sc - medScalar
		if d < 0 {
			d = -d
		}
		if d < bestDiff {
			bestDiff, bestIdx = d, i
		}
	}
	notes := ok[bestIdx].Critique
	if len(errs) > 0 {
		notes += fmt.Sprintf("\n\n[judge: %d/%d replicas failed]", len(errs), n)
	}
	return StageOutput{Soft: soft, Notes: notes}
}

func median(xs []float64) float64 {
	if len(xs) == 0 {
		return 0
	}
	cp := append([]float64(nil), xs...)
	sort.Float64s(cp)
	n := len(cp)
	if n%2 == 1 {
		return cp[n/2]
	}
	return (cp[n/2-1] + cp[n/2]) / 2
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
		if r.Scores[d.Name] < 0 || r.Scores[d.Name] > 1 {
			return nil, fmt.Errorf("dimension %q out of range [0,1]: %v", d.Name, r.Scores[d.Name])
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
	sb.WriteString("You are a strict, principled judge of a methodology document.\n")
	sb.WriteString("The candidate should be documentation/methodology, not a runnable software project.\n\n")
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
	sb.WriteString("\n# Candidate Markdown files\n")
	if len(files) == 0 {
		sb.WriteString("\nNo Markdown files were collected. This should score very poorly.\n")
	}
	for path, content := range files {
		sb.WriteString(fmt.Sprintf("\n## %s\n```markdown\n%s\n```\n", path, truncate(content, 6000)))
	}
	sb.WriteString(`
# Your task
Output ONLY a single JSON object inside a ` + "```json" + ` fenced block:

` + "```json" + `
{
  "scores": { "<dim_name>": <0.0-1.0>, ... },
  "critique": "concrete, specific feedback. Point at Markdown section names or line numbers if possible. Say exactly what to change next. Penalize executable project scaffolding."
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
