// Command dopforge 是一个开箱即用的方法论搜索入口。
//
// 它做四件事:
//  1. 读取一份 goal.json 描述任务(意图/维度/硬门槛)
//  2. 把 dopforge 需要的 5 个 LLM 回调接到 doptime/llm
//  3. 启动 Forge.Run,把搜索预算从命令行读
//  4. 打印帕累托前沿 + 写 trajectory.json
//
// 这个程序本身可以直接 go run。如果你要换 LLM SDK(OpenAI Go、langchaingo、本地
// Ollama 等),只需要替换 buildCallers() 一个函数,其他都不动。所以这个 main
// 既是参考实现,也是"最小可改造单元"。
//
// 用法:
//
//	dopforge -goal ./goal.json -seed ./seed -work ./.forge_work -C 2 -L 2 -K 2
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"text/template"
	"time"

	df "github.com/doptime/dopforge"
	"github.com/doptime/dopharness/gateway"
	"github.com/doptime/dopharness/tools"
	"github.com/doptime/llm"
)

// goalJSON 是 goal.json 的反序列化结构。
//
// 之所以单独一份而不直接用 dopforge.Goal:Goal.HardGates 是接口切片,JSON 没法直接
// 反序列化。我们在 JSON 里只描述 ShellGate 这一种(95% 用户够用了),复杂场景请自己
// 写 Go 代码构造 Goal,跳过这个 JSON 配置层。
type goalJSON struct {
	Description    string `json:"description"`
	LLMJudgeRubric string `json:"llm_judge_rubric"`
	Dimensions     []struct {
		Name           string  `json:"name"`
		HigherIsBetter bool    `json:"higher_is_better"`
		Weight         float64 `json:"weight,omitempty"`
	} `json:"dimensions"`
	HardGates []struct {
		Name    string   `json:"name"`
		Cmd     []string `json:"cmd"`
		Timeout string   `json:"timeout,omitempty"` // "60s" 这种
	} `json:"hard_gates"`
}

func loadGoal(path string) (*df.Goal, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read %s: %w", path, err)
	}
	var raw goalJSON
	if err := json.Unmarshal(data, &raw); err != nil {
		return nil, fmt.Errorf("parse %s: %w", path, err)
	}
	g := &df.Goal{
		Description:    raw.Description,
		LLMJudgeRubric: raw.LLMJudgeRubric,
	}
	for _, d := range raw.Dimensions {
		g.Dimensions = append(g.Dimensions, df.Dimension{
			Name: d.Name, HigherIsBetter: d.HigherIsBetter, Weight: d.Weight,
		})
	}
	for _, h := range raw.HardGates {
		to := 60 * time.Second
		if h.Timeout != "" {
			if d, err := time.ParseDuration(h.Timeout); err == nil {
				to = d
			}
		}
		g.HardGates = append(g.HardGates, df.ShellGate{
			GateName: h.Name, Cmd: h.Cmd, Timeout: to,
		})
	}
	return g, nil
}

func main() {
	var (
		goalPath = flag.String("goal", "./goal.json", "path to goal.json")
		seedDir  = flag.String("seed", "./seed", "seed project directory")
		workRoot = flag.String("work", "./.forge_work", "where to put candidate workdirs")
		C        = flag.Int("C", 2, "parallel lineages")
		L        = flag.Int("L", 2, "generations per lineage")
		K        = flag.Int("K", 2, "children per generation")
	)
	flag.Parse()

	// --------------------------------------------------------
	// 1. 加载 Goal
	// --------------------------------------------------------
	g, err := loadGoal(*goalPath)
	if err != nil {
		log.Fatalf("load goal: %v", err)
	}
	log.Printf("loaded goal with %d dimensions, %d hard gates",
		len(g.Dimensions), len(g.HardGates))

	// --------------------------------------------------------
	// 2. 准备 SeedDir(确保存在,且如果是空的就放个最小 stub)
	// --------------------------------------------------------
	if err := ensureSeed(*seedDir); err != nil {
		log.Fatalf("ensure seed: %v", err)
	}

	// --------------------------------------------------------
	// 3. 接 5 个 LLM Caller(默认接 doptime/llm)
	//    要换 SDK 就只改这一段。
	// --------------------------------------------------------
	callers := buildCallers()

	// --------------------------------------------------------
	// 4. 评估 Pipeline
	// --------------------------------------------------------
	pipeline := &df.Pipeline{
		HardStages: df.WrapHardGates(g.HardGates),
		SoftStages: []df.Stage{&df.LLMJudgeStage{
			Goal:           g,
			Caller:         callers.judge,
			FilesToInclude: collectKeyFiles,
		}},
	}

	// --------------------------------------------------------
	// 5. 共享记忆(强烈推荐启用)
	// --------------------------------------------------------
	sm, err := df.NewSharedMemory(df.SharedMemoryConfig{
		SkillsDir: filepath.Join(*workRoot, ".shared", "skills"),
		Goal:      g,
	})
	if err != nil {
		log.Fatalf("shared memory: %v", err)
	}

	// --------------------------------------------------------
	// 6. Forge
	// --------------------------------------------------------
	f, err := df.New(df.Config{
		Goal:           g,
		SeedDir:        *seedDir,
		WorkRoot:       *workRoot,
		TriageCaller:   callers.triage,
		MainCaller:     callers.main,
		ToolBuilder:    callers.toolBuilder,
		Pipeline:       pipeline,
		SharedMemory:   sm,
		SkillDistiller: callers.distiller,
		Logger: func(event string, fields map[string]any) {
			log.Printf("[%s] %v", event, fields)
		},
	})
	if err != nil {
		log.Fatalf("forge new: %v", err)
	}

	// --------------------------------------------------------
	// 7. Run
	// --------------------------------------------------------
	ctx := context.Background()
	res, err := f.Run(ctx, df.Budget{C: *C, L: *L, K: *K})
	if err != nil {
		log.Fatalf("forge run: %v", err)
	}

	// --------------------------------------------------------
	// 8. 输出
	// --------------------------------------------------------
	if err := res.Trajectory.Save(*workRoot); err != nil {
		log.Printf("save trajectory: %v", err)
	}

	fmt.Printf("\n=== Pareto Frontier (%d candidates) ===\n", len(res.Frontier))
	for _, c := range res.Frontier {
		fmt.Printf("\n  %s @ L%d gen%d\n  workdir: %s\n",
			c.ID, c.Lineage, c.Generation, c.WorkDir)
		for k, v := range c.Score.Soft {
			fmt.Printf("    %-30s %.4f\n", k, v)
		}
	}
	fmt.Println("\nReview the workdirs above. Pick the one that best matches your taste.")
}

// ============================================================
// LLM Caller 集合
// ============================================================
//
// 5 个回调全部接到 doptime/llm。这一段是**唯一**与 LLM SDK 强耦合的代码。
// 换 SDK 时只改 buildCallers() 一个函数。

type callerSet struct {
	triage      gateway.TriageCaller
	main        func(systemPrompt, userPrompt string, toolList []any) error
	judge       df.JudgeCaller
	toolBuilder tools.ToolBuilder
	distiller   df.SkillDistiller
}

func buildCallers() *callerSet {
	return &callerSet{
		triage:      doptimeTriageCaller,
		main:        doptimeMainCaller,
		judge:       doptimeJudgeCaller,
		toolBuilder: doptimeToolBuilder(),
		distiller:   doptimeDistiller,
	}
}

// doptimeToolBuilder 把 dopharness 的 7 种 ToolPayload 桥接到 llm.NewTool。
// 与 dopharness README 里的写法一致。
func doptimeToolBuilder() tools.ToolBuilder {
	return tools.ToolBuilderFunc(func(name, desc string, handler any) any {
		switch h := handler.(type) {
		case func(*tools.ModifyChunkPayload):
			return llm.NewTool(name, desc, h)
		case func(*tools.DeleteChunkPayload):
			return llm.NewTool(name, desc, h)
		case func(*tools.AddChunkPayload):
			return llm.NewTool(name, desc, h)
		case func(*tools.CreateFilePayload):
			return llm.NewTool(name, desc, h)
		case func(*tools.DeleteFilePayload):
			return llm.NewTool(name, desc, h)
		case func(*tools.ReadChunkPayload):
			return llm.NewTool(name, desc, h)
		case func(*tools.SearchChunksByNamePayload):
			return llm.NewTool(name, desc, h)
		}
		panic(fmt.Sprintf("unknown handler type: %T", handler))
	})
}

// doptimeTriageCaller 是 dopharness Pass1 的 LLM 桩。用便宜模型。
func doptimeTriageCaller(p gateway.TriagePromptParams, sink func(*gateway.TriageDecisionPayload)) error {
	tpl := template.Must(template.New("triage").Parse(`你是代码上下文筛选助手。用户任务:
{{.UserPrompt}}

候选 chunk(格式: id | kind | name | signature | refs):
{{range .Views}}{{.ID}} | {{.Kind}} | {{.Name}} | {{.Signature}} | {{.Refs}}
{{end}}

对每个 chunk 通过 TriageDecision 工具提交判断:
- 即将被修改的 → FULL
- 被将要修改的代码引用(参数类型、调用关系、文档章节互引) → SKELETON
- 其余 → IGNORE`))

	var payload *gateway.TriageDecisionPayload
	tool := llm.NewTool("TriageDecision", "提交 chunk 裁定", func(p *gateway.TriageDecisionPayload) {
		payload = p
	})
	// TODO: 这里用 ModelDefault,生产应换成更便宜的小模型(如 Haiku/4o-mini)。
	agent := llm.NewAgent(tpl, tool).UseModels(llm.ModelDefault)
	if err := agent.Call(map[string]any{
		"UserPrompt": p.UserPrompt,
		"Views":      p.Views,
	}); err != nil {
		return err
	}
	if payload == nil {
		return fmt.Errorf("triage: LLM did not call TriageDecision tool")
	}
	sink(payload)
	return nil
}

// doptimeMainCaller 是 dopharness 主修改轮的 LLM 桩。用最强模型。
func doptimeMainCaller(systemPrompt, userPrompt string, toolList []any) error {
	tpl := template.Must(template.New("main").Parse(`{{.UserPrompt}}`))

	llmTools := make([]llm.ToolInterface, 0, len(toolList))
	for _, t := range toolList {
		llmTools = append(llmTools, t.(llm.ToolInterface))
	}

	agent := llm.NewAgent(tpl).UseModels(llm.ModelDefault)
	for _, t := range llmTools {
		agent.UseTools(t)
	}
	// doptime/llm 当前 Agent 只取 user prompt;system 拼前缀
	return agent.Call(map[string]any{
		"UserPrompt": systemPrompt + "\n\n" + userPrompt,
	})
}

// doptimeJudgeCaller 是 LLMJudgeStage 的法官桩。用比 main 便宜半档的模型。
//
// 关键约定:Caller 必须返回 LLM 的**原始回复文本**,LLMJudgeStage 内部会从中提取
// ```json fenced block。所以这里要让 LLM 直接说话,不要走 ToolCall。
func doptimeJudgeCaller(ctx context.Context, prompt string) (string, error) {
	tpl := template.Must(template.New("judge").Parse(`{{.Prompt}}`))

	// doptime/llm 当前没有"纯文本回复"模式,我们绕一下:声明一个 SubmitJudgement
	// 工具收 raw_text 字符串。LLM 会把它的 JSON 包在一次 tool call 里。
	var rawReply string
	tool := llm.NewTool(
		"SubmitJudgement",
		"提交完整的 JSON 评分(包在 ```json fenced block 内的字符串)",
		func(p *struct {
			RawText string `json:"raw_text" comment:"the full JSON response, fenced or not"`
		}) {
			rawReply = p.RawText
		},
	)
	agent := llm.NewAgent(tpl, tool).UseModels(llm.ModelDefault)
	if err := agent.Call(map[string]any{"Prompt": prompt}); err != nil {
		return "", err
	}
	if rawReply == "" {
		return "", fmt.Errorf("judge: LLM returned empty raw_text")
	}
	return rawReply, nil
}

// doptimeDistiller 蒸馏 winner → skill。可空(此时只写 L4)。
//
// 实现思路:让 LLM 看 parent → winner 的 git diff,写一段"这种改法在什么情况下管用"
// 的 SOP 文本。返回 (key, content)。key 用来生成稳定文件名。
//
// v0:简单实现,直接把 winner.Score.Notes 当 skill 内容。生产环境应该单独跑一次 LLM。
func doptimeDistiller(ctx context.Context, winner, parent *df.Candidate) (key, content string, err error) {
	if winner.Score == nil || winner.Score.Notes == "" {
		return "", "", nil // 没东西可蒸馏
	}
	// 简易版:把法官的 critique 包成 skill。生产应让 LLM 提炼一段更通用的 SOP。
	key = fmt.Sprintf("gen%d_lineage%d", winner.Generation, winner.Lineage)
	var sb strings.Builder
	sb.WriteString("# Skill from ")
	sb.WriteString(winner.ID)
	sb.WriteString("\n\n_Auto-distilled. Replace with LLM-generated SOP for better quality._\n\n## Context\n")
	sb.WriteString(fmt.Sprintf("This skill was extracted when candidate %s entered the Pareto frontier of lineage %d at generation %d.\n\n",
		winner.ID, winner.Lineage, winner.Generation))
	sb.WriteString("## What seemed to work (judge's notes)\n\n")
	sb.WriteString(winner.Score.Notes)
	sb.WriteString("\n")
	return key, sb.String(), nil
}

// ============================================================
// 工具函数
// ============================================================

// collectKeyFiles 扫候选解 WorkDir,把 LLM 法官该看的文件读出来。
//
// 现行实现:整文件读取 + 4KB 截断。这里**有一个已知低效点**:dopforge 没有利用
// dopharness 的 chunk 化能力。理想做法是只把"父代到子代发生变化的 chunk"喂给法官,
// token 用量可以降一个数量级,而且评分更稳定。
// 见 README 末尾"已知改进项"。
func collectKeyFiles(workDir string) (map[string]string, error) {
	out := map[string]string{}
	err := filepath.Walk(workDir, func(p string, info os.FileInfo, err error) error {
		if err != nil || info.IsDir() {
			return err
		}
		rel, _ := filepath.Rel(workDir, p)
		if len(rel) > 0 && rel[0] == '.' {
			return nil // 跳过 .dopharness、.git 等
		}
		ext := filepath.Ext(p)
		switch ext {
		case ".go", ".md", ".markdown", ".html", ".js", ".ts", ".css", ".sh":
		default:
			return nil
		}
		data, err := os.ReadFile(p)
		if err != nil {
			return nil
		}
		s := string(data)
		if len(s) > 4096 {
			s = s[:2048] + "\n... [truncated] ...\n" + s[len(s)-2048:]
		}
		out[rel] = s
		return nil
	})
	return out, err
}

// ensureSeed 保证 seed 目录存在且至少有一个能跑通"go build"的最小骨架。
// 已存在 go.mod 的话就不改动,假定用户准备好了。
func ensureSeed(dir string) error {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
		return nil
	}
	if err := os.WriteFile(filepath.Join(dir, "go.mod"),
		[]byte("module seed\n\ngo 1.22\n"), 0o644); err != nil {
		return err
	}
	if err := os.WriteFile(filepath.Join(dir, "main.go"),
		[]byte("package main\n\nfunc main() {}\n"), 0o644); err != nil {
		return err
	}
	if err := os.WriteFile(filepath.Join(dir, "run.sh"),
		[]byte("#!/bin/bash\necho 'seed stub - replace me'\n"), 0o755); err != nil {
		return err
	}
	if err := os.WriteFile(filepath.Join(dir, "README.md"),
		[]byte("# Seed project\n\nReplace this with your own initial implementation.\n"), 0o644); err != nil {
		return err
	}
	// 验证 go build 能过(失败时打 warning,不阻断;让用户在跑搜索时自己看)
	cmd := exec.Command("go", "build", "./...")
	cmd.Dir = dir
	if out, err := cmd.CombinedOutput(); err != nil {
		log.Printf("warn: seed `go build` failed: %v\n%s", err, out)
	}
	return nil
}
