package dopforge

import (
	"context"
	"crypto/sha1"
	"encoding/hex"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/doptime/dopharness/gateway"
	"github.com/doptime/dopharness/harness"
	"github.com/doptime/dopharness/memory"
	"github.com/doptime/dopharness/tools"
)

// ============================================================
// Forge:dopforge 对外的最终门面
// ============================================================
//
// 一个 Forge 实例聚合 Goal、Workspace、Pipeline、SharedMemory 和 dopharness Caller。
// Forge.Run 把这一切串成一次 C×L×K 方法论迭代搜索。

type Forge struct {
	cfg       Config
	workspace *Workspace
}

// Config 是 Forge 的构造参数。
type Config struct {
	Goal *Goal

	// SeedDir:第一代候选解从这里 cp -a。空则白板起步。
	// 对方法论任务,推荐放一个只包含 METHOD.md 的最小种子目录。
	SeedDir string

	// WorkRoot:所有候选解工作目录的根。会自动创建。
	WorkRoot string

	// dopharness 的三个 Caller。
	TriageCaller gateway.TriageCaller
	ExpandCaller gateway.ExpandCaller // 可空(EnableExpand=false 时)
	MainCaller   harness.MainCaller

	// EnableExpand 透传给 dopharness。默认 false。
	EnableExpand bool

	// ToolBuilder 桥接 dopharness 工具到具体 LLM SDK。
	ToolBuilder tools.ToolBuilder

	// Pipeline 是评估器(必填)。
	Pipeline *Pipeline

	// SharedMemory 接入 dopharness 的 L0/L2/L3 记忆,所有 candidate 共享迭代纪律、
	// Goal 事实和跨 lineage 的方法论改进 SOP。强烈推荐启用。
	SharedMemory *SharedMemory

	// SkillDistiller 在某 candidate 进入 lineage BestEver 时被调用,把有效改法蒸馏成
	// 一篇 .md skill 写入共享 SkillsDir。
	SkillDistiller SkillDistiller

	Logger func(event string, fields map[string]any)
}

// Budget 是搜索预算。
type Budget struct {
	C int // 并行 lineage 数,典型 4
	L int // 每条 lineage 代数,典型 6
	K int // 每代孩子数,典型 3
}

// New 构造一个 Forge,做必要的 sanity check。
func New(cfg Config) (*Forge, error) {
	if cfg.Goal == nil {
		return nil, fmt.Errorf("forge: Goal is required")
	}
	if cfg.WorkRoot == "" {
		return nil, fmt.Errorf("forge: WorkRoot is required")
	}
	if cfg.TriageCaller == nil {
		return nil, fmt.Errorf("forge: TriageCaller is required")
	}
	if cfg.MainCaller == nil {
		return nil, fmt.Errorf("forge: MainCaller is required")
	}
	if cfg.ToolBuilder == nil {
		return nil, fmt.Errorf("forge: ToolBuilder is required")
	}
	if cfg.Pipeline == nil {
		return nil, fmt.Errorf("forge: Pipeline is required")
	}
	if cfg.EnableExpand && cfg.ExpandCaller == nil {
		return nil, fmt.Errorf("forge: ExpandCaller required when EnableExpand=true")
	}

	ws, err := NewWorkspace(cfg.WorkRoot, cfg.SeedDir)
	if err != nil {
		return nil, err
	}
	cfg.Pipeline.Goal = cfg.Goal

	return &Forge{cfg: cfg, workspace: ws}, nil
}

// Run 启动一次完整搜索。
func (f *Forge) Run(ctx context.Context, b Budget) (*SearchResult, error) {
	mutator := f.makeMutator()

	var onImprove func(c *Candidate)
	if f.cfg.SharedMemory != nil {
		onImprove = func(c *Candidate) {
			err := f.cfg.SharedMemory.MaybeRecordSkill(ctx, c, c.Parent, f.cfg.SkillDistiller)
			if err != nil && f.cfg.Logger != nil {
				f.cfg.Logger("skill_record_err", map[string]any{
					"candidate": c.ID, "err": err.Error(),
				})
			}
		}
	}

	scfg := SearchConfig{
		Goal:          f.cfg.Goal,
		Workspace:     f.workspace,
		Mutator:       mutator,
		Pipeline:      f.cfg.Pipeline,
		C:             b.C,
		L:             b.L,
		K:             b.K,
		Logger:        f.cfg.Logger,
		CleanupLosers: true,
		Plateau:       StallAfter{N: defaultStallN(b.L)},
		OnImprovement: onImprove,
	}
	return RunSearch(ctx, scfg)
}

func defaultStallN(L int) int {
	switch {
	case L <= 4:
		return 3
	case L <= 8:
		return 4
	default:
		return 5
	}
}

// makeMutator 把 dopharness 包成 Mutator。这是 dopforge 与 dopharness 的唯一接触面。
func (f *Forge) makeMutator() Mutator {
	return func(ctx context.Context, child, parent *Candidate) error {
		hcfg := harness.Config{
			ProjectRoot:  child.WorkDir,
			TriageCaller: f.cfg.TriageCaller,
			ExpandCaller: f.cfg.ExpandCaller,
			MainCaller:   f.cfg.MainCaller,
			EnableExpand: f.cfg.EnableExpand,
			MaxRetries:   3,
		}
		if f.cfg.Logger != nil {
			hcfg.Logger = func(ev string, fields map[string]any) {
				if fields == nil {
					fields = map[string]any{}
				}
				fields["candidate"] = child.ID
				f.cfg.Logger("dopharness."+ev, fields)
			}
		}

		h, err := harness.New(hcfg)
		if err != nil {
			return fmt.Errorf("mutate %s: harness.New: %w", child.ID, err)
		}
		if f.cfg.SharedMemory != nil {
			f.cfg.SharedMemory.ApplyTo(h)
		}
		if _, err := h.Index(ctx); err != nil {
			return fmt.Errorf("mutate %s: index: %w", child.ID, err)
		}
		h.AsLLMTools(f.cfg.ToolBuilder)

		prompt := buildMutatorPrompt(f.cfg.Goal, parent, f.cfg.SharedMemory != nil)
		child.PromptLog = prompt

		_, err = h.Run(ctx, prompt)
		if err != nil {
			return fmt.Errorf("mutate %s: run: %w", child.ID, err)
		}
		return nil
	}
}

func buildMutatorPrompt(g *Goal, parent *Candidate, memoryActive bool) string {
	_ = memoryActive // 保留参数以维持兼容签名;goal 一律写入 prompt
	var sb strings.Builder
	sb.WriteString("<goal>\n")
	sb.WriteString(g.Description)
	sb.WriteString("\n</goal>\n\n")
	if parent != nil && parent.Score != nil && parent.Score.Notes != "" {
		sb.WriteString("<previous_evaluation>\n")
		sb.WriteString("Your previous methodology document was scored. Below is the judge's feedback. ")
		sb.WriteString("Make a substantive improvement, not a cosmetic change.\n\n")
		sb.WriteString(parent.Score.Notes)
		sb.WriteString("\n</previous_evaluation>\n\n")
	}
	sb.WriteString("<anti_patterns>\n")
	sb.WriteString("- Renaming, reordering, or reformatting sections without changing what they teach\n")
	sb.WriteString("- Sprinkling terminology (元认知, 迁移, 刻意练习, 抗遗忘, 反馈闭环) without procedures, examples, or templates that operationalize them\n")
	sb.WriteString("- Replacing concrete examples or templates with abstract principles\n")
	sb.WriteString("- Padding with bullet lists where prose, tables, or worked examples would teach more\n")
	sb.WriteString("</anti_patterns>\n\n")
	sb.WriteString("<instruction>\n")
	sb.WriteString("Improve the methodology Markdown document to better satisfy the goal. ")
	sb.WriteString("The desired output is documentation only, preferably a single METHOD.md unless the goal says otherwise. ")
	sb.WriteString("Do not create source code, Go modules, shell scripts, package manifests, Dockerfiles, test runners, sandboxes, or runnable applications. ")
	sb.WriteString("Use the provided tools only to edit Markdown content and supporting Markdown notes. ")
	sb.WriteString("If you create a file, create only .md/.markdown files. ")
	sb.WriteString("Make a meaningful improvement to structure, concepts, examples, checklists, tradeoffs, anti-patterns, or review loops. ")
	sb.WriteString("Prefer precise sections, operational procedures, and reusable methodology over generic inspirational prose. ")
	sb.WriteString("\n</instruction>")
	return sb.String()
}

// ============================================================
// SharedMemory:接入 dopharness 4 层记忆,真正解锁跨 lineage 学习
// ============================================================

const dopforgeMetaRules = `# dopforge methodology iteration rules

You are operating inside a methodology-document search loop. Your output will be
evaluated, scored, and either advanced as the next generation or discarded.

1. **Documentation only.** The target artifact is methodology Markdown. Do not
   create runnable software projects, Go modules, shell scripts, sandboxes,
   package manifests, Dockerfiles, test runners, or app scaffolding unless the
   goal explicitly asks for them.

2. **Address concrete critique first.** If <previous_evaluation> is present,
   the judge's notes name specific gaps. Fix those before pursuing new ideas.

3. **Substantive over cosmetic.** Renaming sections, reformatting, or adding
   motivational prose rarely moves the score. Improve conceptual structure,
   decision procedures, examples, exercises, feedback loops, or failure modes.

4. **Preserve useful prior structure.** Do not undo prior progress unless you
   can articulate why the previous direction was wrong. Read task skills and L4
   session records before reverting.

5. **Use tools surgically.** Multiple markdown chunk edits are normal. Prefer
   modifying or adding focused sections over rewriting the whole document.

6. **When stuck, search.** If <project_context> is thin, use search/read tools to
   inspect existing Markdown before editing.
`

// SharedMemoryConfig 是构造共享记忆的输入。
type SharedMemoryConfig struct {
	// SkillsDir 是 L3 共享技能目录的绝对路径。所有 candidate 共用此目录。
	SkillsDir string
	// MetaRulesOverride 替换默认 dopforgeMetaRules。空则用默认。
	MetaRulesOverride string
	// Goal 用来渲染 L2 事实段。
	Goal *Goal
}

// SharedMemory 是被注入到每个 dopharness Harness 的 L0/L2/L3。
type SharedMemory struct {
	cfg SharedMemoryConfig

	l0 memory.Layer
	l2 memory.Layer
	l3 *memory.TaskSkillsLayer

	mu       sync.Mutex
	skillSet map[string]struct{}
}

// NewSharedMemory 构造并就地准备 SkillsDir。
func NewSharedMemory(cfg SharedMemoryConfig) (*SharedMemory, error) {
	if cfg.SkillsDir == "" {
		return nil, fmt.Errorf("forge: SharedMemoryConfig.SkillsDir is required")
	}
	if cfg.Goal == nil {
		return nil, fmt.Errorf("forge: SharedMemoryConfig.Goal is required")
	}
	if err := os.MkdirAll(cfg.SkillsDir, 0o755); err != nil {
		return nil, fmt.Errorf("forge: mkdir SkillsDir: %w", err)
	}

	rules := cfg.MetaRulesOverride
	if rules == "" {
		rules = dopforgeMetaRules
	}

	return &SharedMemory{
		cfg:      cfg,
		l0:       &memory.MetaRulesLayer{Fallback: rules},
		l2:       &memory.GlobalFactsLayer{Fallback: renderGoalAsFacts(cfg.Goal)},
		l3:       &memory.TaskSkillsLayer{Dir: cfg.SkillsDir},
		skillSet: make(map[string]struct{}),
	}, nil
}

// ApplyTo 把共享层覆盖到一个 Harness 的 Memory 上。
func (s *SharedMemory) ApplyTo(h *harness.Harness) {
	mem := h.Memory()
	if mem == nil {
		return
	}
	mem.L0 = s.l0
	mem.L2 = s.l2
	mem.L3 = s.l3
}

// RecordSkill 把“什么改动有效”的总结写进共享 SkillsDir。
func (s *SharedMemory) RecordSkill(key, content string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.skillSet[key] = struct{}{}
	fname := sanitizeSkillFilename(key)
	path := filepath.Join(s.cfg.SkillsDir, fname+".md")
	return os.WriteFile(path, []byte(content), 0o644)
}

// SessionRecord 把一段迭代摘要写进 candidate 自己的 L4。
func (s *SharedMemory) SessionRecord(c *Candidate, summary string) error {
	if c.WorkDir == "" {
		return fmt.Errorf("session_record: candidate %s has no WorkDir", c.ID)
	}
	dir := filepath.Join(c.WorkDir, ".dopharness", "memory", "sessions")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	stamp := time.Now().UTC().Format("20060102T150405")
	fname := fmt.Sprintf("%s_%s.md", stamp, c.ID)
	return os.WriteFile(filepath.Join(dir, fname), []byte(summary), 0o644)
}

// SkillDistiller 是用户提供的“看 winner 的产出蒸馏出 skill 文本”的回调。
type SkillDistiller func(ctx context.Context, winner, parent *Candidate) (key, content string, err error)

// MaybeRecordSkill 在 winner 进入 BestEver 时蒸馏成 skill。
func (s *SharedMemory) MaybeRecordSkill(
	ctx context.Context, winner, parent *Candidate, distiller SkillDistiller,
) error {
	if winner.Score != nil && winner.Score.Notes != "" {
		summary := fmt.Sprintf("# %s (gen %d)\n\n## Judge feedback\n\n%s\n",
			winner.ID, winner.Generation, winner.Score.Notes)
		if err := s.SessionRecord(winner, summary); err != nil {
			return err
		}
	}
	if distiller == nil {
		return nil
	}
	key, content, err := distiller(ctx, winner, parent)
	if err != nil {
		return fmt.Errorf("distill skill for %s: %w", winner.ID, err)
	}
	if key == "" || content == "" {
		return nil
	}
	return s.RecordSkill(key, content)
}

// renderGoalAsFacts 把 Goal 序列化成 L2 风格的事实文本。
func renderGoalAsFacts(g *Goal) string {
	var sb strings.Builder
	sb.WriteString("# Methodology Goal\n\n")
	sb.WriteString(g.Description)
	sb.WriteString("\n\n# Evaluation Dimensions\n\n")
	for _, d := range g.Dimensions {
		dir := "higher is better"
		if !d.HigherIsBetter {
			dir = "lower is better"
		}
		sb.WriteString(fmt.Sprintf("- **%s** (%s)\n", d.Name, dir))
	}
	if len(g.HardGates) > 0 {
		sb.WriteString("\n# Hard Gates (must pass)\n\n")
		for _, h := range g.HardGates {
			sb.WriteString("- " + h.Name() + "\n")
		}
	}
	if g.LLMJudgeRubric != "" {
		sb.WriteString("\n# Judge Rubric\n\n")
		sb.WriteString(g.LLMJudgeRubric)
		sb.WriteString("\n")
	}
	return sb.String()
}

func sanitizeSkillFilename(key string) string {
	var clean strings.Builder
	for _, r := range key {
		switch {
		case r >= 'a' && r <= 'z', r >= 'A' && r <= 'Z', r >= '0' && r <= '9':
			clean.WriteRune(r)
		case r == '_' || r == '-' || r == ' ':
			clean.WriteRune('_')
		}
	}
	prefix := clean.String()
	if len(prefix) > 40 {
		prefix = prefix[:40]
	}
	h := sha1.Sum([]byte(key))
	return prefix + "_" + hex.EncodeToString(h[:4])
}
