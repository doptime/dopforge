package dopforge

import (
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"sync/atomic"
	"time"
)

// ============================================================
// Candidate:一份正在演化的方法论实现
// ============================================================
//
// 一个 Candidate 等于一个独立工作目录(WorkDir),里面是真实可运行的代码。
// 它有稳定 ID、出身记录(父代是谁)、分数(可能为 nil 表示还没评估)。
//
// 关键不变量:
//   - WorkDir 是一个 cp -a 出来的独立目录,自己拥有 .dopharness/ 子目录
//   - Parent 永远指向"创造我的那一份",根候选解 Parent == nil
//   - Score 在评估前为 nil,评估后不应再被改写

var idCounter uint64

type Candidate struct {
	ID         string     // 全局唯一,形如 "c00007"
	Lineage    int        // 属于第几条 lineage(C 路并行的某一路)
	Generation int        // 在 lineage 内的代数(L 轮的第几轮)
	Parent     *Candidate // 父代;根候选解为 nil
	WorkDir    string     // 独立工作目录的绝对路径

	// PromptLog 记录 mutate 这一步送给 LLM 的 user prompt。
	// 不参与决策,仅用于审计/调试。
	PromptLog string

	// Score 由 Pipeline.Evaluate 写入;nil 表示尚未评估。
	Score *Score

	CreatedAt time.Time
}

// NewCandidate 构造一个新候选解(WorkDir 由 Workspace 决定)。
func NewCandidate(lineage, generation int, parent *Candidate) *Candidate {
	id := atomic.AddUint64(&idCounter, 1)
	return &Candidate{
		ID:         fmt.Sprintf("c%05d", id),
		Lineage:    lineage,
		Generation: generation,
		Parent:     parent,
		CreatedAt:  time.Now(),
	}
}

// ============================================================
// Score:多维评分向量(无标量)
// ============================================================
//
// 设计核心:不强行把多个维度合成一个标量。Pareto 在多维向量上比较,
// 只在出现"严格被支配"关系时淘汰候选。

type Score struct {
	// Hard 列出每条硬门槛的检查结果。任意 Err != nil 视为整体不通过。
	Hard []GateResult

	// Soft 是按 Goal.Dimensions 给出的多维分数。键名严格对齐 Dimension.Name。
	Soft map[string]float64

	// Notes 是评估器(典型是 LLM judge)给出的文字反馈。会被原样回灌进
	// 下一轮 mutate 的 user prompt,引导改进方向。
	Notes string
}

// GateResult 是一条硬门槛的判定结果。
type GateResult struct {
	Name string
	Err  error // nil 表示通过
}

// Passed 当且仅当所有硬门槛都通过。
func (s *Score) Passed() bool {
	if s == nil {
		return false
	}
	for _, g := range s.Hard {
		if g.Err != nil {
			return false
		}
	}
	return true
}

// Dominates 判断 s 在帕累托意义上是否支配 other。
// 规则:每一维都不差于 other,且至少一维严格优于。
// 没通过硬门槛的 Score 不参与帕累托比较。
func (s *Score) Dominates(other *Score, dims []Dimension) bool {
	if s == nil || other == nil {
		return false
	}
	if !s.Passed() || !other.Passed() {
		return false
	}
	strictlyBetter := false
	for _, d := range dims {
		a, b := s.Soft[d.Name], other.Soft[d.Name]
		if d.HigherIsBetter {
			if a < b {
				return false
			}
			if a > b {
				strictlyBetter = true
			}
		} else {
			if a > b {
				return false
			}
			if a < b {
				strictlyBetter = true
			}
		}
	}
	return strictlyBetter
}

// ============================================================
// Workspace:候选解物理目录管理
// ============================================================
//
// 设计要点:
//   - 每个 Candidate 在磁盘上是 <Root>/L<lineage>/<id>/ 一份完整副本
//   - fork 时 cp -a,不做 git 也不做 union mount —— 简单且对所有语言都 work
//   - 副本里包含 .dopharness/ chunk store,所以增量索引能延续
//   - 副本里也包含 .dopharness/memory/sessions/,所以 lineage 内 L4 历史
//     自动沿父代 → 子代继承(这是 dopharness 4 层记忆中 L4 的天然继承机制)

type Workspace struct {
	Root    string // 例:./.forge_work
	SeedDir string // 第一代候选解从这里 cp;空则白板起步
}

// NewWorkspace 构造并保证 Root 存在。
func NewWorkspace(root, seedDir string) (*Workspace, error) {
	if err := os.MkdirAll(root, 0o755); err != nil {
		return nil, fmt.Errorf("workspace: mkdir root: %w", err)
	}
	if seedDir != "" {
		if _, err := os.Stat(seedDir); err != nil {
			return nil, fmt.Errorf("workspace: seed dir %s: %w", seedDir, err)
		}
	}
	return &Workspace{Root: root, SeedDir: seedDir}, nil
}

// Materialize 给 Candidate 生成 WorkDir 并把内容拷进去。
//
// 拷贝源:有 Parent 时取 Parent.WorkDir,否则取 SeedDir,都没有就建空目录。
// .dopharness/ 子目录会一并继承,使 chunk store 和 L4 sessions 都跨代连续。
func (w *Workspace) Materialize(c *Candidate) error {
	dst := filepath.Join(w.Root, fmt.Sprintf("L%d", c.Lineage), c.ID)
	if err := os.MkdirAll(dst, 0o755); err != nil {
		return fmt.Errorf("workspace: mkdir %s: %w", dst, err)
	}
	var src string
	switch {
	case c.Parent != nil:
		src = c.Parent.WorkDir
	case w.SeedDir != "":
		src = w.SeedDir
	}
	if src != "" {
		if err := copyDir(src, dst); err != nil {
			return fmt.Errorf("workspace: copy %s -> %s: %w", src, dst, err)
		}
	}
	c.WorkDir = dst
	return nil
}

// Cleanup 删除 Candidate 的物理目录。**不要**清理还在被引用为 Parent 的候选,
// 否则其后代失去拷贝源。search.Run 已经处理好了这一约束。
func (w *Workspace) Cleanup(c *Candidate) error {
	if c.WorkDir == "" {
		return nil
	}
	return os.RemoveAll(c.WorkDir)
}

// copyDir 优先用系统 cp -a,fallback 到纯 Go 递归拷贝。
func copyDir(src, dst string) error {
	if _, err := exec.LookPath("cp"); err == nil {
		cmd := exec.Command("cp", "-a", src+"/.", dst)
		if err := cmd.Run(); err == nil {
			return nil
		}
	}
	return filepath.Walk(src, func(p string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		rel, _ := filepath.Rel(src, p)
		target := filepath.Join(dst, rel)
		if info.IsDir() {
			return os.MkdirAll(target, info.Mode())
		}
		return copyFile(p, target, info.Mode())
	})
}

func copyFile(src, dst string, mode os.FileMode) error {
	in, err := os.Open(src)
	if err != nil {
		return err
	}
	defer in.Close()
	if err := os.MkdirAll(filepath.Dir(dst), 0o755); err != nil {
		return err
	}
	out, err := os.OpenFile(dst, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, mode)
	if err != nil {
		return err
	}
	defer out.Close()
	_, err = io.Copy(out, in)
	return err
}
