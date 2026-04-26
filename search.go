package dopforge

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"sync"
	"time"
)

// ============================================================
// Lineage:C 路并行的某一路
// ============================================================
//
// 每个 L 步骤里:在 Latest() 上拷出 K 份 → 各自 mutate → 从 K 个孩子里挑最好,
// Append 推进 lineage。
//
// SimpleTES 论文中"轨迹级奖励"的本地落地:BestEver 不是当前主干,而是 Pareto
// 历史峰值集合。某代主干被劣质孩子带偏不要紧,只要后代翻盘,峰值仍在 BestEver。
// PlateauPolicy 据此判断 lineage 是否真的停滞,而非简单看主干分数。

type Lineage struct {
	ID int

	mu      sync.Mutex
	history []*Candidate // history[0] 是种子,后续每代追加一个"当代赢家"

	// BestEver 是这条 lineage 的 Pareto 历史峰值集合(可能多个非支配点)。
	BestEver []*Candidate

	// lastImprovementGen 记录最近一次 BestEver 扩张时的代数。
	// PlateauPolicy.StallAfter 据此判断停滞。
	lastImprovementGen int

	// stopped 由 search.RunSearch 在 Plateau 命中时设置,后续代不再生成孩子。
	stopped bool

	// OnImprovement 在 Append 检测到 BestEver 扩张时被调用(在锁外)。
	// 典型用途:Forge 蒸馏 skill 写共享 L3 记忆。回调应快速,慢操作请异步处理。
	OnImprovement func(c *Candidate)
}

func NewLineage(id int) *Lineage { return &Lineage{ID: id} }

// Latest 返回当前主干末端。空 lineage 返回 nil。
func (l *Lineage) Latest() *Candidate {
	l.mu.Lock()
	defer l.mu.Unlock()
	if len(l.history) == 0 {
		return nil
	}
	return l.history[len(l.history)-1]
}

// History 返回主干的只读快照。
func (l *Lineage) History() []*Candidate {
	l.mu.Lock()
	defer l.mu.Unlock()
	out := make([]*Candidate, len(l.history))
	copy(out, l.history)
	return out
}

func (l *Lineage) LastImprovementGen() int {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.lastImprovementGen
}

func (l *Lineage) Stopped() bool {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.stopped
}

func (l *Lineage) MarkStopped() {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.stopped = true
}

// Append 推进一步,同时尝试把 c 加入 BestEver。
//
// 副作用:如果 BestEver 因 c 而扩张,记录 lastImprovementGen 并触发 OnImprovement。
func (l *Lineage) Append(c *Candidate, dims []Dimension) {
	l.mu.Lock()
	gen := len(l.history)
	l.history = append(l.history, c)
	l.BestEver = updateFrontier(l.BestEver, c, dims)

	improved := false
	for _, x := range l.BestEver {
		if x == c {
			l.lastImprovementGen = gen
			improved = true
			break
		}
	}
	cb := l.OnImprovement
	l.mu.Unlock()

	if improved && cb != nil {
		cb(c)
	}
}

// ============================================================
// Pareto 前沿与 batch 内决策
// ============================================================

// updateFrontier 把 c 加入 frontier(Pareto 前沿),同时剔除被 c 支配的旧成员。
//
// 没通过硬门槛的 c 不加入,直接返回原 frontier。
func updateFrontier(frontier []*Candidate, c *Candidate, dims []Dimension) []*Candidate {
	if c == nil || c.Score == nil || !c.Score.Passed() {
		return frontier
	}
	kept := make([]*Candidate, 0, len(frontier)+1)
	dominated := false
	for _, x := range frontier {
		switch {
		case x.Score.Dominates(c.Score, dims):
			kept = append(kept, x)
			dominated = true
		case c.Score.Dominates(x.Score, dims):
			// x 被 c 干掉,跳过
		default:
			kept = append(kept, x)
		}
	}
	if !dominated {
		kept = append(kept, c)
	}
	return kept
}

// PickBestInBatch 在同一父代生出的 K 个兄弟里挑一个推进 lineage 主干。
//
// 策略:
//   1. 优先从通过硬门槛的子集里挑;若全没通过,fallback 取硬错误最少的
//   2. 通过者中:取 Pareto 非支配子集
//   3. 子集单点直接返回
//   4. 子集多点:按 Goal.Dimension.Weight 加权打分,选最高
//
// 第 4 步是搜索过程中**唯一**一处把多维折成标量的地方,但只在 batch 内 tiebreak 时用,
// 不影响全局 Pareto 前沿。即便这里挑错了,GlobalFrontier 仍兜底。
func PickBestInBatch(batch []*Candidate, dims []Dimension) *Candidate {
	if len(batch) == 0 {
		return nil
	}
	var passed []*Candidate
	for _, c := range batch {
		if c.Score != nil && c.Score.Passed() {
			passed = append(passed, c)
		}
	}
	if len(passed) == 0 {
		// 全没通过:挑硬错误最少的(便于下一轮尽快爬出深坑)
		sort.SliceStable(batch, func(i, j int) bool {
			return countHardFail(batch[i]) < countHardFail(batch[j])
		})
		return batch[0]
	}

	frontier := []*Candidate{}
	for _, c := range passed {
		frontier = updateFrontier(frontier, c, dims)
	}
	if len(frontier) == 1 {
		return frontier[0]
	}

	sort.SliceStable(frontier, func(i, j int) bool {
		return weightedScore(frontier[i], dims) > weightedScore(frontier[j], dims)
	})
	return frontier[0]
}

func countHardFail(c *Candidate) int {
	if c == nil || c.Score == nil {
		return 1 << 30
	}
	n := 0
	for _, h := range c.Score.Hard {
		if h.Err != nil {
			n++
		}
	}
	return n
}

func weightedScore(c *Candidate, dims []Dimension) float64 {
	if c == nil || c.Score == nil {
		return -1e18
	}
	total := 0.0
	for _, d := range dims {
		v := c.Score.Soft[d.Name]
		w := d.Weight
		if w == 0 {
			w = 1
		}
		if !d.HigherIsBetter {
			v = -v
		}
		total += w * v
	}
	return total
}

// GlobalFrontier 在所有 lineage 跑完后,合并各自的 BestEver 算总前沿。
func GlobalFrontier(lins []*Lineage, dims []Dimension) []*Candidate {
	var out []*Candidate
	for _, l := range lins {
		l.mu.Lock()
		for _, c := range l.BestEver {
			out = updateFrontier(out, c, dims)
		}
		l.mu.Unlock()
	}
	return out
}

// ============================================================
// Plateau:停滞检测,呼应"基线最优时放弃改进"
// ============================================================

type PlateauPolicy interface {
	ShouldStop(l *Lineage, currentGen int) bool
}

// StallAfter 是最常用策略:连续 N 代没有 BestEver 扩张就停。
//
// 推荐:L<=4 设 N=2;L<=8 设 N=3;L>=10 设 N=4。
type StallAfter struct{ N int }

func (p StallAfter) ShouldStop(l *Lineage, currentGen int) bool {
	if p.N <= 0 {
		return false
	}
	return currentGen-l.LastImprovementGen() >= p.N
}

// ============================================================
// Trajectory:最小化轨迹记录
// ============================================================
//
// 完整的 IRFT 风格轨迹后处理 dopforge 不做(因为我们不训练模型);
// 这里只记录足够回顾搜索进程的元数据,用于事后 debug + 未来可能的 forge_train 项目。

type Trajectory struct {
	StartedAt time.Time   `json:"started_at"`
	EndedAt   time.Time   `json:"ended_at"`
	BudgetC   int         `json:"budget_c"`
	BudgetL   int         `json:"budget_l"`
	BudgetK   int         `json:"budget_k"`
	Picks     []TrajPick  `json:"picks"`     // 每个 lineage 每代选了谁
	Frontier  []TrajPoint `json:"frontier"`  // 最终全局前沿
}

type TrajPick struct {
	Lineage    int                `json:"lineage"`
	Generation int                `json:"generation"`
	PickedID   string             `json:"picked_id"`
	PickedSoft map[string]float64 `json:"picked_soft"`
	Notes      string             `json:"notes,omitempty"`
}

type TrajPoint struct {
	ID      string             `json:"id"`
	WorkDir string             `json:"work_dir"`
	Lineage int                `json:"lineage"`
	Soft    map[string]float64 `json:"soft"`
}

// Save 把 Trajectory 序列化到 <root>/trajectory.json。
func (t *Trajectory) Save(root string) error {
	if err := os.MkdirAll(root, 0o755); err != nil {
		return err
	}
	data, err := json.MarshalIndent(t, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(root, "trajectory.json"), data, 0o644)
}

// ============================================================
// 主循环:C × L × K
// ============================================================

// Mutator 把"父代 → 子代"这一步落地的回调。
//
// 实现方典型路径:harness.New → Index → AsLLMTools → Run。
// child.WorkDir 已就绪(parent 的 cp -a 副本)。
type Mutator func(ctx context.Context, child, parent *Candidate) error

// SearchConfig 是一次搜索的全部输入。
type SearchConfig struct {
	Goal      *Goal
	Workspace *Workspace

	Mutator  Mutator
	Pipeline *Pipeline

	C int // 并行 lineage 数,典型 4
	L int // 每条 lineage 代数,典型 6
	K int // 每代孩子数,典型 3

	// EvalParallelism: 同代内 K 个孩子的并发上限。默认 K(全并行)。
	// 设 1 可串行,便于调试。
	EvalParallelism int

	// CleanupLosers: 每代结束后是否删除被淘汰孩子的 WorkDir。默认 true。
	CleanupLosers bool

	// Plateau 决定 lineage 是否提前停止。空 = 从不提前停。
	// 推荐 StallAfter{N: 3}。
	Plateau PlateauPolicy

	// OnImprovement:任一 lineage 检测到 BestEver 扩张时调用。
	// Forge 用它挂 skill 蒸馏。
	OnImprovement func(c *Candidate)

	Logger func(event string, fields map[string]any)
}

type SearchResult struct {
	Frontier   []*Candidate
	Lineages   []*Lineage
	Trajectory *Trajectory
}

// RunSearch 启动 C×L×K 搜索。
func RunSearch(ctx context.Context, cfg SearchConfig) (*SearchResult, error) {
	if cfg.C <= 0 || cfg.L <= 0 || cfg.K <= 0 {
		return nil, fmt.Errorf("search: C/L/K must all be > 0 (got %d/%d/%d)", cfg.C, cfg.L, cfg.K)
	}
	if cfg.Mutator == nil || cfg.Pipeline == nil || cfg.Workspace == nil {
		return nil, fmt.Errorf("search: Mutator/Pipeline/Workspace are all required")
	}
	if cfg.EvalParallelism <= 0 {
		cfg.EvalParallelism = cfg.K
	}
	logf := cfg.Logger
	if logf == nil {
		logf = func(string, map[string]any) {}
	}

	dims := cfg.Goal.Dimensions

	// 初始化 lineages
	lineages := make([]*Lineage, cfg.C)
	for i := range lineages {
		lineages[i] = NewLineage(i)
		if cfg.OnImprovement != nil {
			lineages[i].OnImprovement = cfg.OnImprovement
		}
	}

	traj := &Trajectory{
		StartedAt: time.Now(),
		BudgetC:   cfg.C, BudgetL: cfg.L, BudgetK: cfg.K,
	}
	var trajMu sync.Mutex
	addPick := func(p TrajPick) {
		trajMu.Lock()
		traj.Picks = append(traj.Picks, p)
		trajMu.Unlock()
	}

	// gen=0:种子代
	logf("seed_phase_start", map[string]any{"c": cfg.C})
	for _, l := range lineages {
		seed := NewCandidate(l.ID, 0, nil)
		if err := cfg.Workspace.Materialize(seed); err != nil {
			return nil, fmt.Errorf("search: materialize seed L%d: %w", l.ID, err)
		}
		score, err := cfg.Pipeline.Evaluate(ctx, seed)
		if err != nil {
			return nil, fmt.Errorf("search: evaluate seed L%d: %w", l.ID, err)
		}
		seed.Score = score
		l.Append(seed, dims)
		addPick(TrajPick{
			Lineage: l.ID, Generation: 0, PickedID: seed.ID,
			PickedSoft: seed.Score.Soft, Notes: seed.Score.Notes,
		})
	}

	// gen=1..L:主循环
	for gen := 1; gen <= cfg.L; gen++ {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		alive := 0
		for _, l := range lineages {
			if !l.Stopped() {
				alive++
			}
		}
		if alive == 0 {
			logf("all_lineages_stalled", map[string]any{"gen": gen})
			break
		}
		logf("generation_start", map[string]any{"gen": gen, "alive": alive})

		var wg sync.WaitGroup
		for _, l := range lineages {
			if l.Stopped() {
				continue
			}
			wg.Add(1)
			go func(l *Lineage) {
				defer wg.Done()
				if pick := runOneStep(ctx, cfg, l, gen, dims, logf); pick != nil {
					addPick(*pick)
				}
			}(l)
		}
		wg.Wait()

		// 代后:Plateau 检测
		if cfg.Plateau != nil {
			for _, l := range lineages {
				if l.Stopped() {
					continue
				}
				if cfg.Plateau.ShouldStop(l, gen) {
					l.MarkStopped()
					logf("lineage_stalled", map[string]any{
						"lineage":          l.ID,
						"gen":              gen,
						"last_improvement": l.LastImprovementGen(),
					})
				}
			}
		}
		logf("generation_done", map[string]any{"gen": gen})
	}

	// 收尾
	frontier := GlobalFrontier(lineages, dims)
	for _, c := range frontier {
		traj.Frontier = append(traj.Frontier, TrajPoint{
			ID: c.ID, WorkDir: c.WorkDir, Lineage: c.Lineage, Soft: c.Score.Soft,
		})
	}
	traj.EndedAt = time.Now()

	logf("search_done", map[string]any{
		"frontier_size": len(frontier),
		"duration":      traj.EndedAt.Sub(traj.StartedAt).String(),
	})
	return &SearchResult{Frontier: frontier, Lineages: lineages, Trajectory: traj}, nil
}

// runOneStep:某条 lineage 的一次 L 步骤。
func runOneStep(
	ctx context.Context, cfg SearchConfig, l *Lineage, gen int,
	dims []Dimension, logf func(string, map[string]any),
) *TrajPick {
	parent := l.Latest()
	if parent == nil {
		return nil
	}

	children := make([]*Candidate, cfg.K)
	var inner sync.WaitGroup
	sem := make(chan struct{}, cfg.EvalParallelism)
	for k := 0; k < cfg.K; k++ {
		inner.Add(1)
		go func(k int) {
			defer inner.Done()
			sem <- struct{}{}
			defer func() { <-sem }()

			child := NewCandidate(l.ID, gen, parent)
			if err := cfg.Workspace.Materialize(child); err != nil {
				logf("materialize_err", map[string]any{
					"lineage": l.ID, "gen": gen, "k": k, "err": err.Error(),
				})
				return
			}
			if err := cfg.Mutator(ctx, child, parent); err != nil {
				logf("mutate_err", map[string]any{
					"lineage": l.ID, "gen": gen, "id": child.ID, "err": err.Error(),
				})
				// 即便 mutate 报错也仍然评估——可能从 Pipeline 拿到有用的 Notes
			}
			score, err := cfg.Pipeline.Evaluate(ctx, child)
			if err != nil {
				logf("evaluate_err", map[string]any{
					"lineage": l.ID, "gen": gen, "id": child.ID, "err": err.Error(),
				})
				return
			}
			child.Score = score
			children[k] = child
		}(k)
	}
	inner.Wait()

	live := make([]*Candidate, 0, cfg.K)
	for _, c := range children {
		if c != nil {
			live = append(live, c)
		}
	}
	if len(live) == 0 {
		logf("step_all_failed", map[string]any{"lineage": l.ID, "gen": gen})
		return nil
	}

	winner := PickBestInBatch(live, dims)
	if winner == nil {
		return nil
	}
	l.Append(winner, dims)

	if cfg.CleanupLosers {
		for _, c := range live {
			if c != winner {
				_ = cfg.Workspace.Cleanup(c)
			}
		}
	}

	logf("step_done", map[string]any{
		"lineage": l.ID, "gen": gen, "picked": winner.ID, "children": len(live),
	})
	return &TrajPick{
		Lineage: l.ID, Generation: gen, PickedID: winner.ID,
		PickedSoft: winner.Score.Soft, Notes: winner.Score.Notes,
	}
}
