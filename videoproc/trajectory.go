package videoproc

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// PointF is a floating-point image coordinate (X is the column, Y is the row).
// It is the sub-pixel analogue of cv.Point used by the tracking, stabilization
// and trajectory routines in this package.
type PointF struct {
	// X is the column (horizontal) coordinate.
	X float64
	// Y is the row (vertical) coordinate.
	Y float64
}

// FlowField is a dense two-channel optical-flow field: for each pixel, X holds
// the horizontal displacement and Y the vertical displacement that maps a pixel
// in the source frame to its position in the destination frame. The root cv
// package's FloatMat is single-channel, so this small container pairs two of
// them; it is the flow representation consumed by [WarpByFlow] and
// [InterpolateFlow].
type FlowField struct {
	// X is the per-pixel horizontal displacement.
	X *cv.FloatMat
	// Y is the per-pixel vertical displacement.
	Y *cv.FloatMat
}

// NewFlowField allocates a zero (no-motion) flow field of the given dimensions.
// It panics on non-positive dimensions.
func NewFlowField(rows, cols int) *FlowField {
	if rows <= 0 || cols <= 0 {
		panic("videoproc: NewFlowField requires positive dimensions")
	}
	return &FlowField{X: cv.NewFloatMat(rows, cols), Y: cv.NewFloatMat(rows, cols)}
}

// At returns the (dx, dy) displacement stored at pixel (x, y). It panics if the
// coordinates are out of range.
func (f *FlowField) At(x, y int) (dx, dy float64) {
	return f.X.At(y, x), f.Y.At(y, x)
}

// Set stores the displacement (dx, dy) at pixel (x, y). It panics if the
// coordinates are out of range.
func (f *FlowField) Set(x, y int, dx, dy float64) {
	if y < 0 || y >= f.X.Rows || x < 0 || x >= f.X.Cols {
		panic("videoproc: FlowField.Set out of range")
	}
	f.X.Data[y*f.X.Cols+x] = dx
	f.Y.Data[y*f.Y.Cols+x] = dy
}

// SampleDenseGrid returns integer sampling locations on a regular grid covering
// an image of the given size, one point every step pixels in both directions,
// starting at (step/2, step/2). It is the seeding step of dense-trajectory
// extraction and of grid-based feature tracking. It panics on non-positive
// dimensions or step.
func SampleDenseGrid(rows, cols, step int) []cv.Point {
	if rows <= 0 || cols <= 0 {
		panic("videoproc: SampleDenseGrid requires positive dimensions")
	}
	if step <= 0 {
		panic("videoproc: SampleDenseGrid requires step > 0")
	}
	var pts []cv.Point
	for y := step / 2; y < rows; y += step {
		for x := step / 2; x < cols; x += step {
			pts = append(pts, cv.Point{X: x, Y: y})
		}
	}
	return pts
}

// TrackPoints tracks a set of points from prev to cur by block matching: around
// each point it searches an integer displacement within ±searchRadius that
// minimises the sum of absolute differences of a (2*patchRadius+1)² grayscale
// patch. It returns the tracked positions and a parallel validity slice; a point
// is invalid (and its output position equals its input) when the best match cost
// exceeds maxCost times the patch area (a per-pixel average SAD bound) or the
// point starts out of bounds. This is a simple, deterministic, feature-agnostic
// tracker suitable for stabilization and dense trajectories. It panics on
// non-positive radii or a frame size mismatch.
func TrackPoints(prev, cur *cv.Mat, pts []PointF, searchRadius, patchRadius int, maxCost float64) (tracked []PointF, valid []bool) {
	if prev == nil || cur == nil || prev.Empty() || cur.Empty() {
		panic("videoproc: TrackPoints requires two non-empty frames")
	}
	if prev.Rows != cur.Rows || prev.Cols != cur.Cols {
		panic("videoproc: TrackPoints frame size mismatch")
	}
	if searchRadius <= 0 || patchRadius <= 0 {
		panic("videoproc: TrackPoints requires positive searchRadius and patchRadius")
	}
	gp := videoprocToGray(prev)
	gc := videoprocToGray(cur)
	tracked = make([]PointF, len(pts))
	valid = make([]bool, len(pts))
	area := (2*patchRadius + 1) * (2*patchRadius + 1)
	costLimit := maxCost * float64(area)
	for i, p := range pts {
		px := int(math.Round(p.X))
		py := int(math.Round(p.Y))
		bestCost := math.Inf(1)
		bestDX, bestDY := 0, 0
		for dy := -searchRadius; dy <= searchRadius; dy++ {
			for dx := -searchRadius; dx <= searchRadius; dx++ {
				cost := videoprocPatchSAD(gp, gc, px, py, px+dx, py+dy, patchRadius, bestCost)
				if cost < bestCost {
					bestCost = cost
					bestDX, bestDY = dx, dy
				}
			}
		}
		if bestCost <= costLimit {
			tracked[i] = PointF{X: float64(px + bestDX), Y: float64(py + bestDY)}
			valid[i] = true
		} else {
			tracked[i] = p
			valid[i] = false
		}
	}
	return tracked, valid
}

// videoprocPatchSAD returns the sum of absolute differences between the patch of
// radius r centred at (ax, ay) in a and the patch centred at (bx, by) in b, with
// edge clamping. It early-outs once the running cost reaches limit.
func videoprocPatchSAD(a, b *cv.Mat, ax, ay, bx, by, r int, limit float64) float64 {
	var sum float64
	for oy := -r; oy <= r; oy++ {
		for ox := -r; ox <= r; ox++ {
			av := videoprocGrayAtClamp(a, ax+ox, ay+oy)
			bv := videoprocGrayAtClamp(b, bx+ox, by+oy)
			d := av - bv
			if d < 0 {
				d = -d
			}
			sum += d
		}
		if sum >= limit {
			return sum
		}
	}
	return sum
}

// Trajectory is the ordered path of a single tracked point across consecutive
// frames, the core datum of dense-trajectory descriptors. Points[0] is the seed
// location and each subsequent entry is the point's position in the next frame.
type Trajectory struct {
	// Points is the ordered list of positions, one per frame observed.
	Points []PointF
}

// Length returns the number of positions in the trajectory.
func (t *Trajectory) Length() int {
	return len(t.Points)
}

// Displacement returns the straight-line vector from the trajectory's first to
// its last point as (dx, dy). It returns (0,0) for a trajectory of fewer than
// two points.
func (t *Trajectory) Displacement() (dx, dy float64) {
	if len(t.Points) < 2 {
		return 0, 0
	}
	a := t.Points[0]
	b := t.Points[len(t.Points)-1]
	return b.X - a.X, b.Y - a.Y
}

// TotalLength returns the sum of the Euclidean lengths of the trajectory's
// segments (the arc length of the path). It returns 0 for fewer than two points.
func (t *Trajectory) TotalLength() float64 {
	var sum float64
	for i := 1; i < len(t.Points); i++ {
		sum += math.Hypot(t.Points[i].X-t.Points[i-1].X, t.Points[i].Y-t.Points[i-1].Y)
	}
	return sum
}

// DenseTrajectorySampler extracts dense trajectories by seeding points on a grid
// and tracking each across successive frames with [TrackPoints], mirroring the
// sampling stage of Wang et al.'s dense-trajectory features. Points are tracked
// until they are lost (poor match) or the trajectory reaches MaxLength, at which
// point the completed trajectory is emitted and the grid is re-seeded on the
// next frame.
type DenseTrajectorySampler struct {
	// Step is the grid spacing (pixels) used to seed new trajectories.
	Step int
	// MaxLength caps the number of points in a trajectory before it is emitted.
	MaxLength int
	// SearchRadius is the block-matching search radius passed to TrackPoints.
	SearchRadius int
	// PatchRadius is the block-matching patch radius passed to TrackPoints.
	PatchRadius int
	// MaxCost is the per-pixel average SAD bound passed to TrackPoints.
	MaxCost float64

	prev   *cv.Mat
	active []*Trajectory
}

// NewDenseTrajectorySampler returns a sampler with reasonable defaults for the
// given grid step and maximum trajectory length: search radius 4, patch radius
// 2 and a per-pixel SAD bound of 20. It panics if step or maxLength is
// non-positive.
func NewDenseTrajectorySampler(step, maxLength int) *DenseTrajectorySampler {
	if step <= 0 || maxLength <= 0 {
		panic("videoproc: NewDenseTrajectorySampler requires positive step and maxLength")
	}
	return &DenseTrajectorySampler{
		Step:         step,
		MaxLength:    maxLength,
		SearchRadius: 4,
		PatchRadius:  2,
		MaxCost:      20,
	}
}

// Feed advances every active trajectory to the given frame and returns the
// trajectories that completed on this frame (reached MaxLength). Trajectories
// whose point was lost are dropped. The first call seeds the grid and returns
// nil. Frame dimensions must stay constant across calls.
func (s *DenseTrajectorySampler) Feed(frame *cv.Mat) []*Trajectory {
	if frame == nil || frame.Empty() {
		panic("videoproc: DenseTrajectorySampler.Feed requires a non-empty frame")
	}
	if s.prev == nil {
		s.seed(frame)
		s.prev = frame.Clone()
		return nil
	}
	// track the head of every active trajectory into the new frame.
	heads := make([]PointF, len(s.active))
	for i, tr := range s.active {
		heads[i] = tr.Points[len(tr.Points)-1]
	}
	tracked, valid := TrackPoints(s.prev, frame, heads, s.SearchRadius, s.PatchRadius, s.MaxCost)
	var completed []*Trajectory
	var stillActive []*Trajectory
	for i, tr := range s.active {
		if !valid[i] {
			continue // trajectory lost
		}
		tr.Points = append(tr.Points, tracked[i])
		if len(tr.Points) >= s.MaxLength {
			completed = append(completed, tr)
		} else {
			stillActive = append(stillActive, tr)
		}
	}
	s.active = stillActive
	if len(s.active) == 0 {
		s.seed(frame)
	}
	s.prev = frame.Clone()
	return completed
}

// seed populates the active set with a fresh grid of length-1 trajectories.
func (s *DenseTrajectorySampler) seed(frame *cv.Mat) {
	grid := SampleDenseGrid(frame.Rows, frame.Cols, s.Step)
	s.active = make([]*Trajectory, len(grid))
	for i, g := range grid {
		s.active[i] = &Trajectory{Points: []PointF{{X: float64(g.X), Y: float64(g.Y)}}}
	}
}

// Active returns the trajectories currently being tracked (not yet completed or
// lost). The returned slice must not be modified.
func (s *DenseTrajectorySampler) Active() []*Trajectory {
	return s.active
}
