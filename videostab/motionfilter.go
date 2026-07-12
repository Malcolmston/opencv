package videostab

import (
	"math"
)

// Range is an inclusive index range [First, Last] over the frame sequence. It
// bounds the neighbourhood a motion filter may look at when the full sequence is
// not yet available (as in one-pass stabilization).
type Range struct {
	First int
	Last  int
}

// clampInt clamps v to [lo, hi].
func clampInt(v, lo, hi int) int {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

// MotionStabilizer converts a sequence of inter-frame motions into a sequence of
// per-frame stabilization warps. It mirrors cv::videostab::IMotionStabilizer.
type MotionStabilizer interface {
	// Stabilize fills stabilizationMotions[0..size-1] with the warp to apply to
	// each frame. motions[i] is the motion between frame i and frame i+1 and
	// rng bounds the frames that are available.
	Stabilize(size int, motions []Motion, rng Range, stabilizationMotions []Motion)
}

// MotionFilterBase is a per-frame motion filter: it can produce the
// stabilization warp for a single frame in addition to the batch
// [MotionStabilizer] interface. It mirrors cv::videostab::MotionFilterBase.
type MotionFilterBase interface {
	MotionStabilizer
	// StabilizeAt returns the stabilization warp for a single frame index.
	StabilizeAt(idx int, motions []Motion, rng Range) Motion
}

// GaussianMotionFilter smooths the camera trajectory by replacing the transform
// at each frame with a Gaussian-weighted average of the relative motions to its
// temporal neighbours, exactly as cv::videostab::GaussianMotionFilter does. The
// weight of neighbour j relative to frame i is exp(-(i-j)²/(2·stdev²)).
type GaussianMotionFilter struct {
	radius int
	stdev  float64
}

// NewGaussianMotionFilter creates a Gaussian filter with the given trailing and
// leading radius (in frames). When stdev is not positive it defaults to
// radius/√(-2·ln(1/256)), matching OpenCV, so the weight decays to ~1/256 at the
// window edge. radius must be positive.
func NewGaussianMotionFilter(radius int, stdev float64) *GaussianMotionFilter {
	if radius < 1 {
		panic("videostab: NewGaussianMotionFilter requires radius >= 1")
	}
	if stdev <= 0 {
		stdev = float64(radius) / math.Sqrt(-2*math.Log(1.0/256.0))
	}
	return &GaussianMotionFilter{radius: radius, stdev: stdev}
}

// Radius returns the filter radius.
func (f *GaussianMotionFilter) Radius() int { return f.radius }

// Stdev returns the Gaussian standard deviation used for weighting.
func (f *GaussianMotionFilter) Stdev() float64 { return f.stdev }

// StabilizeAt returns the Gaussian-weighted average transform for frame idx.
func (f *GaussianMotionFilter) StabilizeAt(idx int, motions []Motion, rng Range) Motion {
	var res Motion
	var sum float64
	lo := clampInt(idx-f.radius, rng.First, rng.Last)
	hi := clampInt(idx+f.radius, rng.First, rng.Last)
	for i := lo; i <= hi; i++ {
		d := float64(idx - i)
		w := math.Exp(-d * d / (2 * f.stdev * f.stdev))
		res = res.add(GetMotion(idx, i, motions).scaled(w))
		sum += w
	}
	if sum == 0 {
		return IdentityMotion()
	}
	return res.scaled(1 / sum)
}

// Stabilize fills stabilizationMotions with the per-frame Gaussian-smoothed
// transforms.
func (f *GaussianMotionFilter) Stabilize(size int, motions []Motion, rng Range, out []Motion) {
	for i := 0; i < size; i++ {
		out[i] = f.StabilizeAt(i, motions, rng)
	}
}

// MotionStabilizationPipeline chains several motion stabilizers, applying each
// in turn to the running stabilization estimate. It mirrors
// cv::videostab::MotionStabilizationPipeline.
type MotionStabilizationPipeline struct {
	stabilizers []MotionStabilizer
}

// NewMotionStabilizationPipeline creates an empty pipeline.
func NewMotionStabilizationPipeline() *MotionStabilizationPipeline {
	return &MotionStabilizationPipeline{}
}

// Add appends a stabilizer to the pipeline and returns the pipeline for
// chaining.
func (p *MotionStabilizationPipeline) Add(s MotionStabilizer) *MotionStabilizationPipeline {
	p.stabilizers = append(p.stabilizers, s)
	return p
}

// Len reports how many stabilizers the pipeline contains.
func (p *MotionStabilizationPipeline) Len() int { return len(p.stabilizers) }

// Stabilize runs every stabilizer in sequence. The first stabilizer sees the raw
// motions; each subsequent stabilizer refines the accumulated stabilization
// motions produced so far by composing its own correction on top.
func (p *MotionStabilizationPipeline) Stabilize(size int, motions []Motion, rng Range, out []Motion) {
	for i := 0; i < size; i++ {
		out[i] = IdentityMotion()
	}
	for _, s := range p.stabilizers {
		step := make([]Motion, size)
		s.Stabilize(size, motions, rng, step)
		for i := 0; i < size; i++ {
			out[i] = step[i].Mul(out[i])
		}
	}
}

// LpMotionStabilizer computes stabilization warps by fitting a smooth camera
// path to the accumulated trajectory. It minimises a weighted sum of the L1
// norms of the first, second and third temporal derivatives of the smoothed
// path (subject to staying close to the original path), which favours a path
// that is piecewise constant / linear / parabolic — the objective used by
// cv::videostab::LpMotionStabilizer. The L1 problem is solved with iteratively
// reweighted least squares rather than a linear-programming solver.
type LpMotionStabilizer struct {
	// W1, W2, W3 weight the first-, second- and third-derivative penalties.
	w1, w2, w3 float64
	iters      int
}

// NewLpMotionStabilizer returns an LP-style stabilizer with OpenCV's default
// derivative weights (1, 10, 100).
func NewLpMotionStabilizer() *LpMotionStabilizer {
	return &LpMotionStabilizer{w1: 1, w2: 10, w3: 100, iters: 20}
}

// SetWeights overrides the first/second/third derivative penalty weights.
func (s *LpMotionStabilizer) SetWeights(w1, w2, w3 float64) {
	s.w1, s.w2, s.w3 = w1, w2, w3
}

// Stabilize smooths each element of the accumulated affine trajectory and emits
// the correction warp for every frame.
func (s *LpMotionStabilizer) Stabilize(size int, motions []Motion, rng Range, out []Motion) {
	if size == 0 {
		return
	}
	// Accumulate the trajectory C_i = GetMotion(0, i) mapping frame 0 into
	// frame i, then smooth each of the six affine parameters independently.
	cum := make([]Motion, size)
	for i := 0; i < size; i++ {
		cum[i] = GetMotion(0, i, motions)
	}
	const nParams = 6
	series := make([][]float64, nParams)
	for p := 0; p < nParams; p++ {
		series[p] = make([]float64, size)
		for i := 0; i < size; i++ {
			series[p][i] = cum[i][paramIndex(p)]
		}
	}
	smoothed := make([][]float64, nParams)
	for p := 0; p < nParams; p++ {
		smoothed[p] = s.smoothSeries(series[p])
	}
	for i := 0; i < size; i++ {
		sm := Motion{
			smoothed[0][i], smoothed[1][i], smoothed[2][i],
			smoothed[3][i], smoothed[4][i], smoothed[5][i],
			0, 0, 1,
		}
		if inv, ok := cum[i].Inverse(); ok {
			out[i] = sm.Mul(inv)
		} else {
			out[i] = IdentityMotion()
		}
	}
}

// paramIndex maps a parameter number 0..5 to its Motion element index.
func paramIndex(p int) int {
	// Affine parameters live in the first two rows: indices 0,1,2,3,4,5.
	return p
}

// smoothSeries minimises data.fidelity + w1·|Δ| + w2·|Δ²| + w3·|Δ³| of the
// sequence in the L1 sense via iteratively reweighted least squares.
func (s *LpMotionStabilizer) smoothSeries(x []float64) []float64 {
	n := len(x)
	if n <= 2 {
		out := make([]float64, n)
		copy(out, x)
		return out
	}
	y := make([]float64, n)
	copy(y, x)
	const eps = 1e-4
	for it := 0; it < s.iters; it++ {
		// Build the tridiagonal-plus normal system for the weighted least
		// squares step. Data term keeps y close to x; derivative terms penalise
		// roughness with IRLS weights.
		a := make([][]float64, n)
		for i := range a {
			a[i] = make([]float64, n)
		}
		b := make([]float64, n)
		for i := 0; i < n; i++ {
			a[i][i] += 1
			b[i] += x[i]
		}
		addDiff(a, y, 1, s.w1, eps) // first derivative
		addDiff(a, y, 2, s.w2, eps) // second derivative
		addDiff(a, y, 3, s.w3, eps) // third derivative
		sol, ok := solveLinear(a, b)
		if !ok {
			break
		}
		y = sol
	}
	return y
}

// addDiff adds the IRLS-weighted L1 penalty on the order-th finite difference of
// the current estimate y into the normal system (a, implicit b=0 contribution).
func addDiff(a [][]float64, y []float64, order int, weight, eps float64) {
	n := len(y)
	if weight <= 0 || n <= order {
		return
	}
	coeffs := diffCoeffs(order)
	for i := 0; i+order < n; i++ {
		// Residual of this difference row under the current estimate.
		var r float64
		for k, c := range coeffs {
			r += c * y[i+k]
		}
		w := weight / math.Max(math.Abs(r), eps)
		for k1, c1 := range coeffs {
			for k2, c2 := range coeffs {
				a[i+k1][i+k2] += w * c1 * c2
			}
		}
	}
}

// diffCoeffs returns the finite-difference stencil for the given derivative
// order (binomial coefficients with alternating sign).
func diffCoeffs(order int) []float64 {
	switch order {
	case 1:
		return []float64{-1, 1}
	case 2:
		return []float64{1, -2, 1}
	case 3:
		return []float64{-1, 3, -3, 1}
	default:
		return []float64{1}
	}
}
