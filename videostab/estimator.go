package videostab

import (
	"math"
	"math/rand"

	"github.com/malcolmston/opencv/video"
)

// RansacParams configures the RANSAC loop used by [MotionEstimatorRansacL2]. It
// mirrors cv::videostab::RansacParams.
type RansacParams struct {
	// Size is the number of correspondences drawn for each minimal hypothesis.
	Size int
	// Thresh is the maximum reprojection error (in pixels) for a correspondence
	// to be counted as an inlier.
	Thresh float64
	// Eps is the assumed maximum proportion of outliers among the data.
	Eps float64
	// Prob is the desired probability that at least one all-inlier sample is
	// drawn.
	Prob float64
}

// DefaultRansacParams returns sensible RANSAC parameters for the given motion
// model, matching cv::videostab::RansacParams::default2dMotion: the minimal
// sample size is the model's degrees-of-freedom sample count and the error,
// outlier-ratio and success-probability defaults are 0.5, 0.5 and 0.99.
func DefaultRansacParams(model MotionModel) RansacParams {
	return RansacParams{Size: model.minPoints(), Thresh: 0.5, Eps: 0.5, Prob: 0.99}
}

// NumIters returns the number of RANSAC iterations required to reach the
// configured success probability given the outlier ratio and sample size,
// following the standard log formula ceil(log(1-prob)/log(1-(1-eps)^size)).
func (p RansacParams) NumIters() int {
	num := math.Log(1 - p.Prob)
	den := math.Log(1 - math.Pow(1-p.Eps, float64(p.Size)))
	if den == 0 || math.IsInf(den, 0) {
		return 1
	}
	n := int(math.Ceil(num / den))
	if n < 1 {
		return 1
	}
	return n
}

// MotionEstimatorBase estimates a global 2-D motion from a set of point
// correspondences. It is the sparse counterpart of [ImageMotionEstimator] and
// mirrors cv::videostab::MotionEstimatorBase.
type MotionEstimatorBase interface {
	// Estimate fits the configured motion model to the correspondences
	// from→to. The boolean result reports whether a reliable estimate was
	// obtained; on failure the identity transform is returned.
	Estimate(from, to []video.PointF) (Motion, bool)
	// MotionModel returns the model this estimator fits.
	MotionModel() MotionModel
	// SetMotionModel changes the model this estimator fits.
	SetMotionModel(m MotionModel)
}

// MotionEstimatorRansacL2 estimates global motion with a RANSAC hypothesis loop
// followed by an L2 (least-squares) refinement over the recovered inlier set. It
// is robust to a large fraction of mismatched correspondences and mirrors
// cv::videostab::MotionEstimatorRansacL2.
type MotionEstimatorRansacL2 struct {
	model    MotionModel
	ransac   RansacParams
	minInlie float64
	rng      *rand.Rand
}

// NewMotionEstimatorRansacL2 creates a RANSAC/L2 estimator for the given model
// with default RANSAC parameters and a fixed random seed (so results are
// deterministic). The minimum inlier ratio required to accept an estimate
// defaults to 0.1.
func NewMotionEstimatorRansacL2(model MotionModel) *MotionEstimatorRansacL2 {
	return &MotionEstimatorRansacL2{
		model:    model,
		ransac:   DefaultRansacParams(model),
		minInlie: 0.1,
		rng:      rand.New(rand.NewSource(1)),
	}
}

// MotionModel returns the fitted model.
func (e *MotionEstimatorRansacL2) MotionModel() MotionModel { return e.model }

// SetMotionModel changes the fitted model and resets the RANSAC parameters to
// the model's defaults.
func (e *MotionEstimatorRansacL2) SetMotionModel(m MotionModel) {
	e.model = m
	e.ransac = DefaultRansacParams(m)
}

// SetRansacParams overrides the RANSAC configuration.
func (e *MotionEstimatorRansacL2) SetRansacParams(p RansacParams) { e.ransac = p }

// RansacParams returns the current RANSAC configuration.
func (e *MotionEstimatorRansacL2) RansacParams() RansacParams { return e.ransac }

// SetMinInlierRatio sets the minimum fraction of inliers required to accept an
// estimate.
func (e *MotionEstimatorRansacL2) SetMinInlierRatio(r float64) { e.minInlie = r }

// SetSeed reseeds the internal random number generator, allowing reproducible
// but configurable RANSAC sampling.
func (e *MotionEstimatorRansacL2) SetSeed(seed int64) {
	e.rng = rand.New(rand.NewSource(seed))
}

// Estimate fits the model with RANSAC and refines it over the inliers.
func (e *MotionEstimatorRansacL2) Estimate(from, to []video.PointF) (Motion, bool) {
	n := len(from)
	if n != len(to) || n < e.model.minPoints() {
		return IdentityMotion(), false
	}
	best, inliers, ok := estimateRansac(from, to, e.model, e.ransac, e.rng)
	if !ok {
		return IdentityMotion(), false
	}
	// Collect inliers and refine with an ordinary least-squares fit.
	var fi, ti []video.PointF
	for i, in := range inliers {
		if in {
			fi = append(fi, from[i])
			ti = append(ti, to[i])
		}
	}
	if float64(len(fi))/float64(n) < e.minInlie {
		return IdentityMotion(), false
	}
	if refined, rok := fitMotion(fi, ti, e.model, nil); rok {
		best = refined
	}
	return best, true
}

// estimateRansac runs the RANSAC hypothesis-and-verify loop, returning the best
// transform, the corresponding inlier mask and whether any hypothesis produced
// enough inliers.
func estimateRansac(from, to []video.PointF, model MotionModel, params RansacParams, rng *rand.Rand) (Motion, []bool, bool) {
	n := len(from)
	size := params.Size
	if size < model.minPoints() {
		size = model.minPoints()
	}
	iters := params.NumIters()
	if iters > 2000 {
		iters = 2000
	}
	threshSq := params.Thresh * params.Thresh

	var bestModel Motion
	bestCount := -1
	var bestMask []bool

	idx := make([]int, size)
	subFrom := make([]video.PointF, size)
	subTo := make([]video.PointF, size)

	for it := 0; it < iters; it++ {
		sampleIndices(rng, n, idx)
		for k, j := range idx {
			subFrom[k] = from[j]
			subTo[k] = to[j]
		}
		m, ok := fitMotion(subFrom, subTo, model, nil)
		if !ok {
			continue
		}
		count := 0
		mask := make([]bool, n)
		for i := 0; i < n; i++ {
			x, y := m.Apply(from[i].X, from[i].Y)
			dx, dy := x-to[i].X, y-to[i].Y
			if dx*dx+dy*dy <= threshSq {
				mask[i] = true
				count++
			}
		}
		if count > bestCount {
			bestCount = count
			bestModel = m
			bestMask = mask
		}
	}
	if bestCount < model.minPoints() {
		return IdentityMotion(), nil, false
	}
	return bestModel, bestMask, true
}

// sampleIndices fills dst with len(dst) distinct random indices in [0, n).
func sampleIndices(rng *rand.Rand, n int, dst []int) {
	for k := range dst {
		for {
			c := rng.Intn(n)
			dup := false
			for j := 0; j < k; j++ {
				if dst[j] == c {
					dup = true
					break
				}
			}
			if !dup {
				dst[k] = c
				break
			}
		}
	}
}

// MotionEstimatorL1 estimates global motion by minimising the sum of absolute
// reprojection errors (an L1 fit). The L1 problem is solved with iteratively
// reweighted least squares, which down-weights large residuals and therefore
// tolerates outliers without an explicit RANSAC loop. It mirrors
// cv::videostab::MotionEstimatorL1.
type MotionEstimatorL1 struct {
	model MotionModel
	iters int
}

// NewMotionEstimatorL1 creates an L1 estimator for the given model.
func NewMotionEstimatorL1(model MotionModel) *MotionEstimatorL1 {
	return &MotionEstimatorL1{model: model, iters: 12}
}

// MotionModel returns the fitted model.
func (e *MotionEstimatorL1) MotionModel() MotionModel { return e.model }

// SetMotionModel changes the fitted model.
func (e *MotionEstimatorL1) SetMotionModel(m MotionModel) { e.model = m }

// Estimate fits the model in the L1 sense via iteratively reweighted least
// squares.
func (e *MotionEstimatorL1) Estimate(from, to []video.PointF) (Motion, bool) {
	n := len(from)
	if n != len(to) || n < e.model.minPoints() {
		return IdentityMotion(), false
	}
	m, ok := fitMotion(from, to, e.model, nil)
	if !ok {
		return IdentityMotion(), false
	}
	weights := make([]float64, n)
	const eps = 1e-3
	for it := 0; it < e.iters; it++ {
		for i := 0; i < n; i++ {
			r := residual(m, from[i], to[i])
			weights[i] = 1 / math.Max(r, eps)
		}
		next, nok := fitMotion(from, to, e.model, weights)
		if !nok {
			break
		}
		m = next
	}
	return m, true
}

// EstimateGlobalMotionLeastSquares fits the model to the correspondences with a
// single ordinary least-squares pass (no outlier rejection). It is exported for
// callers that already have clean correspondences.
func EstimateGlobalMotionLeastSquares(from, to []video.PointF, model MotionModel) (Motion, bool) {
	return fitMotion(from, to, model, nil)
}
