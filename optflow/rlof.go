package optflow

import (
	"image"
	"math"

	cv "github.com/malcolmston/opencv"
)

// RLOFParams configures the Robust Local Optical Flow trackers
// ([CalcOpticalFlowSparseRLOF] and [CalcOpticalFlowDenseRLOF]). The defaults
// from [DefaultRLOFParams] enable the robustness and illumination features that
// distinguish RLOF from plain Lucas-Kanade.
type RLOFParams struct {
	// WinRadius is the half-size of the tracking window; the window is
	// (2·WinRadius+1)² pixels.
	WinRadius int
	// MaxIters is the maximum number of iteratively-reweighted least-squares
	// updates per point.
	MaxIters int
	// MinEigen is the smallest acceptable eigenvalue of the (normalised)
	// structure tensor; below it a point is declared untrackable.
	MinEigen float64
	// HuberDelta is the residual scale of the Huber M-estimator. Residuals
	// larger than HuberDelta (in intensity units) are down-weighted linearly
	// instead of quadratically, giving robustness to outliers and occlusions.
	HuberDelta float64
	// Illumination enables a per-window additive illumination model (the mean
	// residual is removed each iteration), making the tracker invariant to
	// uniform brightness changes between frames.
	Illumination bool
	// GridStep is the seed spacing (in pixels) used by the dense variant before
	// edge-aware interpolation.
	GridStep int
}

// DefaultRLOFParams returns a robust RLOF configuration
// (WinRadius=6, MaxIters=20, MinEigen=1e-3, HuberDelta=12, Illumination=true,
// GridStep=6).
func DefaultRLOFParams() RLOFParams {
	return RLOFParams{
		WinRadius:    6,
		MaxIters:     20,
		MinEigen:     1e-3,
		HuberDelta:   12.0,
		Illumination: true,
		GridStep:     6,
	}
}

func (p RLOFParams) validate() {
	if p.WinRadius < 1 {
		panic("optflow: RLOF requires WinRadius >= 1")
	}
	if p.MaxIters < 1 {
		panic("optflow: RLOF requires MaxIters >= 1")
	}
	if p.HuberDelta <= 0 {
		panic("optflow: RLOF requires HuberDelta > 0")
	}
	if p.GridStep < 1 {
		panic("optflow: RLOF requires GridStep >= 1")
	}
}

// CalcOpticalFlowSparseRLOF tracks a set of points from prev to next with Robust
// Local Optical Flow (Senst et al.), the illumination-robust successor to sparse
// Lucas-Kanade used by OpenCV's cv::optflow::calcOpticalFlowSparseRLOF.
//
// For each input point the displacement is estimated by iteratively-reweighted
// least squares over the tracking window: every iteration warps next by the
// current estimate, forms the intensity residual, optionally removes its mean
// (the additive illumination model when Illumination is set), assigns each pixel
// a Huber weight that shrinks the influence of large residuals (occlusions,
// specularities, noise) and solves the weighted normal equations for a flow
// increment. Points whose weighted structure tensor is ill-conditioned are
// marked untracked.
//
// It returns nextPts, the sub-pixel destination of each input point (prevPts[i]
// + estimated flow), and status, true where tracking succeeded. On failure
// nextPts[i] echoes the input location. prev and next must be non-empty and
// identically sized; multi-channel inputs are converted to grayscale. p is
// validated (see [RLOFParams]). The result is deterministic.
func CalcOpticalFlowSparseRLOF(prev, next *cv.Mat, prevPts []image.Point, p RLOFParams) (nextPts []PointF, status []bool) {
	requirePair(prev, next, "CalcOpticalFlowSparseRLOF")
	p.validate()
	pg := grayGrid(prev)
	ng := grayGrid(next)
	gx, gy := sobelGradients(pg)

	nextPts = make([]PointF, len(prevPts))
	status = make([]bool, len(prevPts))
	for i, pt := range prevPts {
		if pt.X < 0 || pt.X >= pg.Cols || pt.Y < 0 || pt.Y >= pg.Rows {
			nextPts[i] = PointF{X: float64(pt.X), Y: float64(pt.Y)}
			continue
		}
		u, v, ok := robustLKPoint(pg, ng, gx, gy, pt.X, pt.Y, p)
		if ok {
			nextPts[i] = PointF{X: float64(pt.X) + u, Y: float64(pt.Y) + v}
			status[i] = true
		} else {
			nextPts[i] = PointF{X: float64(pt.X), Y: float64(pt.Y)}
		}
	}
	return nextPts, status
}

// CalcOpticalFlowDenseRLOF computes a dense flow field from prev to next with the
// dense RLOF pipeline of OpenCV's cv::optflow::calcOpticalFlowDenseRLOF: robust
// sparse RLOF on a regular grid of seeds followed by edge-aware interpolation to
// every pixel.
//
// A grid of seeds (spacing GridStep) is tracked with [CalcOpticalFlowSparseRLOF];
// the surviving robust matches are then densified with [InterpolateFlowGuided]
// using the prev frame as the edge guide, so the dense field respects motion
// boundaries. This combines RLOF's per-seed robustness with a globally smooth,
// boundary-aware result.
//
// prev and next must be non-empty and identically sized; multi-channel inputs
// are converted to grayscale. p is validated (see [RLOFParams]). The result is
// deterministic.
func CalcOpticalFlowDenseRLOF(prev, next *cv.Mat, p RLOFParams) *FlowField {
	requirePair(prev, next, "CalcOpticalFlowDenseRLOF")
	p.validate()
	rows, cols := prev.Rows, prev.Cols

	seeds := gridSeeds(rows, cols, p.GridStep)
	nextPts, status := CalcOpticalFlowSparseRLOF(prev, next, seeds, p)

	points := make([]PointF, 0, len(seeds))
	vectors := make([]PointF, 0, len(seeds))
	for i, ok := range status {
		if !ok {
			continue
		}
		points = append(points, PointF{X: float64(seeds[i].X), Y: float64(seeds[i].Y)})
		vectors = append(vectors, PointF{
			X: nextPts[i].X - float64(seeds[i].X),
			Y: nextPts[i].Y - float64(seeds[i].Y),
		})
	}
	if len(points) == 0 {
		return NewFlowField(rows, cols)
	}
	sigmaS := math.Max(float64(p.GridStep)*1.5, 4)
	return InterpolateFlowGuided(prev, points, vectors, sigmaS, 20.0)
}

// robustLKPoint estimates the displacement of point (px, py) by iteratively-
// reweighted Lucas-Kanade with Huber weights and an optional additive
// illumination model. It returns ok=false when the weighted structure tensor is
// ill-conditioned.
func robustLKPoint(prev, next, gx, gy *grid, px, py int, p RLOFParams) (u, v float64, ok bool) {
	r := p.WinRadius
	winArea := float64((2*r + 1) * (2*r + 1))
	delta := p.HuberDelta

	for iter := 0; iter < p.MaxIters; iter++ {
		// First pass: residuals and (optionally) their mean for the additive
		// illumination model.
		var meanRes float64
		if p.Illumination {
			for wy := -r; wy <= r; wy++ {
				for wx := -r; wx <= r; wx++ {
					sx, sy := px+wx, py+wy
					meanRes += next.bilinear(float64(sx)+u, float64(sy)+v) - prev.atClamp(sx, sy)
				}
			}
			meanRes /= winArea
		}

		// Weighted normal equations.
		var a11, a12, a22, b1, b2 float64
		for wy := -r; wy <= r; wy++ {
			for wx := -r; wx <= r; wx++ {
				sx, sy := px+wx, py+wy
				res := next.bilinear(float64(sx)+u, float64(sy)+v) - prev.atClamp(sx, sy) - meanRes
				// Huber weight: 1 for small residuals, delta/|res| beyond.
				w := 1.0
				if ar := math.Abs(res); ar > delta {
					w = delta / ar
				}
				gxv := gx.atClamp(sx, sy)
				gyv := gy.atClamp(sx, sy)
				a11 += w * gxv * gxv
				a12 += w * gxv * gyv
				a22 += w * gyv * gyv
				b1 += w * gxv * res
				b2 += w * gyv * res
			}
		}

		det := a11*a22 - a12*a12
		tr := a11 + a22
		disc := math.Sqrt(math.Max(tr*tr-4*det, 0))
		minEig := (tr - disc) / 2 / winArea
		if det < 1e-9 || minEig < p.MinEigen {
			if iter == 0 {
				return 0, 0, false
			}
			return u, v, true
		}
		// Solve A·Δ = -b (residual = next - prev, so we descend it).
		du, dv := solve2x2(a11, a12, a22, -b1, -b2)
		u += du
		v += dv
		if du*du+dv*dv < 1e-6 {
			break
		}
	}
	return u, v, true
}
