package optflow

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// PointF is a sub-pixel 2-D point with float64 coordinates (column X, row Y). It
// is the floating-point analogue of image.Point used by the sub-pixel sparse
// trackers and scattered-data interpolators in this package.
type PointF struct {
	// X is the horizontal (column) coordinate.
	X float64
	// Y is the vertical (row) coordinate.
	Y float64
}

// requireSamples validates the parallel point/vector inputs to the interpolators.
func requireSamples(rows, cols int, points []PointF, vectors []PointF, fn string) {
	if rows <= 0 || cols <= 0 {
		panic("optflow: " + fn + " requires positive dimensions")
	}
	if len(points) != len(vectors) {
		panic("optflow: " + fn + " requires len(points) == len(vectors)")
	}
}

// InterpolateFlow densifies a set of sparse flow samples into a full FlowField
// of size rows×cols using purely geometric (Gaussian / Shepard) scattered-data
// interpolation. Each output pixel is the weighted average of the sample
// vectors, where a sample at point p contributes with weight
//
//	exp(−|q − p|² / (2·sigma²))   (sigma > 0)
//
// or, when sigma ≤ 0, an inverse-square-distance (Shepard) weight 1/|q − p|²
// that interpolates the samples exactly at their own locations. points and
// vectors are parallel slices: vectors[i] is the (u, v) displacement measured at
// points[i]. This is the geometry-only counterpart of
// [CalcOpticalFlowSparseToDense]'s edge-aware splatting and of
// [InterpolateFlowGuided].
//
// With no samples the zero field is returned. len(points) must equal
// len(vectors) and the dimensions must be positive.
func InterpolateFlow(rows, cols int, points, vectors []PointF, sigma float64) *FlowField {
	requireSamples(rows, cols, points, vectors, "InterpolateFlow")
	flow := NewFlowField(rows, cols)
	if len(points) == 0 {
		return flow
	}
	gaussian := sigma > 0
	inv2s2 := 0.0
	if gaussian {
		inv2s2 = 1.0 / (2 * sigma * sigma)
	}
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			var su, sv, wsum float64
			exact := -1
			for k := range points {
				dx := float64(x) - points[k].X
				dy := float64(y) - points[k].Y
				d2 := dx*dx + dy*dy
				if d2 == 0 {
					exact = k
					break
				}
				var w float64
				if gaussian {
					w = math.Exp(-d2 * inv2s2)
				} else {
					w = 1.0 / d2
				}
				su += w * vectors[k].X
				sv += w * vectors[k].Y
				wsum += w
			}
			if exact >= 0 {
				flow.set(y, x, vectors[exact].X, vectors[exact].Y)
			} else if wsum > 0 {
				flow.set(y, x, su/wsum, sv/wsum)
			}
		}
	}
	return flow
}

// InterpolateFlowGuided densifies sparse flow samples with edge-aware weighting
// driven by a guide image. Each output pixel combines a spatial Gaussian on the
// image-plane distance to a sample (scale sigmaS) with a range Gaussian on the
// grayscale difference between the pixel and the sample location (scale sigmaC),
// so displacements are preferentially borrowed from samples lying on the same
// side of an intensity edge. This keeps motion boundaries crisp and is the
// densification step used by the dense RLOF pipeline (see
// [CalcOpticalFlowDenseRLOF]).
//
// guide supplies the intensity structure (multi-channel inputs are converted to
// grayscale) and defines the output size; points and vectors are parallel
// slices of sample locations and their (u, v) displacements. sigmaS and sigmaC
// must be > 0. With no samples the zero field is returned.
func InterpolateFlowGuided(guide *cv.Mat, points, vectors []PointF, sigmaS, sigmaC float64) *FlowField {
	if guide == nil || guide.Empty() {
		panic("optflow: InterpolateFlowGuided requires a non-empty guide image")
	}
	if sigmaS <= 0 || sigmaC <= 0 {
		panic("optflow: InterpolateFlowGuided requires sigmaS > 0 and sigmaC > 0")
	}
	g := grayGrid(guide)
	rows, cols := g.Rows, g.Cols
	if len(points) != len(vectors) {
		panic("optflow: InterpolateFlowGuided requires len(points) == len(vectors)")
	}
	flow := NewFlowField(rows, cols)
	if len(points) == 0 {
		return flow
	}
	// Pre-sample each seed's guide intensity once.
	seedVal := make([]float64, len(points))
	for k := range points {
		seedVal[k] = g.bilinear(points[k].X, points[k].Y)
	}
	invS2 := 1.0 / (2 * sigmaS * sigmaS)
	invC2 := 1.0 / (2 * sigmaC * sigmaC)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			pv := g.at(x, y)
			var su, sv, wsum float64
			for k := range points {
				dx := float64(x) - points[k].X
				dy := float64(y) - points[k].Y
				d2 := dx*dx + dy*dy
				dc := pv - seedVal[k]
				w := math.Exp(-d2*invS2 - dc*dc*invC2)
				su += w * vectors[k].X
				sv += w * vectors[k].Y
				wsum += w
			}
			if wsum > 0 {
				flow.set(y, x, su/wsum, sv/wsum)
			}
		}
	}
	return flow
}
