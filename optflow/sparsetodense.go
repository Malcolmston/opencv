package optflow

import (
	"image"
	"math"

	cv "github.com/malcolmston/opencv"
)

// CalcOpticalFlowSparseToDense estimates a dense flow field by first tracking a
// set of sparse seed points with local Lucas-Kanade and then interpolating those
// matches to every pixel with edge-aware weighting.
//
// For each seed in sparsePts the function runs an iterative Lucas-Kanade update
// (structure tensor assembled from Sobel gradients, refined with sub-pixel
// warping) to obtain a displacement. Seeds whose structure tensor is
// ill-conditioned (untextured neighbourhoods) are discarded. The surviving
// sparse matches are then splatted to a dense field: every output pixel is the
// weighted average of the seed displacements, where the weight of a seed
// combines a spatial Gaussian (distance in the image plane) with a range
// Gaussian on the grayscale difference between the pixel and the seed. The range
// term makes the interpolation edge-aware — flow is preferentially borrowed from
// seeds lying on the same side of an intensity edge, so motion boundaries stay
// crisp instead of being blurred across object contours.
//
// If sparsePts is empty a regularly spaced grid of seeds is generated
// automatically. prev and next must be non-empty and identically sized;
// multi-channel inputs are converted to grayscale. The result is deterministic.
func CalcOpticalFlowSparseToDense(prev, next *cv.Mat, sparsePts []image.Point) *FlowField {
	requirePair(prev, next, "CalcOpticalFlowSparseToDense")
	pg := grayGrid(prev)
	ng := grayGrid(next)
	rows, cols := pg.Rows, pg.Cols

	if len(sparsePts) == 0 {
		sparsePts = gridSeeds(rows, cols, 8)
	}

	gx, gy := sobelGradients(pg)

	const (
		winRadius = 4
		lkIters   = 8
	)
	type match struct {
		x, y int
		u, v float64
		val  float64 // seed grayscale, for the range term
	}
	matches := make([]match, 0, len(sparsePts))
	for _, pt := range sparsePts {
		if pt.X < 0 || pt.X >= cols || pt.Y < 0 || pt.Y >= rows {
			continue
		}
		u, v, ok := lucasKanadePoint(pg, ng, gx, gy, pt.X, pt.Y, winRadius, lkIters)
		if !ok {
			continue
		}
		matches = append(matches, match{x: pt.X, y: pt.Y, u: u, v: v, val: pg.at(pt.X, pt.Y)})
	}

	flow := NewFlowField(rows, cols)
	if len(matches) == 0 {
		return flow // no reliable seeds: zero field
	}

	// Spatial scale from seed density; range scale is a fixed intensity spread.
	sigmaS := math.Max(float64(cols+rows)/float64(2*len(matches))+1, 4)
	invS2 := 1.0 / (2 * sigmaS * sigmaS)
	const sigmaC = 20.0
	invC2 := 1.0 / (2 * sigmaC * sigmaC)

	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			pv := pg.at(x, y)
			var su, sv, wsum float64
			for k := range matches {
				m := &matches[k]
				ddx := float64(x - m.x)
				ddy := float64(y - m.y)
				dist2 := ddx*ddx + ddy*ddy
				dc := pv - m.val
				w := math.Exp(-dist2*invS2 - dc*dc*invC2)
				su += w * m.u
				sv += w * m.v
				wsum += w
			}
			if wsum > 0 {
				flow.set(y, x, su/wsum, sv/wsum)
			}
		}
	}
	return flow
}

// gridSeeds returns a regularly spaced grid of seed points with the given
// spacing (in pixels), always including at least one point.
func gridSeeds(rows, cols, spacing int) []image.Point {
	if spacing < 1 {
		spacing = 1
	}
	pts := make([]image.Point, 0)
	for y := spacing / 2; y < rows; y += spacing {
		for x := spacing / 2; x < cols; x += spacing {
			pts = append(pts, image.Point{X: x, Y: y})
		}
	}
	if len(pts) == 0 {
		pts = append(pts, image.Point{X: cols / 2, Y: rows / 2})
	}
	return pts
}

// lucasKanadePoint estimates the displacement of a single point (px, py) from
// prev to next by iterative Lucas-Kanade. The 2x2 structure tensor is built once
// over the (2·winRadius+1)² window from the prev-frame gradients; each iteration
// warps next by the current estimate, forms the temporal residual and solves for
// an increment. The final bool is false when the tensor is ill-conditioned
// (determinant or minimum eigenvalue below a small threshold), signalling an
// unreliable, texture-poor point.
func lucasKanadePoint(prev, next, gx, gy *grid, px, py, winRadius, iters int) (u, v float64, ok bool) {
	var a11, a12, a22 float64
	for wy := -winRadius; wy <= winRadius; wy++ {
		for wx := -winRadius; wx <= winRadius; wx++ {
			gxv := gx.atClamp(px+wx, py+wy)
			gyv := gy.atClamp(px+wx, py+wy)
			a11 += gxv * gxv
			a12 += gxv * gyv
			a22 += gyv * gyv
		}
	}
	det := a11*a22 - a12*a12
	// Minimum eigenvalue of the 2x2 symmetric tensor.
	tr := a11 + a22
	disc := math.Sqrt(math.Max(tr*tr-4*det, 0))
	minEig := (tr - disc) / 2
	if det < 1e-6 || minEig < 1e-3 {
		return 0, 0, false
	}
	invDet := 1.0 / det

	for it := 0; it < iters; it++ {
		var b1, b2 float64
		for wy := -winRadius; wy <= winRadius; wy++ {
			for wx := -winRadius; wx <= winRadius; wx++ {
				sx := px + wx
				sy := py + wy
				ip := prev.atClamp(sx, sy)
				in := next.bilinear(float64(sx)+u, float64(sy)+v)
				// Temporal residual It = next(x+d) - prev(x).
				it := in - ip
				b1 += gx.atClamp(sx, sy) * it
				b2 += gy.atClamp(sx, sy) * it
			}
		}
		// Solve A·Δ = -b for the update that reduces the residual.
		du := -(a22*b1 - a12*b2) * invDet
		dv := -(a11*b2 - a12*b1) * invDet
		u += du
		v += dv
		if du*du+dv*dv < 1e-4 {
			break
		}
	}
	return u, v, true
}
