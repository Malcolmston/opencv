package optflow

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// SimpleFlowParams configures [CalcOpticalFlowSimpleFlow]. The defaults from
// [DefaultSimpleFlowParams] follow the spirit of Tao et al., "SimpleFlow: A
// Non-iterative, Sublinear Optical Flow Algorithm" (2012).
type SimpleFlowParams struct {
	// Scales is the number of pyramid levels (>= 1); each level extends the
	// reachable displacement.
	Scales int
	// SearchRadius is the per-level integer search radius of candidate flows
	// around the inherited estimate.
	SearchRadius int
	// WinRadius is the half-size of the correlation window over which each
	// candidate flow's energy is accumulated.
	WinRadius int
	// SigmaS is the spatial scale of the bilateral weighting inside the
	// correlation window (image-plane distance).
	SigmaS float64
	// SigmaC is the colour/intensity scale of the bilateral weighting (range
	// term), making the energy edge-aware.
	SigmaC float64
}

// DefaultSimpleFlowParams returns a sensible SimpleFlow configuration
// (Scales=3, SearchRadius=3, WinRadius=3, SigmaS=4.5, SigmaC=25).
func DefaultSimpleFlowParams() SimpleFlowParams {
	return SimpleFlowParams{
		Scales:       3,
		SearchRadius: 3,
		WinRadius:    3,
		SigmaS:       4.5,
		SigmaC:       25.0,
	}
}

func (p SimpleFlowParams) validate() {
	if p.Scales < 1 {
		panic("optflow: SimpleFlow requires Scales >= 1")
	}
	if p.SearchRadius < 1 || p.WinRadius < 1 {
		panic("optflow: SimpleFlow requires SearchRadius >= 1 and WinRadius >= 1")
	}
	if p.SigmaS <= 0 || p.SigmaC <= 0 {
		panic("optflow: SimpleFlow requires SigmaS > 0 and SigmaC > 0")
	}
}

// CalcOpticalFlowSimpleFlow computes a dense flow field from prev to next with
// the SimpleFlow algorithm (Tao et al., 2012), the stdlib port of OpenCV's
// cv::optflow::calcOpticalFlowSF.
//
// The method is non-iterative: at each pyramid level, for every pixel, it
// evaluates the matching energy of each candidate integer flow in a
// SearchRadius window around the flow inherited from the coarser level. A
// candidate's energy is the bilaterally-weighted sum of absolute intensity
// differences over a WinRadius correlation window, where each neighbour's weight
// combines a spatial Gaussian (scale SigmaS) with a range Gaussian on its
// intensity difference from the centre pixel (scale SigmaC) — the same
// edge-preserving weighting as a bilateral filter, which keeps the flow sharp
// across motion boundaries. The candidate energies are turned into a softmin
// distribution and the flow is taken as the energy-weighted mean of the
// candidates, yielding a sub-pixel estimate. The result is refined coarse-to-
// fine over Scales pyramid levels.
//
// prev and next must be non-empty and identically sized; multi-channel inputs
// are converted to grayscale. p is validated (see [SimpleFlowParams]). The
// computation is fully deterministic.
func CalcOpticalFlowSimpleFlow(prev, next *cv.Mat, p SimpleFlowParams) *FlowField {
	requirePair(prev, next, "CalcOpticalFlowSimpleFlow")
	p.validate()

	i0 := grayGrid(prev)
	i1 := grayGrid(next)
	pPyr, nPyr := scalePyramidLevels(i0, i1, p.Scales-1, 2*p.WinRadius+1)
	nl := len(pPyr)

	var u, v []float64
	var pr, pc int
	for lvl := nl - 1; lvl >= 0; lvl-- {
		I0 := pPyr[lvl]
		I1 := nPyr[lvl]
		rows, cols := I0.Rows, I0.Cols
		if u == nil {
			u = make([]float64, rows*cols)
			v = make([]float64, rows*cols)
		} else {
			u, v = upscaleFlow(u, v, pr, pc, rows, cols)
		}
		simpleFlowScale(I0, I1, u, v, p)
		pr, pc = rows, cols
	}
	return flowFromPlanes(u, v, pr, pc)
}

// simpleFlowScale refines u, v in place for one pyramid level.
func simpleFlowScale(I0, I1 *grid, u, v []float64, p SimpleFlowParams) {
	rows, cols := I0.Rows, I0.Cols
	wr := p.WinRadius
	sr := p.SearchRadius
	invS2 := 1.0 / (2 * p.SigmaS * p.SigmaS)
	invC2 := 1.0 / (2 * p.SigmaC * p.SigmaC)

	// Precompute bilateral spatial weights (range term depends on the pixel and
	// is folded in per-pixel below).
	nu := make([]float64, rows*cols)
	nv := make([]float64, rows*cols)

	span := 2*sr + 1
	energy := make([]float64, span*span) // reused per pixel, indexed [dy+sr][dx+sr]
	eAt := func(dx, dy int) float64 { return energy[(dy+sr)*span+(dx+sr)] }

	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			i := y*cols + x
			baseU := math.Round(u[i])
			baseV := math.Round(v[i])
			centre := I0.at(x, y)

			// Evaluate the bilaterally-weighted matching energy of every candidate
			// integer flow, and track the discrete argmin.
			bestE := math.Inf(1)
			var bestDx, bestDy int
			for dy := -sr; dy <= sr; dy++ {
				for dx := -sr; dx <= sr; dx++ {
					fu := baseU + float64(dx)
					fv := baseV + float64(dy)
					var e, wsum float64
					for wy := -wr; wy <= wr; wy++ {
						for wx := -wr; wx <= wr; wx++ {
							nx, ny := x+wx, y+wy
							i0v := I0.atClamp(nx, ny)
							// Bilateral weight: spatial + range (vs centre).
							dc := i0v - centre
							ws := math.Exp(-float64(wx*wx+wy*wy)*invS2 - dc*dc*invC2)
							i1v := I1.bilinear(float64(nx)+fu, float64(ny)+fv)
							e += ws * math.Abs(i0v-i1v)
							wsum += ws
						}
					}
					if wsum > 0 {
						e /= wsum
					}
					energy[(dy+sr)*span+(dx+sr)] = e
					if e < bestE {
						bestE = e
						bestDx = dx
						bestDy = dy
					}
				}
			}

			// Sub-pixel refinement by 1-D parabola fitting around the argmin,
			// independently in x and y (falling back to the integer optimum at the
			// search border where a neighbour is unavailable).
			subDx := parabolicOffset(bestDx, sr, func(k int) float64 { return eAt(k, bestDy) })
			subDy := parabolicOffset(bestDy, sr, func(k int) float64 { return eAt(bestDx, k) })
			nu[i] = baseU + float64(bestDx) + subDx
			nv[i] = baseV + float64(bestDy) + subDy
		}
	}
	copy(u, nu)
	copy(v, nv)
}

// parabolicOffset returns the sub-pixel correction in [-0.5, 0.5] to a discrete
// minimum at index k (within [-r, r]) given the energy sampler e. It fits a
// parabola through e(k-1), e(k), e(k+1); at the search border, or when the
// samples are not convex, it returns 0.
func parabolicOffset(k, r int, e func(int) float64) float64 {
	if k <= -r || k >= r {
		return 0
	}
	em := e(k - 1)
	e0 := e(k)
	ep := e(k + 1)
	den := em - 2*e0 + ep
	if den <= 1e-12 {
		return 0
	}
	off := 0.5 * (em - ep) / den
	if off > 0.5 {
		off = 0.5
	} else if off < -0.5 {
		off = -0.5
	}
	return off
}
