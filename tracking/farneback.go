package tracking

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// FarnebackParams configures the dense block-matching flow estimator
// [CalcOpticalFlowFarneback].
type FarnebackParams struct {
	// Levels is the number of pyramid levels (>= 1). Coarser levels capture
	// larger motions. Zero is treated as 3.
	Levels int
	// WindowRadius is the half-size of the square correlation patch matched
	// between frames. Zero is treated as 4.
	WindowRadius int
	// SearchRadius is the half-size, in pixels, of the displacement search at
	// each level. Zero is treated as 4.
	SearchRadius int
}

// DefaultFarnebackParams returns sensible defaults (3 levels, window radius 4,
// search radius 4).
func DefaultFarnebackParams() FarnebackParams {
	return FarnebackParams{Levels: 3, WindowRadius: 4, SearchRadius: 4}
}

// normalized fills in default values for zero fields.
func (p FarnebackParams) normalized() FarnebackParams {
	if p.Levels <= 0 {
		p.Levels = 3
	}
	if p.WindowRadius <= 0 {
		p.WindowRadius = 4
	}
	if p.SearchRadius <= 0 {
		p.SearchRadius = 4
	}
	return p
}

// CalcOpticalFlowFarneback computes a dense optical-flow field from prev to next.
//
// This is a compact, deterministic block-matching estimator that stands in for
// Gunnar Farneback's polynomial-expansion algorithm (the same practical role the
// sibling cv/video package fills): flow is estimated coarse-to-fine over a
// Gaussian pyramid, and at each level and pixel the integer displacement within
// a search window that minimises the sum-of-squared-differences of a square
// correlation patch is selected, refined to sub-pixel accuracy by a parabola fit
// on the SSD surface. Coarse-level estimates are propagated (and doubled) to
// finer levels so displacements larger than the patch can be recovered.
//
// prev and next must be non-empty and identically sized; multi-channel inputs
// are converted to grayscale. The computation is fully deterministic.
func CalcOpticalFlowFarneback(prev, next *cv.Mat, params FarnebackParams) *FlowField {
	requirePair(prev, next, "CalcOpticalFlowFarneback")
	p := params.normalized()
	pyrPrev := trackingBuildPyramid(trackingToGrayF(prev), p.Levels-1)
	pyrNext := trackingBuildPyramid(trackingToGrayF(next), p.Levels-1)
	top := len(pyrPrev) - 1

	var u, v []float64 // flow at the current (coarser to finer) level
	var curRows, curCols int

	for lvl := top; lvl >= 0; lvl-- {
		gp := pyrPrev[lvl]
		gn := pyrNext[lvl]
		rows, cols := gp.rows, gp.cols

		if u == nil {
			u = make([]float64, rows*cols)
			v = make([]float64, rows*cols)
		} else {
			// Upsample the flow from the coarser level and double it.
			nu := make([]float64, rows*cols)
			nv := make([]float64, rows*cols)
			for y := 0; y < rows; y++ {
				sy := y * curRows / rows
				if sy >= curRows {
					sy = curRows - 1
				}
				for x := 0; x < cols; x++ {
					sx := x * curCols / cols
					if sx >= curCols {
						sx = curCols - 1
					}
					nu[y*cols+x] = u[sy*curCols+sx] * 2
					nv[y*cols+x] = v[sy*curCols+sx] * 2
				}
			}
			u, v = nu, nv
		}

		w := p.WindowRadius
		s := p.SearchRadius
		newU := make([]float64, rows*cols)
		newV := make([]float64, rows*cols)
		for y := 0; y < rows; y++ {
			for x := 0; x < cols; x++ {
				i := y*cols + x
				baseU := u[i]
				baseV := v[i]
				bestU, bestV, _ := trackingBlockSearch(gp, gn, x, y, w, s, baseU, baseV)
				newU[i] = bestU
				newV[i] = bestV
			}
		}
		u, v = newU, newV
		curRows, curCols = rows, cols
	}

	ff := NewFlowField(curRows, curCols)
	copy(ff.u, u)
	copy(ff.v, v)
	return ff
}

// trackingBlockSearch finds the sub-pixel displacement of the patch centred at
// (x, y) in gp that best matches gn, searching integer offsets in
// [-s, s] around the seed (baseU, baseV) and refining with a 1-D parabola fit.
func trackingBlockSearch(gp, gn *trackingGray, x, y, w, s int, baseU, baseV float64) (du, dv, cost float64) {
	bu := int(math.Round(baseU))
	bv := int(math.Round(baseV))
	bestCost := math.Inf(1)
	bestDX, bestDY := bu, bv
	costAt := func(ddx, ddy int) float64 {
		var sum float64
		for wy := -w; wy <= w; wy++ {
			for wx := -w; wx <= w; wx++ {
				a := gp.at(y+wy, x+wx)
				b := gn.at(y+wy+ddy, x+wx+ddx)
				d := a - b
				sum += d * d
			}
		}
		return sum
	}
	for oy := -s; oy <= s; oy++ {
		for ox := -s; ox <= s; ox++ {
			c := costAt(bu+ox, bv+oy)
			if c < bestCost {
				bestCost = c
				bestDX = bu + ox
				bestDY = bv + oy
			}
		}
	}
	// Sub-pixel refinement via parabola fit on the SSD surface.
	fx := subpixel(costAt(bestDX-1, bestDY), bestCost, costAt(bestDX+1, bestDY))
	fy := subpixel(costAt(bestDX, bestDY-1), bestCost, costAt(bestDX, bestDY+1))
	return float64(bestDX) + fx, float64(bestDY) + fy, bestCost
}

// subpixel returns the offset in [-0.5, 0.5] of the parabola vertex through the
// three samples (cm, c0, cp) at positions (-1, 0, 1). It returns 0 when the
// samples do not form an upward parabola.
func subpixel(cm, c0, cp float64) float64 {
	denom := cm - 2*c0 + cp
	if denom <= 0 {
		return 0
	}
	off := 0.5 * (cm - cp) / denom
	if off > 0.5 {
		off = 0.5
	} else if off < -0.5 {
		off = -0.5
	}
	return off
}
