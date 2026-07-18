package superres

import (
	"math"
	"math/bits"

	cv "github.com/malcolmston/opencv"
)

// nediWindow is the half-width of the local training window used to estimate
// interpolation coefficients.
const nediWindow = 4

// nediRegularization is the relative ridge term added to the diagonal of the
// local covariance matrix (as a fraction of the mean diagonal) so the
// least-squares system stays well-conditioned in flat/smooth regions.
const nediRegularization = 1e-2

// nediEstimate estimates one interpolated sample. It fits, by least squares
// over a (2·nediWindow+1)² window of plane p centred at (ci, cj), a linear
// predictor of a pixel from its four neighbours at the given offsets, then
// applies those coefficients to the supplied target neighbour values nb. When
// wantParity is 0 or 1 only window pixels whose (row+col) parity matches are
// used as training targets (needed in the second NEDI pass, where only the
// even-parity lattice is known); pass -1 to use every window pixel. If the
// system is singular or yields unreasonable weights it falls back to the plain
// average of nb, so smooth and flat regions are always handled gracefully.
func nediEstimate(p *superresPlane, ci, cj int, off [4][2]int, nb [4]float64, wantParity int) float64 {
	var c [16]float64
	var v [4]float64
	for di := -nediWindow; di <= nediWindow; di++ {
		for dj := -nediWindow; dj <= nediWindow; dj++ {
			pi := ci + di
			pj := cj + dj
			if wantParity >= 0 && (pi+pj)&1 != wantParity {
				continue
			}
			y := p.at(pi, pj)
			var x [4]float64
			for k := 0; k < 4; k++ {
				x[k] = p.at(pi+off[k][0], pj+off[k][1])
			}
			for a := 0; a < 4; a++ {
				v[a] += x[a] * y
				for b := 0; b < 4; b++ {
					c[a*4+b] += x[a] * x[b]
				}
			}
		}
	}
	// Tikhonov (ridge) regularisation proportional to the data scale keeps the
	// system well-conditioned in smooth/flat regions, where the four
	// predictors are nearly collinear, while barely biasing the weights on
	// strong edges. An absolute term handles the all-zero window.
	avgDiag := (c[0] + c[5] + c[10] + c[15]) / 4
	ridge := nediRegularization*avgDiag + 1e-9
	for a := 0; a < 4; a++ {
		c[a*4+a] += ridge
	}
	avg := 0.25 * (nb[0] + nb[1] + nb[2] + nb[3])
	w, ok := superresSolve(c[:], v[:], 4)
	if !ok {
		return avg
	}
	var val, wsum float64
	for k := 0; k < 4; k++ {
		val += w[k] * nb[k]
		wsum += w[k]
	}
	// Reject unstable fits (weights that do not roughly form a partition of
	// unity) to avoid ringing artefacts.
	if wsum < 0.5 || wsum > 1.5 || math.IsNaN(val) {
		return avg
	}
	// Bound the interpolated value to the neighbour range (with a small margin
	// for edge overshoot) so border pixels cannot produce gross artefacts.
	lo, hi := nb[0], nb[0]
	for _, u := range nb[1:] {
		if u < lo {
			lo = u
		}
		if u > hi {
			hi = u
		}
	}
	margin := 0.25 * (hi - lo)
	if val < lo-margin {
		val = lo - margin
	} else if val > hi+margin {
		val = hi + margin
	}
	return val
}

// nediDoublePlane doubles the resolution of a single float plane using the
// two-pass New Edge-Directed Interpolation scheme.
func nediDoublePlane(lr *superresPlane) *superresPlane {
	h, w := lr.rows, lr.cols
	hr := newSuperresPlane(2*h, 2*w)
	// Pass 0: copy the low-resolution samples onto the even lattice.
	for i := 0; i < h; i++ {
		for j := 0; j < w; j++ {
			hr.set(2*i, 2*j, lr.atRaw(i, j))
		}
	}
	// Pass 1: interpolate the cell centres (odd, odd) from the four diagonal
	// low-resolution neighbours, with coefficients trained on the
	// low-resolution image (same diagonal geometry across scales).
	diagOff := [4][2]int{{-1, -1}, {-1, 1}, {1, -1}, {1, 1}} // TL, TR, BL, BR
	for i := 0; i < h; i++ {
		for j := 0; j < w; j++ {
			nb := [4]float64{
				lr.at(i, j),     // TL
				lr.at(i, j+1),   // TR
				lr.at(i+1, j),   // BL
				lr.at(i+1, j+1), // BR
			}
			hr.set(2*i+1, 2*j+1, nediEstimate(lr, i, j, diagOff, nb, -1))
		}
	}
	// Pass 2: fill the remaining (odd sum) sites from their four axis-aligned
	// neighbours (all now known). Coefficients are trained on the known
	// even-sum lattice using the geometrically equivalent diagonal
	// configuration.
	knownDiagOff := [4][2]int{{-1, -1}, {-1, 1}, {1, 1}, {1, -1}} // UL, UR, DR, DL
	axisOff := [4][2]int{{-1, 0}, {0, 1}, {1, 0}, {0, -1}}        // N, E, S, W
	for r := 0; r < 2*h; r++ {
		for cc := 0; cc < 2*w; cc++ {
			if (r+cc)%2 == 0 {
				continue // already known
			}
			var nb [4]float64
			var valid [4]bool
			nValid := 0
			var sum float64
			for k := 0; k < 4; k++ {
				ny := r + axisOff[k][0]
				nx := cc + axisOff[k][1]
				if ny < 0 || ny >= 2*h || nx < 0 || nx >= 2*w {
					continue // out of image: not a known sample
				}
				nb[k] = hr.atRaw(ny, nx)
				valid[k] = true
				nValid++
				sum += nb[k]
			}
			if nValid == 4 {
				// Interior: use the covariance-based edge-directed estimate,
				// trained on the known even-parity lattice.
				hr.set(r, cc, nediEstimate(hr, r, cc, knownDiagOff, nb, 0))
			} else {
				// Border: average the available known neighbours.
				hr.set(r, cc, sum/float64(nValid))
			}
		}
	}
	return hr
}

// NEDIDouble upscales src by exactly 2× using New Edge-Directed Interpolation
// (Li & Orchard). Unlike fixed kernels, NEDI adapts to local image structure —
// estimating per-pixel weights from the local covariance — so it interpolates
// along edges instead of across them, markedly reducing the staircasing and
// blur that bicubic upscaling produces on diagonal edges. Each channel is
// processed independently. It panics if src is empty.
func NEDIDouble(src *cv.Mat) *cv.Mat {
	if src.Empty() {
		panic("superres: NEDIDouble on empty Mat")
	}
	planes := superresSplitPlanes(src)
	for i, p := range planes {
		planes[i] = nediDoublePlane(p)
	}
	return superresMergePlanes(planes)
}

// NEDI upscales src by an integer power-of-two scale (2, 4, 8, …) by applying
// [NEDIDouble] repeatedly. It panics if scale is not a power of two >= 1.
// A scale of 1 returns a clone.
func NEDI(src *cv.Mat, scale int) *cv.Mat {
	if scale < 1 || bits.OnesCount(uint(scale)) != 1 {
		panic("superres: NEDI requires scale to be a power of two")
	}
	out := src
	for s := scale; s > 1; s /= 2 {
		out = NEDIDouble(out)
	}
	if out == src {
		return src.Clone()
	}
	return out
}

// EdgeDirectedResize upscales src to an arbitrary width×height using edge-
// directed interpolation: it applies [NEDIDouble] until both dimensions reach
// or exceed the target, then resamples down to the exact size with Catmull-Rom
// bicubic. It is a convenience for when the target is not a clean power-of-two
// multiple of the source. It panics if width or height is smaller than the
// source dimension (this routine only enlarges) or src is empty.
func EdgeDirectedResize(src *cv.Mat, width, height int) *cv.Mat {
	if src.Empty() {
		panic("superres: EdgeDirectedResize on empty Mat")
	}
	if width < src.Cols || height < src.Rows {
		panic("superres: EdgeDirectedResize only enlarges; use Resize to shrink")
	}
	out := src
	for out.Cols < width || out.Rows < height {
		out = NEDIDouble(out)
	}
	if out.Cols == width && out.Rows == height {
		if out == src {
			return src.Clone()
		}
		return out
	}
	return BicubicResize(out, width, height)
}
