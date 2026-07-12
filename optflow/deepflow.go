package optflow

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// DeepFlowParams configures [DeepFlow]. The defaults from
// [DefaultDeepFlowParams] give a good balance of large-displacement capability
// (from the descriptor-matching term) and smoothness (from the variational
// refinement).
type DeepFlowParams struct {
	// Alpha is the smoothness weight of the variational refinement (larger →
	// smoother flow).
	Alpha float64
	// Beta weights the descriptor-matching constraint that anchors the flow to
	// the coarse block-matching correspondences, enabling large displacements.
	Beta float64
	// Scales is the number of pyramid levels (>= 1).
	Scales int
	// Warps is the number of warping (re-linearisation) steps per scale.
	Warps int
	// Iterations is the number of Gauss-Seidel sweeps per warp.
	Iterations int
	// MatchRadius is the integer search radius of the coarse block matcher.
	MatchRadius int
	// PatchRadius is the half-size of the block-matching correlation window.
	PatchRadius int
}

// DefaultDeepFlowParams returns a sensible DeepFlow configuration
// (Alpha=6, Beta=0.4, Scales=4, Warps=3, Iterations=25, MatchRadius=4,
// PatchRadius=3).
func DefaultDeepFlowParams() DeepFlowParams {
	return DeepFlowParams{
		Alpha:       6.0,
		Beta:        0.4,
		Scales:      4,
		Warps:       3,
		Iterations:  25,
		MatchRadius: 4,
		PatchRadius: 3,
	}
}

func (p DeepFlowParams) validate() {
	if p.Alpha <= 0 {
		panic("optflow: DeepFlow requires Alpha > 0")
	}
	if p.Beta < 0 {
		panic("optflow: DeepFlow requires Beta >= 0")
	}
	if p.Scales < 1 || p.Warps < 1 || p.Iterations < 1 {
		panic("optflow: DeepFlow requires Scales, Warps, Iterations >= 1")
	}
	if p.MatchRadius < 1 || p.PatchRadius < 1 {
		panic("optflow: DeepFlow requires MatchRadius >= 1 and PatchRadius >= 1")
	}
}

// DeepFlow computes a dense optical-flow field from prev to next with a
// lightweight reimplementation of the DeepFlow idea (Weinzaepfel et al., 2013):
// a dense descriptor-matching correspondence field that guides a coarse-to-fine
// variational refinement.
//
// At the coarsest pyramid level an integer block matcher establishes a
// large-displacement correspondence for every pixel by minimising patch SSD over
// a MatchRadius search window. That match field is propagated down the pyramid
// and enters the variational energy as an extra quadratic anchoring term (weight
// Beta), alongside the linearised brightness-constancy data term and a
// first-order smoothness prior (weight Alpha). Each pyramid level warps the next
// frame by the current flow and runs Iterations Gauss-Seidel sweeps of the
// resulting 2×2-per-pixel system, repeated over Warps re-linearisations. The
// matching term lets the method follow motions far larger than a purely
// differential solver such as [CalcOpticalFlowDenseHS] could, while the
// variational term keeps the field smooth and sub-pixel accurate.
//
// prev and next must be non-empty and identically sized; multi-channel inputs
// are converted to grayscale. p is validated (see [DeepFlowParams]). The
// computation is fully deterministic.
func DeepFlow(prev, next *cv.Mat, p DeepFlowParams) *FlowField {
	requirePair(prev, next, "DeepFlow")
	p.validate()

	i0 := grayGrid(prev)
	i1 := grayGrid(next)
	pPyr, nPyr := scalePyramidLevels(i0, i1, p.Scales-1, 2*p.PatchRadius+1)
	nl := len(pPyr)

	var u, v []float64
	var mu, mv, mw []float64 // descriptor-match field + per-pixel confidence
	var pr, pc int

	for lvl := nl - 1; lvl >= 0; lvl-- {
		I0 := pPyr[lvl]
		I1 := nPyr[lvl]
		rows, cols := I0.Rows, I0.Cols

		if u == nil {
			u = make([]float64, rows*cols)
			v = make([]float64, rows*cols)
			// Establish large-displacement correspondences at the coarsest level,
			// keeping only unambiguous (ratio-test) matches so the periodic or
			// texture-poor regions fall back to the variational term.
			mu, mv, mw = blockMatchField(I0, I1, p.MatchRadius, p.PatchRadius)
			for i := range u {
				if mw[i] > 0 {
					u[i] = mu[i]
					v[i] = mv[i]
				}
			}
		} else {
			u, v = upscaleFlow(u, v, pr, pc, rows, cols)
			mu, mv = upscaleFlow(mu, mv, pr, pc, rows, cols)
			mw = upscaleConfidence(mw, pr, pc, rows, cols)
		}

		deepFlowScale(I0, I1, u, v, mu, mv, mw, p)
		pr, pc = rows, cols
	}
	return flowFromPlanes(u, v, pr, pc)
}

// blockMatchField returns, for every pixel, the integer displacement in
// [-radius, radius]² minimising the SSD of a (2·patchRadius+1)² patch between a
// and b, together with a per-pixel confidence mw (1 for an unambiguous match, 0
// otherwise). A match is accepted only when the best SSD is clearly better than
// the best competing displacement at least two pixels away (Lowe-style ratio
// test), which rejects the ambiguous optima of periodic or flat texture. Ties
// break toward the smaller-magnitude offset then row-major order, so the field
// is deterministic.
func blockMatchField(a, b *grid, radius, patchRadius int) (mu, mv, mw []float64) {
	rows, cols := a.Rows, a.Cols
	mu = make([]float64, rows*cols)
	mv = make([]float64, rows*cols)
	mw = make([]float64, rows*cols)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			best := math.Inf(1)
			bestMag := math.Inf(1)
			var bdx, bdy int
			for dy := -radius; dy <= radius; dy++ {
				for dx := -radius; dx <= radius; dx++ {
					var ssd float64
					for wy := -patchRadius; wy <= patchRadius; wy++ {
						for wx := -patchRadius; wx <= patchRadius; wx++ {
							pa := a.atClamp(x+wx, y+wy)
							pb := b.atClamp(x+wx+dx, y+wy+dy)
							d := pa - pb
							ssd += d * d
						}
					}
					mag := float64(dx*dx + dy*dy)
					if ssd < best || (ssd == best && mag < bestMag) {
						best = ssd
						bestMag = mag
						bdx = dx
						bdy = dy
					}
				}
			}
			// Second-best SSD among displacements well separated from the best.
			second := math.Inf(1)
			for dy := -radius; dy <= radius; dy++ {
				for dx := -radius; dx <= radius; dx++ {
					if (dx-bdx)*(dx-bdx)+(dy-bdy)*(dy-bdy) < 4 {
						continue
					}
					var ssd float64
					for wy := -patchRadius; wy <= patchRadius; wy++ {
						for wx := -patchRadius; wx <= patchRadius; wx++ {
							pa := a.atClamp(x+wx, y+wy)
							pb := b.atClamp(x+wx+dx, y+wy+dy)
							d := pa - pb
							ssd += d * d
						}
					}
					if ssd < second {
						second = ssd
					}
				}
			}
			i := y*cols + x
			mu[i] = float64(bdx)
			mv[i] = float64(bdy)
			if second > 1e-9 && best <= 0.6*second {
				mw[i] = 1
			}
		}
	}
	return mu, mv, mw
}

// upscaleConfidence resamples a coarse per-pixel confidence map to a finer grid
// with nearest-neighbour sampling, so a match stays trusted (weight 1) or
// untrusted (weight 0) without being blurred into fractional values.
func upscaleConfidence(cw []float64, oldRows, oldCols, newRows, newCols int) []float64 {
	out := make([]float64, newRows*newCols)
	for y := 0; y < newRows; y++ {
		sy := clampInt(y*oldRows/newRows, 0, oldRows-1)
		for x := 0; x < newCols; x++ {
			sx := clampInt(x*oldCols/newCols, 0, oldCols-1)
			out[y*newCols+x] = cw[sy*oldCols+sx]
		}
	}
	return out
}

// deepFlowScale refines u, v in place for one pyramid level by warping the next
// frame and solving the coupled data + smoothness + matching system with
// Gauss-Seidel sweeps.
func deepFlowScale(I0, I1 *grid, u, v, mu, mv, mw []float64, p DeepFlowParams) {
	rows, cols := I0.Rows, I0.Cols
	n := rows * cols
	i1x, i1y := sobelGradients(I1)
	a2 := p.Alpha * p.Alpha

	for warp := 0; warp < p.Warps; warp++ {
		i1w := warpGrid(I1, u, v)
		iwx := warpGrid(i1x, u, v)
		iwy := warpGrid(i1y, u, v)

		// Constant part of the linearised temporal term:
		// It' = I1w - I0 - Ix*u0 - Iy*v0 with (u0, v0) the warp flow.
		itc := make([]float64, n)
		u0 := make([]float64, n)
		v0 := make([]float64, n)
		copy(u0, u)
		copy(v0, v)
		for i := 0; i < n; i++ {
			itc[i] = i1w.Data[i] - I0.Data[i] - iwx.Data[i]*u0[i] - iwy.Data[i]*v0[i]
		}

		for it := 0; it < p.Iterations; it++ {
			for y := 0; y < rows; y++ {
				for x := 0; x < cols; x++ {
					i := y*cols + x
					ub := neighbourAvg(u, x, y, rows, cols)
					vb := neighbourAvg(v, x, y, rows, cols)
					ix := iwx.Data[i]
					iy := iwy.Data[i]
					// The matching term is active only where the descriptor match
					// passed the ratio test (mw[i] == 1); elsewhere the flow is
					// driven purely by the data and smoothness terms.
					beta := p.Beta * mw[i]
					// Solve the per-pixel 2×2 system arising from
					// (Ix u + Iy v + It')² + α²|u-ub|² + β|u-m|².
					a11 := ix*ix + a2 + beta
					a12 := ix * iy
					a22 := iy*iy + a2 + beta
					b1 := a2*ub + beta*mu[i] - ix*itc[i]
					b2 := a2*vb + beta*mv[i] - iy*itc[i]
					nu, nv := solve2x2(a11, a12, a22, b1, b2)
					u[i] = nu
					v[i] = nv
				}
			}
		}
	}
}

// neighbourAvg returns the 4-neighbour average of a plane at (x, y) with border
// replication — the discrete Laplacian smoothing term of the variational solver.
func neighbourAvg(f []float64, x, y, rows, cols int) float64 {
	cx := func(v int) int { return clampInt(v, 0, cols-1) }
	cy := func(v int) int { return clampInt(v, 0, rows-1) }
	idx := func(xx, yy int) int { return cy(yy)*cols + cx(xx) }
	return 0.25 * (f[idx(x-1, y)] + f[idx(x+1, y)] + f[idx(x, y-1)] + f[idx(x, y+1)])
}
