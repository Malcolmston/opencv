package inpaint

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// InpaintDiffusion fills the pixels of img selected by mask by solving the
// Laplace equation over the region with the surrounding pixels as Dirichlet
// boundary conditions — the smoothest (harmonic) interpolation that agrees with
// the border. It is solved by in-place Gauss-Seidel sweeps until convergence or
// an iteration cap. A uniform surround yields that uniform value exactly; a
// linear ramp is reproduced exactly. img is not modified — a filled clone is
// returned. img may be single- or three-channel; mask must match its size.
func InpaintDiffusion(img *cv.Mat, mask *Mask) *cv.Mat {
	inpaintRequireImage(img, "InpaintDiffusion")
	inpaintRequireMaskMatch(img, mask, "InpaintDiffusion")
	out := img.Clone()
	inpaintHarmonicFill(out, mask)
	return out
}

// inpaintHarmonicFill solves Laplace's equation on the masked region of m
// in place, seeding the interior with the boundary mean for a faster start.
func inpaintHarmonicFill(m *cv.Mat, mask *Mask) {
	cols, ch := m.Cols, m.Channels
	var idxs []int
	for i, v := range mask.Data {
		if v {
			idxs = append(idxs, i)
		}
	}
	if len(idxs) == 0 {
		return
	}

	// Seed interior with the mean of the known pixels.
	meanCh := make([]float64, ch)
	var cnt float64
	for i := range mask.Data {
		if mask.Data[i] {
			continue
		}
		y, x := i/cols, i%cols
		for c := 0; c < ch; c++ {
			meanCh[c] += float64(m.At(y, x, c))
		}
		cnt++
	}
	if cnt > 0 {
		for c := 0; c < ch; c++ {
			meanCh[c] /= cnt
		}
		for _, i := range idxs {
			y, x := i/cols, i%cols
			for c := 0; c < ch; c++ {
				m.Set(y, x, c, inpaintClampU8(meanCh[c]))
			}
		}
	}

	const maxIter = 10000
	const tol = 0.05
	for iter := 0; iter < maxIter; iter++ {
		var maxDelta float64
		for _, i := range idxs {
			y, x := i/cols, i%cols
			for c := 0; c < ch; c++ {
				v := 0.25 * (float64(inpaintAtRep(m, y-1, x, c)) + float64(inpaintAtRep(m, y+1, x, c)) +
					float64(inpaintAtRep(m, y, x-1, c)) + float64(inpaintAtRep(m, y, x+1, c)))
				old := float64(m.At(y, x, c))
				if d := math.Abs(v - old); d > maxDelta {
					maxDelta = d
				}
				m.Set(y, x, c, inpaintClampU8(v))
			}
		}
		if maxDelta < tol {
			break
		}
	}
}

// InpaintNavierStokes fills the pixels of img selected by mask with a
// Bertalmio-style fluid-dynamics scheme: the region is first initialised
// harmonically (see [InpaintDiffusion]) and then refined by transporting the
// image smoothness (its Laplacian) along the isophotes — the level lines
// perpendicular to the intensity gradient — so that edges arriving at the
// boundary are continued into the hole rather than merely blurred across it.
//
// img is not modified — a filled clone is returned. img may be single- or
// three-channel; mask must match its size. Because a uniform or linear region
// has zero Laplacian, the transport is inert on such regions and they are
// reproduced exactly by the harmonic initialisation. iterations bounds the
// transport sweeps (a non-positive value uses a sensible default).
func InpaintNavierStokes(img *cv.Mat, mask *Mask, iterations int) *cv.Mat {
	inpaintRequireImage(img, "InpaintNavierStokes")
	inpaintRequireMaskMatch(img, mask, "InpaintNavierStokes")
	if iterations <= 0 {
		iterations = 200
	}
	out := img.Clone()
	inpaintHarmonicFill(out, mask)

	rows, cols, ch := out.Rows, out.Cols, out.Channels
	var idxs []int
	for i, v := range mask.Data {
		if v {
			idxs = append(idxs, i)
		}
	}
	if len(idxs) == 0 {
		return out
	}

	// Work per channel in float to preserve precision across sweeps.
	const dt = 0.05
	work := make([]float64, rows*cols)
	lap := make([]float64, rows*cols)
	for c := 0; c < ch; c++ {
		for i := 0; i < rows*cols; i++ {
			y, x := i/cols, i%cols
			work[i] = float64(out.At(y, x, c))
		}
		for it := 0; it < iterations; it++ {
			// Smoothness field w = Laplacian of the current estimate.
			for i := 0; i < rows*cols; i++ {
				y, x := i/cols, i%cols
				lap[i] = laplaceAt(work, cols, rows, y, x)
			}
			var maxDelta float64
			for _, i := range idxs {
				y, x := i/cols, i%cols
				// Isophote direction: perpendicular to the intensity gradient,
				// N = (-Iy, Ix).
				ix := centralDiff(work, cols, rows, y, x, 1)
				iy := centralDiff(work, cols, rows, y, x, 0)
				nx, ny := -iy, ix
				// Advect the smoothness along the isophote: -N·∇w.
				wx := centralDiff(lap, cols, rows, y, x, 1)
				wy := centralDiff(lap, cols, rows, y, x, 0)
				beta := -(nx*wx + ny*wy)
				// Normalise by gradient magnitude for stability.
				g := math.Hypot(ix, iy) + 1e-3
				upd := dt * beta / g
				// Clamp the per-step update so the explicit scheme stays stable.
				if upd > 1 {
					upd = 1
				} else if upd < -1 {
					upd = -1
				}
				nv := work[i] + upd
				if nv < 0 {
					nv = 0
				} else if nv > 255 {
					nv = 255
				}
				if d := math.Abs(nv - work[i]); d > maxDelta {
					maxDelta = d
				}
				work[i] = nv
			}
			// Light anisotropic diffusion sweep keeps the transport stable and
			// re-imposes the harmonic tendency inside the hole.
			for _, i := range idxs {
				y, x := i/cols, i%cols
				work[i] = 0.25 * (valRep(work, cols, rows, y-1, x) + valRep(work, cols, rows, y+1, x) +
					valRep(work, cols, rows, y, x-1) + valRep(work, cols, rows, y, x+1))
			}
			if maxDelta < 1e-3 {
				break
			}
		}
		for _, i := range idxs {
			y, x := i/cols, i%cols
			out.Set(y, x, c, inpaintClampU8(work[i]))
		}
	}
	return out
}

// valRep reads work at (y, x) with BORDER_REPLICATE clamping.
func valRep(work []float64, cols, rows, y, x int) float64 {
	y = inpaintClampInt(y, 0, rows-1)
	x = inpaintClampInt(x, 0, cols-1)
	return work[y*cols+x]
}

// centralDiff returns the central difference of work along axis (0=y, 1=x).
func centralDiff(work []float64, cols, rows, y, x, axis int) float64 {
	if axis == 1 {
		return (valRep(work, cols, rows, y, x+1) - valRep(work, cols, rows, y, x-1)) / 2
	}
	return (valRep(work, cols, rows, y+1, x) - valRep(work, cols, rows, y-1, x)) / 2
}

// laplaceAt returns the 5-point Laplacian of work at (y, x).
func laplaceAt(work []float64, cols, rows, y, x int) float64 {
	return valRep(work, cols, rows, y-1, x) + valRep(work, cols, rows, y+1, x) +
		valRep(work, cols, rows, y, x-1) + valRep(work, cols, rows, y, x+1) -
		4*valRep(work, cols, rows, y, x)
}
