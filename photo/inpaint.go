package photo

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// InpaintMethod selects the region-filling algorithm used by [Inpaint].
type InpaintMethod int

const (
	// InpaintTelea fills the masked region with a simplified fast-marching
	// scheme (after Telea, 2004): unknown pixels are processed in order of
	// increasing distance from the mask boundary, each set to a distance-weighted
	// average of its already-known neighbours. It reconstructs smooth gradients
	// from the boundary inward.
	InpaintTelea InpaintMethod = iota
	// InpaintNS fills the masked region by solving the Laplace equation over it
	// (a diffusion / harmonic fill) with the surrounding pixels as boundary
	// conditions, via in-place Gauss–Seidel iteration. The result is the
	// smoothest interpolation that agrees with the border.
	InpaintNS
)

// Inpaint reconstructs the pixels of img marked by mask from the surrounding,
// unmasked pixels. mask must have the same width and height as img; a pixel is
// inpainted where mask's first channel is non-zero. img may be single- or
// three-channel. radius sets the neighbourhood radius (in pixels, minimum 1)
// used when averaging known neighbours for [InpaintTelea]; it is ignored by
// [InpaintNS]. The original img is not modified — a filled clone is returned.
//
// Both methods propagate colour inward, so a masked hole surrounded by a
// uniform region is filled with that region's colour.
func Inpaint(img, mask *cv.Mat, radius float64, method InpaintMethod) *cv.Mat {
	if img == nil || img.Empty() {
		panic("photo: Inpaint given an empty image")
	}
	requireSameSize(img, mask, "Inpaint")
	rows, cols := img.Rows, img.Cols

	// known[i] is true where the pixel is kept (not inpainted).
	known := make([]bool, rows*cols)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			known[y*cols+x] = mask.At(y, x, 0) == 0
		}
	}

	out := img.Clone()
	switch method {
	case InpaintNS:
		diffusionInpaint(out, known)
	default: // InpaintTelea
		r := int(radius)
		if r < 1 {
			r = 1
		}
		teleaInpaint(out, known, r)
	}
	return out
}

// teleaInpaint fills unknown pixels in ascending order of their step-distance
// from the known region, each as a distance-weighted average of known
// neighbours inside a (2r+1)² window.
func teleaInpaint(m *cv.Mat, known []bool, r int) {
	rows, cols, ch := m.Rows, m.Cols, m.Channels

	// Multi-source BFS from the known region assigns each unknown pixel a layer
	// (its distance in steps to the nearest known pixel); the traversal order is
	// exactly the fill order we want.
	dist := make([]int, rows*cols)
	filled := make([]bool, rows*cols)
	copy(filled, known)
	var frontier []int
	for i, k := range known {
		if k {
			dist[i] = 0
		} else {
			dist[i] = -1
		}
	}
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			if !known[y*cols+x] {
				continue
			}
			// A known pixel adjacent to an unknown seeds the frontier.
			for _, d := range neighbors4 {
				ny, nx := y+d[0], x+d[1]
				if ny >= 0 && ny < rows && nx >= 0 && nx < cols && !known[ny*cols+nx] {
					frontier = append(frontier, y*cols+x)
					break
				}
			}
		}
	}

	var order []int
	seen := make([]bool, rows*cols)
	for _, i := range frontier {
		seen[i] = true
	}
	for len(frontier) > 0 {
		var next []int
		for _, i := range frontier {
			y, x := i/cols, i%cols
			for _, d := range neighbors4 {
				ny, nx := y+d[0], x+d[1]
				if ny < 0 || ny >= rows || nx < 0 || nx >= cols {
					continue
				}
				j := ny*cols + nx
				if known[j] || seen[j] {
					continue
				}
				seen[j] = true
				dist[j] = dist[i] + 1
				order = append(order, j)
				next = append(next, j)
			}
		}
		frontier = next
	}

	for _, i := range order {
		y, x := i/cols, i%cols
		for c := 0; c < ch; c++ {
			var sum, wsum float64
			for dy := -r; dy <= r; dy++ {
				for dx := -r; dx <= r; dx++ {
					ny, nx := y+dy, x+dx
					if ny < 0 || ny >= rows || nx < 0 || nx >= cols {
						continue
					}
					j := ny*cols + nx
					if !filled[j] {
						continue
					}
					w := 1.0 / (1.0 + math.Hypot(float64(dy), float64(dx)))
					sum += w * float64(m.At(ny, nx, c))
					wsum += w
				}
			}
			if wsum > 0 {
				m.Set(y, x, c, clampU8(sum/wsum))
			}
		}
		filled[i] = true
	}
}

// diffusionInpaint solves the Laplace equation on the unknown region with the
// known pixels as Dirichlet boundary, using in-place Gauss–Seidel sweeps until
// the largest update falls below a small threshold (or an iteration cap).
func diffusionInpaint(m *cv.Mat, known []bool) {
	rows, cols, ch := m.Rows, m.Cols, m.Channels

	// Seed unknown pixels with the mean of the known pixels for a faster start.
	meanCh := make([]float64, ch)
	var cnt float64
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			if known[y*cols+x] {
				for c := 0; c < ch; c++ {
					meanCh[c] += float64(m.At(y, x, c))
				}
				cnt++
			}
		}
	}
	if cnt > 0 {
		for c := 0; c < ch; c++ {
			meanCh[c] /= cnt
		}
		for y := 0; y < rows; y++ {
			for x := 0; x < cols; x++ {
				if !known[y*cols+x] {
					for c := 0; c < ch; c++ {
						m.Set(y, x, c, clampU8(meanCh[c]))
					}
				}
			}
		}
	}

	const maxIter = 10000
	const tol = 0.3
	for iter := 0; iter < maxIter; iter++ {
		var maxDelta float64
		for y := 0; y < rows; y++ {
			for x := 0; x < cols; x++ {
				if known[y*cols+x] {
					continue
				}
				for c := 0; c < ch; c++ {
					v := 0.25 * (float64(atRep(m, y-1, x, c)) + float64(atRep(m, y+1, x, c)) +
						float64(atRep(m, y, x-1, c)) + float64(atRep(m, y, x+1, c)))
					old := float64(m.At(y, x, c))
					if d := math.Abs(v - old); d > maxDelta {
						maxDelta = d
					}
					m.Set(y, x, c, clampU8(v))
				}
			}
		}
		if maxDelta < tol {
			break
		}
	}
}

var neighbors4 = [4][2]int{{-1, 0}, {1, 0}, {0, -1}, {0, 1}}
