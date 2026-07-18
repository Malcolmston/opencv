package inpaint

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// CriminisiOptions configures [InpaintCriminisi].
type CriminisiOptions struct {
	// PatchRadius is the half-size of the square patch; the patch side is
	// 2*PatchRadius+1. Values of 2..4 (5x5..9x9) are typical. A non-positive
	// value uses 4.
	PatchRadius int
	// SearchRadius limits candidate source-patch centres to a window of this
	// half-size around the target patch, greatly reducing cost on large images.
	// A non-positive value searches the whole image.
	SearchRadius int
}

// DefaultCriminisiOptions returns the default exemplar-based settings
// (PatchRadius 4, whole-image search).
func DefaultCriminisiOptions() CriminisiOptions {
	return CriminisiOptions{PatchRadius: 4, SearchRadius: 0}
}

// InpaintCriminisi fills the pixels of img selected by mask using Criminisi,
// Pérez and Toyama's (2004) exemplar-based image completion. It repeatedly
// selects the highest-priority patch on the fill front — priority being the
// product of a confidence term (how much of the patch is already reliable) and
// a data term (how strongly an isophote/edge meets the front there) — and fills
// its unknown pixels by copying the best-matching fully-known patch found
// elsewhere in the image. This propagates both linear structure and texture, so
// it outperforms pure diffusion on textured or edge-rich holes.
//
// img may be single- or three-channel; mask must match its size (true = fill).
// img is not modified — a filled clone is returned. A uniform or single-region
// surround is reproduced exactly. This is the most expensive routine in the
// package; keep images and holes modest.
func InpaintCriminisi(img *cv.Mat, mask *Mask, opts CriminisiOptions) *cv.Mat {
	inpaintRequireImage(img, "InpaintCriminisi")
	inpaintRequireMaskMatch(img, mask, "InpaintCriminisi")
	half := opts.PatchRadius
	if half <= 0 {
		half = 4
	}
	out := img.Clone()
	rows, cols, ch := out.Rows, out.Cols, out.Channels

	// filled[i] true where the pixel value is reliable (never masked, or already
	// completed). confidence holds Criminisi's C term.
	filled := make([]bool, rows*cols)
	confidence := make([]float64, rows*cols)
	remaining := 0
	for i, v := range mask.Data {
		if v {
			remaining++
		} else {
			filled[i] = true
			confidence[i] = 1
		}
	}
	if remaining == 0 {
		return out
	}

	patchArea := float64((2*half + 1) * (2*half + 1))

	for remaining > 0 {
		// 1. Fill front: masked pixels adjacent (4-conn) to a filled pixel.
		var front [][2]int
		for i := 0; i < rows*cols; i++ {
			if filled[i] {
				continue
			}
			y, x := i/cols, i%cols
			isFront := false
			for _, d := range neighbors4 {
				ny, nx := y+d[0], x+d[1]
				if ny >= 0 && ny < rows && nx >= 0 && nx < cols && filled[ny*cols+nx] {
					isFront = true
					break
				}
			}
			if isFront {
				front = append(front, [2]int{y, x})
			}
		}
		if len(front) == 0 {
			// Isolated remaining pixels (should not happen): fill harmonically.
			for i := 0; i < rows*cols; i++ {
				if !filled[i] {
					y, x := i/cols, i%cols
					for c := 0; c < ch; c++ {
						out.Set(y, x, c, inpaintClampU8(inpaintNeighborMean(out, filled, y, x, c)))
					}
					filled[i] = true
					remaining--
				}
			}
			break
		}

		// 2. Priorities: pick the front pixel maximising confidence*data.
		bestP := -1.0
		var target [2]int
		var targetConf float64
		for _, p := range front {
			y, x := p[0], p[1]
			// Confidence term.
			var cSum float64
			for dy := -half; dy <= half; dy++ {
				for dx := -half; dx <= half; dx++ {
					ny, nx := y+dy, x+dx
					if ny >= 0 && ny < rows && nx >= 0 && nx < cols && filled[ny*cols+nx] {
						cSum += confidence[ny*cols+nx]
					}
				}
			}
			conf := cSum / patchArea
			// Data term: |isophote · normal| / 255.
			nx, ny := inpaintFrontNormal(filled, cols, rows, y, x)
			ix, iy := inpaintIsophote(out, filled, y, x)
			data := math.Abs(ix*nx+iy*ny)/255.0 + 1e-3
			pr := conf * data
			if pr > bestP {
				bestP = pr
				target = [2]int{y, x}
				targetConf = conf
			}
		}

		// 3. Best-matching source patch (SSD over known pixels of the target).
		ty, tx := target[0], target[1]
		by, bx := inpaintBestExemplar(out, filled, ty, tx, half, opts.SearchRadius)

		// 4. Copy unknown target-patch pixels from the source patch.
		for dy := -half; dy <= half; dy++ {
			for dx := -half; dx <= half; dx++ {
				py, px := ty+dy, tx+dx
				if py < 0 || py >= rows || px < 0 || px >= cols {
					continue
				}
				if filled[py*cols+px] {
					continue
				}
				sy := inpaintClampInt(by+dy, 0, rows-1)
				sx := inpaintClampInt(bx+dx, 0, cols-1)
				for c := 0; c < ch; c++ {
					out.Set(py, px, c, out.At(sy, sx, c))
				}
				filled[py*cols+px] = true
				confidence[py*cols+px] = targetConf
				remaining--
			}
		}
	}
	return out
}

// inpaintNeighborMean averages the filled 4-neighbours of (y, x) for channel c.
func inpaintNeighborMean(m *cv.Mat, filled []bool, y, x, c int) float64 {
	cols := m.Cols
	var s, n float64
	for _, d := range neighbors4 {
		ny, nx := y+d[0], x+d[1]
		if ny >= 0 && ny < m.Rows && nx >= 0 && nx < m.Cols && filled[ny*cols+nx] {
			s += float64(m.At(ny, nx, c))
			n++
		}
	}
	if n == 0 {
		return float64(m.At(y, x, c))
	}
	return s / n
}

// inpaintFrontNormal returns the unit normal to the fill front at (y, x),
// estimated as the gradient of the filled indicator (points from hole toward
// known).
func inpaintFrontNormal(filled []bool, cols, rows, y, x int) (nx, ny float64) {
	ind := func(yy, xx int) float64 {
		yy = inpaintClampInt(yy, 0, rows-1)
		xx = inpaintClampInt(xx, 0, cols-1)
		if filled[yy*cols+xx] {
			return 1
		}
		return 0
	}
	nx = (ind(y, x+1) - ind(y, x-1)) / 2
	ny = (ind(y+1, x) - ind(y-1, x)) / 2
	mag := math.Hypot(nx, ny)
	if mag < 1e-9 {
		return 0, 0
	}
	return nx / mag, ny / mag
}

// inpaintIsophote returns the isophote direction (∇I rotated 90°) of the luma at
// (y, x), computed from filled neighbours only.
func inpaintIsophote(m *cv.Mat, filled []bool, y, x int) (ix, iy float64) {
	cols := m.Cols
	lum := func(yy, xx int) (float64, bool) {
		if yy < 0 || yy >= m.Rows || xx < 0 || xx >= m.Cols || !filled[yy*cols+xx] {
			return 0, false
		}
		return inpaintLuma(m, yy, xx), true
	}
	c, _ := lum(y, x)
	if l, ok := lum(y, x+1); ok {
		if r, ok2 := lum(y, x-1); ok2 {
			ix = (l - r) / 2
		} else {
			ix = l - c
		}
	} else if r, ok := lum(y, x-1); ok {
		ix = c - r
	}
	if d, ok := lum(y+1, x); ok {
		if u, ok2 := lum(y-1, x); ok2 {
			iy = (d - u) / 2
		} else {
			iy = d - c
		}
	} else if u, ok := lum(y-1, x); ok {
		iy = c - u
	}
	// Rotate gradient by 90° to get the isophote (edge) direction.
	return -iy, ix
}

// inpaintBestExemplar finds the fully-filled patch centre minimising the sum of
// squared differences against the filled pixels of the target patch at (ty, tx).
// Ties are broken by scan order (smallest y then x), keeping the search
// deterministic.
func inpaintBestExemplar(m *cv.Mat, filled []bool, ty, tx, half, searchRadius int) (by, bx int) {
	rows, cols, ch := m.Rows, m.Cols, m.Channels
	y0, y1 := half, rows-1-half
	x0, x1 := half, cols-1-half
	if searchRadius > 0 {
		y0 = inpaintClampInt(ty-searchRadius, half, rows-1-half)
		y1 = inpaintClampInt(ty+searchRadius, half, rows-1-half)
		x0 = inpaintClampInt(tx-searchRadius, half, cols-1-half)
		x1 = inpaintClampInt(tx+searchRadius, half, cols-1-half)
	}
	best := math.Inf(1)
	by, bx = ty, tx
	for cy := y0; cy <= y1; cy++ {
		for cx := x0; cx <= x1; cx++ {
			// Candidate patch must be entirely filled.
			ok := true
			for dy := -half; dy <= half && ok; dy++ {
				for dx := -half; dx <= half; dx++ {
					if !filled[(cy+dy)*cols+(cx+dx)] {
						ok = false
						break
					}
				}
			}
			if !ok {
				continue
			}
			var ssd float64
			for dy := -half; dy <= half; dy++ {
				for dx := -half; dx <= half; dx++ {
					py, px := ty+dy, tx+dx
					if py < 0 || py >= rows || px < 0 || px >= cols {
						continue
					}
					if !filled[py*cols+px] {
						continue // only compare known target pixels
					}
					for c := 0; c < ch; c++ {
						d := float64(m.At(py, px, c)) - float64(m.At(cy+dy, cx+dx, c))
						ssd += d * d
					}
				}
				if ssd >= best {
					break
				}
			}
			if ssd < best {
				best = ssd
				by, bx = cy, cx
			}
		}
	}
	return by, bx
}
