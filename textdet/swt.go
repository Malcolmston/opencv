package textdet

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// SWTPolarity selects whether the Stroke-Width Transform seeks dark strokes on a
// light background or the reverse, which fixes the direction in which rays are
// cast from each edge pixel.
type SWTPolarity int

const (
	// SWTDarkOnLight casts rays along the negative image gradient, the usual
	// case of dark text on a light page.
	SWTDarkOnLight SWTPolarity = iota
	// SWTLightOnDark casts rays along the positive image gradient, for light
	// text on a dark background.
	SWTLightOnDark
)

// SWTOptions configures [StrokeWidthTransform].
type SWTOptions struct {
	// Polarity chooses the stroke/background contrast direction.
	Polarity SWTPolarity
	// LowThreshold and HighThreshold are the hysteresis gradient thresholds of
	// the internal Canny edge detector.
	LowThreshold, HighThreshold float64
	// MaxStrokeWidth caps the ray length in pixels. A value <= 0 uses the image
	// diagonal, so no valid stroke is missed.
	MaxStrokeWidth int
	// AngleTolerance is the maximum angular deviation, in radians, allowed
	// between a start edge's gradient and the reverse of the opposite edge's
	// gradient for a ray to be accepted. A value of Pi/2 accepts any opposing
	// hemisphere; smaller values are stricter.
	AngleTolerance float64
}

// DefaultSWTOptions returns options tuned for dark text on a light background:
// Canny thresholds 50/150, no explicit stroke cap, and a Pi/2 angle tolerance.
func DefaultSWTOptions() SWTOptions {
	return SWTOptions{
		Polarity:       SWTDarkOnLight,
		LowThreshold:   50,
		HighThreshold:  150,
		MaxStrokeWidth: 0,
		AngleTolerance: math.Pi / 2,
	}
}

// SWTResult holds the per-pixel stroke widths produced by
// [StrokeWidthTransform]. A value of 0 marks a pixel that no ray crossed
// (background or interior not spanned by a stroke); a positive value is the
// estimated local stroke width in pixels.
type SWTResult struct {
	// Rows is the image height.
	Rows int
	// Cols is the image width.
	Cols int
	// Width is the row-major stroke-width map.
	Width []float64
}

// At returns the stroke width at column x, row y, or 0 if the coordinates are
// out of range.
func (r *SWTResult) At(x, y int) float64 {
	if x < 0 || y < 0 || x >= r.Cols || y >= r.Rows {
		return 0
	}
	return r.Width[y*r.Cols+x]
}

// Max returns the largest stroke width in the map (0 if the map is empty of
// strokes).
func (r *SWTResult) Max() float64 {
	m := 0.0
	for _, v := range r.Width {
		if v > m {
			m = v
		}
	}
	return m
}

// ToMat renders the stroke-width map as a single-channel [cv.Mat] for
// visualization, linearly scaling widths in (0, Max] to grey levels [1,255];
// pixels with no stroke stay 0.
func (r *SWTResult) ToMat() *cv.Mat {
	dst := cv.NewMat(r.Rows, r.Cols, 1)
	max := r.Max()
	if max <= 0 {
		return dst
	}
	for i, v := range r.Width {
		if v > 0 {
			g := 1 + int(math.Round(v/max*254))
			if g > 255 {
				g = 255
			}
			dst.Data[i] = uint8(g)
		}
	}
	return dst
}

// StrokeWidthTransform computes the Stroke-Width Transform of src. It detects
// edges with Canny, computes the image gradient, and from every edge pixel casts
// a ray (along the gradient direction chosen by opts.Polarity) until it reaches
// an opposing edge; the distance between the two edges is the local stroke
// width, assigned as the minimum over all rays crossing each pixel. A second
// pass replaces each ray's samples by the minimum of themselves and the ray
// median, which corrects widths at corners and junctions. It returns [ErrEmpty]
// for an empty image.
func StrokeWidthTransform(src *cv.Mat, opts SWTOptions) (*SWTResult, error) {
	gray, rows, cols, err := textdetGray(src)
	if err != nil {
		return nil, err
	}
	grayMat := cv.NewMat(rows, cols, 1)
	copy(grayMat.Data, gray)
	edgeMat := cv.Canny(grayMat, opts.LowThreshold, opts.HighThreshold)
	gx, gy := textdetSobel(gray, rows, cols)

	edges := make([]bool, rows*cols)
	for i := range edges {
		if edgeMat.Data[i] != 0 {
			edges[i] = true
		}
	}

	maxStroke := opts.MaxStrokeWidth
	if maxStroke <= 0 {
		maxStroke = int(math.Ceil(math.Hypot(float64(rows), float64(cols))))
	}
	sign := -1.0
	if opts.Polarity == SWTLightOnDark {
		sign = 1.0
	}
	tolCos := math.Cos(opts.AngleTolerance)

	width := make([]float64, rows*cols)
	for i := range width {
		width[i] = math.Inf(1)
	}
	type rayRecord struct {
		pixels []int
		w      float64
	}
	var rays []rayRecord

	for py := 0; py < rows; py++ {
		for px := 0; px < cols; px++ {
			pidx := py*cols + px
			if !edges[pidx] {
				continue
			}
			gpx, gpy := gx[pidx], gy[pidx]
			mag := math.Hypot(gpx, gpy)
			if mag < 1e-9 {
				continue
			}
			dx := sign * gpx / mag
			dy := sign * gpy / mag
			fx := float64(px) + 0.5
			fy := float64(py) + 0.5
			curX, curY := px, py
			pixels := []int{pidx}
			found := false
			var w float64
			for step := 0; step < maxStroke; step++ {
				fx += dx
				fy += dy
				nx := int(math.Floor(fx))
				ny := int(math.Floor(fy))
				if nx == curX && ny == curY {
					continue
				}
				curX, curY = nx, ny
				if nx < 0 || ny < 0 || nx >= cols || ny >= rows {
					break
				}
				nidx := ny*cols + nx
				pixels = append(pixels, nidx)
				if edges[nidx] {
					qgx, qgy := gx[nidx], gy[nidx]
					qmag := math.Hypot(qgx, qgy)
					if qmag < 1e-9 {
						break
					}
					dot := (gpx*qgx + gpy*qgy) / (mag * qmag)
					// Accept when the two gradients are roughly opposite.
					if dot <= -tolCos {
						w = math.Hypot(float64(nx-px), float64(ny-py))
						found = true
					}
					break
				}
			}
			if !found || w <= 0 {
				continue
			}
			for _, idx := range pixels {
				if w < width[idx] {
					width[idx] = w
				}
			}
			rays = append(rays, rayRecord{pixels: pixels, w: w})
		}
	}

	// Second pass: clamp each ray to its own median to fix junctions.
	for _, r := range rays {
		vals := make([]float64, 0, len(r.pixels))
		for _, idx := range r.pixels {
			if !math.IsInf(width[idx], 1) {
				vals = append(vals, width[idx])
			}
		}
		med := textdetMedianFloat(vals)
		if med <= 0 {
			continue
		}
		for _, idx := range r.pixels {
			if med < width[idx] {
				width[idx] = med
			}
		}
	}

	out := make([]float64, rows*cols)
	for i, v := range width {
		if !math.IsInf(v, 1) {
			out[i] = v
		}
	}
	return &SWTResult{Rows: rows, Cols: cols, Width: out}, nil
}

// SWTLetter is a candidate glyph found by grouping neighbouring pixels of
// similar stroke width in an [SWTResult].
type SWTLetter struct {
	// Bounds is the upright bounding box of the candidate.
	Bounds cv.Rect
	// Area is the number of stroke pixels in the candidate.
	Area int
	// MeanStrokeWidth is the mean stroke width over the candidate's pixels.
	MeanStrokeWidth float64
	// StrokeWidthStd is the standard deviation of the stroke width, small for
	// genuine letters whose strokes have consistent thickness.
	StrokeWidthStd float64
}

// SWTLetters groups the stroke pixels of res into connected candidates. Two
// 8-adjacent stroke pixels join the same candidate when the ratio of their
// stroke widths does not exceed ratio (a typical value is 3.0). Candidates with
// fewer than minArea pixels are discarded. The result is ordered top-to-bottom
// then left-to-right. It returns [ErrInvalidArgument] if ratio < 1 or the result
// map is empty.
func SWTLetters(res *SWTResult, ratio float64, minArea int) ([]SWTLetter, error) {
	if ratio < 1 || res == nil || len(res.Width) == 0 {
		return nil, ErrInvalidArgument
	}
	rows, cols := res.Rows, res.Cols
	n := rows * cols
	uf := &textdetUnionFind{parent: make([]int, n)}
	for i := range uf.parent {
		uf.parent[i] = i
	}
	stroke := func(i int) bool { return res.Width[i] > 0 }
	similar := func(a, b float64) bool {
		hi, lo := a, b
		if lo > hi {
			hi, lo = lo, hi
		}
		if lo <= 0 {
			return false
		}
		return hi/lo <= ratio
	}
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			idx := y*cols + x
			if !stroke(idx) {
				continue
			}
			// Compare with the already-scanned left, up, up-left, up-right.
			if x > 0 && stroke(idx-1) && similar(res.Width[idx], res.Width[idx-1]) {
				uf.union(idx, idx-1)
			}
			if y > 0 && stroke(idx-cols) && similar(res.Width[idx], res.Width[idx-cols]) {
				uf.union(idx, idx-cols)
			}
			if y > 0 && x > 0 && stroke(idx-cols-1) && similar(res.Width[idx], res.Width[idx-cols-1]) {
				uf.union(idx, idx-cols-1)
			}
			if y > 0 && x < cols-1 && stroke(idx-cols+1) && similar(res.Width[idx], res.Width[idx-cols+1]) {
				uf.union(idx, idx-cols+1)
			}
		}
	}

	type acc struct {
		area                   int
		minX, minY, maxX, maxY int
		sumW, sumW2            float64
	}
	groups := make(map[int]*acc)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			idx := y*cols + x
			if !stroke(idx) {
				continue
			}
			r := uf.find(idx)
			a := groups[r]
			if a == nil {
				a = &acc{minX: cols, minY: rows, maxX: -1, maxY: -1}
				groups[r] = a
			}
			a.area++
			if x < a.minX {
				a.minX = x
			}
			if x > a.maxX {
				a.maxX = x
			}
			if y < a.minY {
				a.minY = y
			}
			if y > a.maxY {
				a.maxY = y
			}
			w := res.Width[idx]
			a.sumW += w
			a.sumW2 += w * w
		}
	}

	var letters []SWTLetter
	for _, a := range groups {
		if a.area < minArea {
			continue
		}
		mean := a.sumW / float64(a.area)
		variance := a.sumW2/float64(a.area) - mean*mean
		if variance < 0 {
			variance = 0
		}
		letters = append(letters, SWTLetter{
			Bounds:          cv.Rect{X: a.minX, Y: a.minY, Width: a.maxX - a.minX + 1, Height: a.maxY - a.minY + 1},
			Area:            a.area,
			MeanStrokeWidth: mean,
			StrokeWidthStd:  math.Sqrt(variance),
		})
	}
	sortLetters(letters)
	return letters, nil
}

// sortLetters orders letters top-to-bottom then left-to-right, deterministically.
func sortLetters(ls []SWTLetter) {
	for i := 1; i < len(ls); i++ {
		v := ls[i]
		j := i - 1
		for j >= 0 && letterAfter(ls[j], v) {
			ls[j+1] = ls[j]
			j--
		}
		ls[j+1] = v
	}
}

// letterAfter reports whether a should sort after b.
func letterAfter(a, b SWTLetter) bool {
	if a.Bounds.Y != b.Bounds.Y {
		return a.Bounds.Y > b.Bounds.Y
	}
	return a.Bounds.X > b.Bounds.X
}
