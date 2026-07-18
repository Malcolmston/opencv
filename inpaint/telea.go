package inpaint

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// InpaintTelea reconstructs the pixels of img selected by mask using Telea's
// (2004) Fast Marching Method inpainting. Unknown pixels are visited in order of
// increasing distance from the region boundary (the eikonal ordering of
// [FastMarcher]); each is estimated as a weighted average of the already-known
// pixels within radius, where each contribution is corrected by the local image
// gradient and weighted by three factors — a directional term (favouring
// neighbours along the boundary normal), a geometric term (nearer neighbours
// dominate) and a level term (neighbours at a similar distance dominate). This
// propagates smooth gradients and thin structures into the hole.
//
// img may be single- or three-channel; mask must match its size (true = fill).
// radius is the neighbourhood radius in pixels (minimum 1). The original img is
// not modified — a filled clone is returned. A uniform surround is reproduced
// exactly; a linear gradient is reproduced up to discretisation.
func InpaintTelea(img *cv.Mat, mask *Mask, radius int) *cv.Mat {
	inpaintRequireImage(img, "InpaintTelea")
	inpaintRequireMaskMatch(img, mask, "InpaintTelea")
	if radius < 1 {
		radius = 1
	}
	out := img.Clone()

	// known[i] tracks pixels whose value is final. It starts as the complement
	// of the mask and grows as the front advances; the FMM visit order is the
	// exact fill order.
	rows, cols := img.Rows, img.Cols
	known := make([]bool, rows*cols)
	for i, v := range mask.Data {
		known[i] = !v
	}

	fm := NewFastMarcher(mask)
	var order [][2]int
	t := fm.Solve(func(y, x int) {
		order = append(order, [2]int{y, x})
	})

	for _, p := range order {
		y, x := p[0], p[1]
		inpaintTeleaPoint(out, known, t, y, x, radius)
		known[y*cols+x] = true
	}
	return out
}

// inpaintTeleaPoint sets pixel (y, x) to the gradient-corrected, triple-weighted
// average of the known pixels within radius, then leaves marking it known to the
// caller.
func inpaintTeleaPoint(m *cv.Mat, known []bool, t []float64, y, x, radius int) {
	rows, cols, ch := m.Rows, m.Cols, m.Channels

	// Boundary normal direction: gradient of the arrival-time field at (y, x).
	nx := inpaintGradT(t, cols, rows, y, x, 1)
	ny := inpaintGradT(t, cols, rows, y, x, 0)

	sum := make([]float64, ch)
	var wsum float64
	tp := t[y*cols+x]
	for dy := -radius; dy <= radius; dy++ {
		for dx := -radius; dx <= radius; dx++ {
			if dx == 0 && dy == 0 {
				continue
			}
			qy, qx := y+dy, x+dx
			if qy < 0 || qy >= rows || qx < 0 || qx >= cols {
				continue
			}
			if !known[qy*cols+qx] {
				continue
			}
			fdx := float64(dx)
			fdy := float64(dy)
			dist2 := fdx*fdx + fdy*fdy
			if dist2 > float64(radius*radius) {
				continue
			}
			dist := math.Sqrt(dist2)

			// Directional term: alignment of (p-q) with the normal.
			dir := math.Abs((-fdy*ny + -fdx*nx) / dist)
			if dir < 1e-6 {
				dir = 1e-6
			}
			// Geometric term: inverse squared distance.
			geom := 1.0 / dist2
			// Level term: closeness in arrival time.
			lev := 1.0 / (1.0 + math.Abs(tp-t[qy*cols+qx]))
			w := dir * geom * lev

			for c := 0; c < ch; c++ {
				// Gradient of the (partially known) image at q, one-sided over
				// known neighbours, used to extrapolate q's value to p.
				gx, gy := inpaintGradImg(m, known, qy, qx, c)
				val := float64(m.At(qy, qx, c)) + gx*(-fdx) + gy*(-fdy)
				sum[c] += w * val
			}
			wsum += w
		}
	}
	if wsum <= 0 {
		return
	}
	for c := 0; c < ch; c++ {
		m.Set(y, x, c, inpaintClampU8(sum[c]/wsum))
	}
}

// inpaintGradT returns the central finite difference of the arrival-time field
// along axis (0 = y, 1 = x) at (y, x), used as the boundary normal.
func inpaintGradT(t []float64, cols, rows, y, x, axis int) float64 {
	if axis == 1 {
		xm := inpaintClampInt(x-1, 0, cols-1)
		xp := inpaintClampInt(x+1, 0, cols-1)
		return (t[y*cols+xp] - t[y*cols+xm]) / 2
	}
	ym := inpaintClampInt(y-1, 0, rows-1)
	yp := inpaintClampInt(y+1, 0, rows-1)
	return (t[yp*cols+x] - t[ym*cols+x]) / 2
}

// inpaintGradImg returns the image gradient (gx, gy) of channel c at (y, x),
// using only known neighbours (one-sided where a side is unknown).
func inpaintGradImg(m *cv.Mat, known []bool, y, x, c int) (gx, gy float64) {
	cols := m.Cols
	leftK := x-1 >= 0 && known[y*cols+(x-1)]
	rightK := x+1 < m.Cols && known[y*cols+(x+1)]
	upK := y-1 >= 0 && known[(y-1)*cols+x]
	downK := y+1 < m.Rows && known[(y+1)*cols+x]
	cval := float64(m.At(y, x, c))
	switch {
	case leftK && rightK:
		gx = (float64(m.At(y, x+1, c)) - float64(m.At(y, x-1, c))) / 2
	case rightK:
		gx = float64(m.At(y, x+1, c)) - cval
	case leftK:
		gx = cval - float64(m.At(y, x-1, c))
	}
	switch {
	case upK && downK:
		gy = (float64(m.At(y+1, x, c)) - float64(m.At(y-1, x, c))) / 2
	case downK:
		gy = float64(m.At(y+1, x, c)) - cval
	case upK:
		gy = cval - float64(m.At(y-1, x, c))
	}
	return gx, gy
}
