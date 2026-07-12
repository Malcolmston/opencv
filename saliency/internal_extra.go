package saliency

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// clonePlane returns an independent copy of p.
func clonePlane(p *plane) *plane {
	q := newPlane(p.rows, p.cols)
	copy(q.data, p.data)
	return q
}

// mean returns the arithmetic mean of the plane's samples (0 for an empty
// plane).
func (p *plane) mean() float64 {
	if len(p.data) == 0 {
		return 0
	}
	var s float64
	for _, v := range p.data {
		s += v
	}
	return s / float64(len(p.data))
}

// addScaled adds w*q into p in place. The two planes must be the same size.
func (p *plane) addScaled(q *plane, w float64) {
	for i, v := range q.data {
		p.data[i] += w * v
	}
}

// absDiffPlanes returns |a-b| sample-wise; the planes must be the same size.
func absDiffPlanes(a, b *plane) *plane {
	out := newPlane(a.rows, a.cols)
	for i := range a.data {
		out.data[i] = math.Abs(a.data[i] - b.data[i])
	}
	return out
}

// rgbPlanes splits img into three float planes holding its red, green and blue
// samples, each in [0,255]. Single-channel (or other) input yields three
// identical grayscale planes. It panics if img is nil or empty.
func rgbPlanes(img *cv.Mat) (r, g, b *plane) {
	if img == nil || img.Empty() {
		panic("saliency: input image is empty")
	}
	if img.Channels != 3 {
		gray := grayPlane(img)
		return gray, clonePlane(gray), clonePlane(gray)
	}
	rows, cols := img.Rows, img.Cols
	r = newPlane(rows, cols)
	g = newPlane(rows, cols)
	b = newPlane(rows, cols)
	for p := 0; p < img.Total(); p++ {
		base := p * 3
		r.data[p] = float64(img.Data[base+0])
		g.data[p] = float64(img.Data[base+1])
		b.data[p] = float64(img.Data[base+2])
	}
	return r, g, b
}

// labPlanes returns the CIE L*a*b* channel planes of img using the root
// package's 8-bit Lab encoding (each channel in [0,255]). Single-channel input
// is promoted to gray RGB first. It panics if img is nil or empty.
func labPlanes(img *cv.Mat) (l, a, b *plane) {
	if img == nil || img.Empty() {
		panic("saliency: input image is empty")
	}
	var rgb *cv.Mat
	switch img.Channels {
	case 3:
		rgb = img
	case 1:
		rgb = cv.CvtColor(img, cv.ColorGray2RGB)
	default:
		gray := cv.NewMat(img.Rows, img.Cols, 1)
		for p := 0; p < img.Total(); p++ {
			gray.Data[p] = img.Data[p*img.Channels]
		}
		rgb = cv.CvtColor(gray, cv.ColorGray2RGB)
	}
	lab := cv.CvtColor(rgb, cv.ColorRGB2Lab)
	rows, cols := lab.Rows, lab.Cols
	l = newPlane(rows, cols)
	a = newPlane(rows, cols)
	b = newPlane(rows, cols)
	for p := 0; p < lab.Total(); p++ {
		base := p * 3
		l.data[p] = float64(lab.Data[base+0])
		a.data[p] = float64(lab.Data[base+1])
		b.data[p] = float64(lab.Data[base+2])
	}
	return l, a, b
}

// conv3x3 convolves p with the 3×3 kernel k (row-major, 9 elements), replicating
// borders, and returns the result.
func conv3x3(p *plane, k [9]float64) *plane {
	out := newPlane(p.rows, p.cols)
	for y := 0; y < p.rows; y++ {
		for x := 0; x < p.cols; x++ {
			var sum float64
			idx := 0
			for dy := -1; dy <= 1; dy++ {
				for dx := -1; dx <= 1; dx++ {
					sum += k[idx] * p.atReplicate(y+dy, x+dx)
					idx++
				}
			}
			out.data[y*p.cols+x] = sum
		}
	}
	return out
}

// gaussPyramid returns a dyadic Gaussian pyramid of p: element 0 is p itself
// and each subsequent level is the previous one blurred and halved in each
// dimension. The pyramid stops early if a level would fall below 2×2 or after
// levels entries.
func gaussPyramid(p *plane, levels int) []*plane {
	pyr := []*plane{clonePlane(p)}
	cur := pyr[0]
	for i := 1; i < levels; i++ {
		nr := (cur.rows + 1) / 2
		nc := (cur.cols + 1) / 2
		if nr < 2 || nc < 2 {
			break
		}
		blurred := gaussianBlurPlane(cur, 5, 1.0)
		cur = resizePlane(blurred, nr, nc)
		pyr = append(pyr, cur)
	}
	return pyr
}

// floodBorderMask returns, for the binary plane described by mask (true/false
// per pixel), a boolean slice marking every pixel that equals target and lies
// in a 4-connected component touching the image border.
func floodBorderMask(mask []bool, rows, cols int, target bool) []bool {
	touch := make([]bool, rows*cols)
	queue := make([]int, 0, rows+cols)
	push := func(y, x int) {
		if y < 0 || y >= rows || x < 0 || x >= cols {
			return
		}
		i := y*cols + x
		if touch[i] || mask[i] != target {
			return
		}
		touch[i] = true
		queue = append(queue, i)
	}
	for x := 0; x < cols; x++ {
		push(0, x)
		push(rows-1, x)
	}
	for y := 0; y < rows; y++ {
		push(y, 0)
		push(y, cols-1)
	}
	for len(queue) > 0 {
		i := queue[len(queue)-1]
		queue = queue[:len(queue)-1]
		y, x := i/cols, i%cols
		push(y-1, x)
		push(y+1, x)
		push(y, x-1)
		push(y, x+1)
	}
	return touch
}
