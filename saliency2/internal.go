package saliency2

import (
	"math"
	"math/cmplx"

	cv "github.com/malcolmston/opencv"
)

// saliency2ClampInt clamps v to the inclusive range [lo, hi].
func saliency2ClampInt(v, lo, hi int) int {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

// saliency2NextPow2 returns the smallest power of two that is >= n. For n <= 1
// it returns 1.
func saliency2NextPow2(n int) int {
	if n <= 1 {
		return 1
	}
	p := 1
	for p < n {
		p <<= 1
	}
	return p
}

// saliency2GrayFloat returns a single-channel float grid of img's luminance,
// with samples in [0,255]. Multi-channel input is converted with the parent
// package's RGB-to-gray weights; single-channel input is copied. It panics if
// img is nil or empty.
func saliency2GrayFloat(img *cv.Mat) *SaliencyMap {
	if img == nil || img.Empty() {
		panic("saliency2: nil or empty input image")
	}
	g := img
	if img.Channels != 1 {
		g = cv.CvtColor(img, cv.ColorRGB2Gray)
	}
	out := NewSaliencyMap(g.Rows, g.Cols)
	n := g.Rows * g.Cols
	ch := g.Channels
	for i := 0; i < n; i++ {
		out.Data[i] = float64(g.Data[i*ch])
	}
	return out
}

// saliency2RGBFloat returns the red, green and blue planes of img as float
// grids in [0,255]. Single-channel input is replicated across the three planes.
func saliency2RGBFloat(img *cv.Mat) (r, g, b *SaliencyMap) {
	if img == nil || img.Empty() {
		panic("saliency2: nil or empty input image")
	}
	r = NewSaliencyMap(img.Rows, img.Cols)
	g = NewSaliencyMap(img.Rows, img.Cols)
	b = NewSaliencyMap(img.Rows, img.Cols)
	n := img.Rows * img.Cols
	ch := img.Channels
	if ch >= 3 {
		for i := 0; i < n; i++ {
			r.Data[i] = float64(img.Data[i*ch+0])
			g.Data[i] = float64(img.Data[i*ch+1])
			b.Data[i] = float64(img.Data[i*ch+2])
		}
	} else {
		for i := 0; i < n; i++ {
			v := float64(img.Data[i*ch])
			r.Data[i], g.Data[i], b.Data[i] = v, v, v
		}
	}
	return r, g, b
}

// saliency2LabFloat returns the L*, a* and b* planes of img as float grids,
// using the parent package's 8-bit CIE L*a*b* encoding (all in [0,255]).
// Single-channel input is promoted to RGB first.
func saliency2LabFloat(img *cv.Mat) (l, a, b *SaliencyMap) {
	if img == nil || img.Empty() {
		panic("saliency2: nil or empty input image")
	}
	src := img
	if img.Channels == 1 {
		src = cv.CvtColor(img, cv.ColorGray2RGB)
	}
	lab := cv.CvtColor(src, cv.ColorRGB2Lab)
	l = NewSaliencyMap(lab.Rows, lab.Cols)
	a = NewSaliencyMap(lab.Rows, lab.Cols)
	b = NewSaliencyMap(lab.Rows, lab.Cols)
	n := lab.Rows * lab.Cols
	for i := 0; i < n; i++ {
		l.Data[i] = float64(lab.Data[i*3+0])
		a.Data[i] = float64(lab.Data[i*3+1])
		b.Data[i] = float64(lab.Data[i*3+2])
	}
	return l, a, b
}

// saliency2GaussKernel returns a normalised 1-D Gaussian kernel of the given
// odd size. A non-positive sigma is derived from the size the same way OpenCV
// does.
func saliency2GaussKernel(ksize int, sigma float64) []float64 {
	if ksize < 1 {
		ksize = 1
	}
	if ksize%2 == 0 {
		ksize++
	}
	if sigma <= 0 {
		sigma = 0.3*(float64(ksize-1)*0.5-1) + 0.8
	}
	r := ksize / 2
	k := make([]float64, ksize)
	var sum float64
	for i := -r; i <= r; i++ {
		v := math.Exp(-float64(i*i) / (2 * sigma * sigma))
		k[i+r] = v
		sum += v
	}
	for i := range k {
		k[i] /= sum
	}
	return k
}

// saliency2GaussianBlurMap returns a Gaussian-blurred copy of m using a
// separable kernel of the given odd size and standard deviation, replicating
// the border sample.
func saliency2GaussianBlurMap(m *SaliencyMap, ksize int, sigma float64) *SaliencyMap {
	k := saliency2GaussKernel(ksize, sigma)
	r := len(k) / 2
	rows, cols := m.Rows, m.Cols
	tmp := NewSaliencyMap(rows, cols)
	for y := 0; y < rows; y++ {
		base := y * cols
		for x := 0; x < cols; x++ {
			var s float64
			for t := -r; t <= r; t++ {
				xx := saliency2ClampInt(x+t, 0, cols-1)
				s += k[t+r] * m.Data[base+xx]
			}
			tmp.Data[base+x] = s
		}
	}
	out := NewSaliencyMap(rows, cols)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			var s float64
			for t := -r; t <= r; t++ {
				yy := saliency2ClampInt(y+t, 0, rows-1)
				s += k[t+r] * tmp.Data[yy*cols+x]
			}
			out.Data[y*cols+x] = s
		}
	}
	return out
}

// saliency2Integral returns the summed-area table of m with dimensions
// (Rows+1)x(Cols+1); element (y,x) holds the sum of all samples above and to
// the left of pixel (y-1, x-1).
func saliency2Integral(m *SaliencyMap) []float64 {
	w := m.Cols + 1
	ii := make([]float64, (m.Rows+1)*w)
	for y := 0; y < m.Rows; y++ {
		var rowsum float64
		for x := 0; x < m.Cols; x++ {
			rowsum += m.Data[y*m.Cols+x]
			ii[(y+1)*w+(x+1)] = ii[y*w+(x+1)] + rowsum
		}
	}
	return ii
}

// saliency2BoxMean returns the mean of the samples inside the axis-aligned
// window of radius r centred on (y, x), using the summed-area table ii for the
// grid of the given size. The window is clamped to the grid bounds.
func saliency2BoxMean(ii []float64, rows, cols, y, x, r int) float64 {
	w := cols + 1
	y0 := saliency2ClampInt(y-r, 0, rows)
	y1 := saliency2ClampInt(y+r+1, 0, rows)
	x0 := saliency2ClampInt(x-r, 0, cols)
	x1 := saliency2ClampInt(x+r+1, 0, cols)
	area := float64((y1 - y0) * (x1 - x0))
	if area == 0 {
		return 0
	}
	s := ii[y1*w+x1] - ii[y0*w+x1] - ii[y1*w+x0] + ii[y0*w+x0]
	return s / area
}

// saliency2BoxBlurMap returns a box-averaged copy of m with the given window
// radius, using an integral image so the cost is independent of radius.
func saliency2BoxBlurMap(m *SaliencyMap, radius int) *SaliencyMap {
	if radius < 1 {
		return m.Clone()
	}
	ii := saliency2Integral(m)
	out := NewSaliencyMap(m.Rows, m.Cols)
	for y := 0; y < m.Rows; y++ {
		for x := 0; x < m.Cols; x++ {
			out.Data[y*m.Cols+x] = saliency2BoxMean(ii, m.Rows, m.Cols, y, x, radius)
		}
	}
	return out
}

// saliency2ResizeMap returns a bilinearly resampled copy of m at the requested
// dimensions. Identical dimensions yield a clone.
func saliency2ResizeMap(m *SaliencyMap, newRows, newCols int) *SaliencyMap {
	if newRows < 1 {
		newRows = 1
	}
	if newCols < 1 {
		newCols = 1
	}
	if newRows == m.Rows && newCols == m.Cols {
		return m.Clone()
	}
	out := NewSaliencyMap(newRows, newCols)
	sy := float64(m.Rows) / float64(newRows)
	sx := float64(m.Cols) / float64(newCols)
	for y := 0; y < newRows; y++ {
		fy := (float64(y)+0.5)*sy - 0.5
		y0 := int(math.Floor(fy))
		ty := fy - float64(y0)
		y0c := saliency2ClampInt(y0, 0, m.Rows-1)
		y1c := saliency2ClampInt(y0+1, 0, m.Rows-1)
		for x := 0; x < newCols; x++ {
			fx := (float64(x)+0.5)*sx - 0.5
			x0 := int(math.Floor(fx))
			tx := fx - float64(x0)
			x0c := saliency2ClampInt(x0, 0, m.Cols-1)
			x1c := saliency2ClampInt(x0+1, 0, m.Cols-1)
			v00 := m.Data[y0c*m.Cols+x0c]
			v01 := m.Data[y0c*m.Cols+x1c]
			v10 := m.Data[y1c*m.Cols+x0c]
			v11 := m.Data[y1c*m.Cols+x1c]
			top := v00 + (v01-v00)*tx
			bot := v10 + (v11-v10)*tx
			out.Data[y*newCols+x] = top + (bot-top)*ty
		}
	}
	return out
}

// saliency2Sobel returns the horizontal and vertical Sobel derivative grids of
// m, replicating the border sample.
func saliency2Sobel(m *SaliencyMap) (gx, gy *SaliencyMap) {
	rows, cols := m.Rows, m.Cols
	gx = NewSaliencyMap(rows, cols)
	gy = NewSaliencyMap(rows, cols)
	at := func(y, x int) float64 {
		return m.Data[saliency2ClampInt(y, 0, rows-1)*cols+saliency2ClampInt(x, 0, cols-1)]
	}
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			tl, tc, tr := at(y-1, x-1), at(y-1, x), at(y-1, x+1)
			ml, _, mr := at(y, x-1), at(y, x), at(y, x+1)
			bl, bc, br := at(y+1, x-1), at(y+1, x), at(y+1, x+1)
			gx.Data[y*cols+x] = (tr + 2*mr + br) - (tl + 2*ml + bl)
			gy.Data[y*cols+x] = (bl + 2*bc + br) - (tl + 2*tc + tr)
		}
	}
	return gx, gy
}

// saliency2GaussPyramid returns a Gaussian pyramid of m: level 0 is m itself
// and each subsequent level is a Gaussian-smoothed half-resolution copy of the
// previous one. The pyramid stops once the smaller dimension would fall below
// minSize, and never exceeds maxLevels levels.
func saliency2GaussPyramid(m *SaliencyMap, maxLevels, minSize int) []*SaliencyMap {
	pyr := []*SaliencyMap{m.Clone()}
	cur := m
	for len(pyr) < maxLevels {
		nr, nc := cur.Rows/2, cur.Cols/2
		if nr < minSize || nc < minSize {
			break
		}
		blurred := saliency2GaussianBlurMap(cur, 5, 1.0)
		down := saliency2ResizeMap(blurred, nr, nc)
		pyr = append(pyr, down)
		cur = down
	}
	return pyr
}

// saliency2FFT1D performs an in-place radix-2 Cooley-Tukey FFT (or inverse FFT)
// on a, whose length must be a power of two. The inverse transform divides by
// the length.
func saliency2FFT1D(a []complex128, inverse bool) {
	n := len(a)
	if n <= 1 {
		return
	}
	for i, j := 1, 0; i < n; i++ {
		bit := n >> 1
		for ; j&bit != 0; bit >>= 1 {
			j ^= bit
		}
		j ^= bit
		if i < j {
			a[i], a[j] = a[j], a[i]
		}
	}
	for length := 2; length <= n; length <<= 1 {
		ang := 2 * math.Pi / float64(length)
		if !inverse {
			ang = -ang
		}
		wlen := cmplx.Rect(1, ang)
		for i := 0; i < n; i += length {
			w := complex(1, 0)
			half := length / 2
			for k := 0; k < half; k++ {
				u := a[i+k]
				v := a[i+k+half] * w
				a[i+k] = u + v
				a[i+k+half] = u - v
				w *= wlen
			}
		}
	}
	if inverse {
		inv := complex(1/float64(n), 0)
		for i := range a {
			a[i] *= inv
		}
	}
}

// saliency2FFT2D performs an in-place 2-D FFT (or inverse) on a field of h rows
// by w columns stored row-major. Both dimensions must be powers of two.
func saliency2FFT2D(field []complex128, h, w int, inverse bool) {
	row := make([]complex128, w)
	for y := 0; y < h; y++ {
		copy(row, field[y*w:(y+1)*w])
		saliency2FFT1D(row, inverse)
		copy(field[y*w:(y+1)*w], row)
	}
	col := make([]complex128, h)
	for x := 0; x < w; x++ {
		for y := 0; y < h; y++ {
			col[y] = field[y*w+x]
		}
		saliency2FFT1D(col, inverse)
		for y := 0; y < h; y++ {
			field[y*w+x] = col[y]
		}
	}
}

// saliency2PerimeterMean returns the mean normed-gradient value sampled along
// the border of the axis-aligned window with top-left (x0, y0) and size hxw in
// the grid ng.
func saliency2PerimeterMean(ng *SaliencyMap, y0, x0, h, w int) float64 {
	rows, cols := ng.Rows, ng.Cols
	var sum float64
	var count int
	add := func(y, x int) {
		if y < 0 || y >= rows || x < 0 || x >= cols {
			return
		}
		sum += ng.Data[y*cols+x]
		count++
	}
	yb := y0 + h - 1
	xr := x0 + w - 1
	for x := x0; x <= xr; x++ {
		add(y0, x)
		add(yb, x)
	}
	for y := y0 + 1; y < yb; y++ {
		add(y, x0)
		add(y, xr)
	}
	if count == 0 {
		return 0
	}
	return sum / float64(count)
}
