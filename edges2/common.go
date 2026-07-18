package edges2

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// edges2RequireGray panics unless m is a usable single-channel matrix.
func edges2RequireGray(m *cv.Mat, op string) {
	if m == nil || m.Empty() {
		panic("edges2: " + op + ": empty source matrix")
	}
	if m.Channels != 1 {
		panic("edges2: " + op + ": operation requires a single-channel matrix")
	}
}

// edges2Clamp clamps i to the range [0, n-1], implementing edge replication.
func edges2Clamp(i, n int) int {
	if i < 0 {
		return 0
	}
	if i >= n {
		return n - 1
	}
	return i
}

// edges2ClampU8 rounds v to the nearest integer and clamps it to [0,255].
func edges2ClampU8(v float64) uint8 {
	v = math.Round(v)
	if v <= 0 {
		return 0
	}
	if v >= 255 {
		return 255
	}
	return uint8(v)
}

// edges2Sample returns the single-channel sample of m at (y, x) as a float,
// clamping out-of-range coordinates to the nearest edge.
func edges2Sample(m *cv.Mat, y, x int) float64 {
	y = edges2Clamp(y, m.Rows)
	x = edges2Clamp(x, m.Cols)
	return float64(m.Data[y*m.Cols+x])
}

// edges2GaussianKsize returns an odd Gaussian window size covering three
// standard deviations on each side of the centre, never smaller than 3.
func edges2GaussianKsize(sigma float64) int {
	r := int(math.Ceil(3 * sigma))
	if r < 1 {
		r = 1
	}
	return 2*r + 1
}

// edges2Blur returns a Gaussian-smoothed copy of src using the parent
// package's separable Gaussian, choosing a window size from sigma.
func edges2Blur(src *cv.Mat, sigma float64) *cv.Mat {
	return cv.GaussianBlur(src, edges2GaussianKsize(sigma), sigma)
}

// edges2Convolve correlates src with a small 2-D kernel and returns the signed
// response as a FloatGrid, replicating the border.
func edges2Convolve(src *cv.Mat, k [][]float64) *FloatGrid {
	rows, cols := src.Rows, src.Cols
	kh := len(k)
	kw := len(k[0])
	ah := kh / 2
	aw := kw / 2
	out := NewFloatGrid(rows, cols)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			var s float64
			for j := 0; j < kh; j++ {
				for i := 0; i < kw; i++ {
					s += k[j][i] * edges2Sample(src, y+j-ah, x+i-aw)
				}
			}
			out.Data[y*cols+x] = s
		}
	}
	return out
}

// edges2ConvolveGrid correlates a FloatGrid with a small 2-D kernel and
// returns a new FloatGrid, replicating the border.
func edges2ConvolveGrid(g *FloatGrid, k [][]float64) *FloatGrid {
	rows, cols := g.Rows, g.Cols
	kh := len(k)
	kw := len(k[0])
	ah := kh / 2
	aw := kw / 2
	out := NewFloatGrid(rows, cols)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			var s float64
			for j := 0; j < kh; j++ {
				yy := edges2Clamp(y+j-ah, rows)
				for i := 0; i < kw; i++ {
					xx := edges2Clamp(x+i-aw, cols)
					s += k[j][i] * g.Data[yy*cols+xx]
				}
			}
			out.Data[y*cols+x] = s
		}
	}
	return out
}

// edges2GaussianKernel2D builds a normalised square Gaussian kernel of the
// given odd size and standard deviation.
func edges2GaussianKernel2D(size int, sigma float64) [][]float64 {
	r := size / 2
	k := make([][]float64, size)
	var sum float64
	for j := 0; j < size; j++ {
		k[j] = make([]float64, size)
		for i := 0; i < size; i++ {
			dy := float64(j - r)
			dx := float64(i - r)
			v := math.Exp(-(dx*dx + dy*dy) / (2 * sigma * sigma))
			k[j][i] = v
			sum += v
		}
	}
	for j := range k {
		for i := range k[j] {
			k[j][i] /= sum
		}
	}
	return k
}
