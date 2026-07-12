package quality

import (
	"fmt"
	"math"

	cv "github.com/malcolmston/opencv"
)

// grid is a dense, row-major matrix of float64 samples used for all
// intermediate computation. Working in float avoids the rounding and clamping
// that [cv.Mat]'s 8-bit storage would impose between filtering steps.
type grid struct {
	rows, cols int
	data       []float64
}

// newGrid allocates a zero-filled grid of the given size.
func newGrid(rows, cols int) grid {
	return grid{rows: rows, cols: cols, data: make([]float64, rows*cols)}
}

// idx returns the flat offset of element (y, x).
func (g grid) idx(y, x int) int { return y*g.cols + x }

// at returns element (y, x) with replicate (BORDER_REPLICATE) handling of
// out-of-range coordinates, matching the border policy of the root package's
// convolutions.
func (g grid) at(y, x int) float64 {
	if y < 0 {
		y = 0
	} else if y >= g.rows {
		y = g.rows - 1
	}
	if x < 0 {
		x = 0
	} else if x >= g.cols {
		x = g.cols - 1
	}
	return g.data[y*g.cols+x]
}

// requireComparable panics unless a and b are non-empty and share the same
// dimensions and channel count.
func requireComparable(a, b *cv.Mat, name string) {
	if a.Empty() || b.Empty() {
		panic(fmt.Sprintf("quality: %s given an empty image", name))
	}
	if a.Rows != b.Rows || a.Cols != b.Cols {
		panic(fmt.Sprintf("quality: %s size mismatch %dx%d vs %dx%d",
			name, a.Rows, a.Cols, b.Rows, b.Cols))
	}
	if a.Channels != b.Channels {
		panic(fmt.Sprintf("quality: %s channel mismatch %d vs %d",
			name, a.Channels, b.Channels))
	}
}

// requireImage panics unless m is a non-empty image.
func requireImage(m *cv.Mat, name string) {
	if m.Empty() {
		panic(fmt.Sprintf("quality: %s given an empty image", name))
	}
}

// toGray reduces a Mat to a single-channel float grid in the range [0, 255].
// A one-channel image is copied verbatim; a three-channel image is converted
// with the BT.601 luma weights (the same weights the root package's
// [cv.ColorRGB2Gray] uses); any other channel count falls back to channel 0.
func toGray(m *cv.Mat) grid {
	g := newGrid(m.Rows, m.Cols)
	switch m.Channels {
	case 1:
		for i := range g.data {
			g.data[i] = float64(m.Data[i])
		}
	case 3:
		for p := 0; p < m.Total(); p++ {
			b := p * 3
			r := float64(m.Data[b+0])
			gg := float64(m.Data[b+1])
			bb := float64(m.Data[b+2])
			g.data[p] = 0.299*r + 0.587*gg + 0.114*bb
		}
	default:
		ch := m.Channels
		for p := 0; p < m.Total(); p++ {
			g.data[p] = float64(m.Data[p*ch])
		}
	}
	return g
}

// mul returns the element-wise product a·b. The grids must share a shape.
func mul(a, b grid) grid {
	out := newGrid(a.rows, a.cols)
	for i := range out.data {
		out.data[i] = a.data[i] * b.data[i]
	}
	return out
}

// gaussBlur applies a separable Gaussian blur with replicate borders, reusing
// the root package's [cv.GaussianKernel1D] for the weights.
func gaussBlur(src grid, ksize int, sigma float64) grid {
	k := cv.GaussianKernel1D(ksize, sigma)
	r := ksize / 2
	tmp := newGrid(src.rows, src.cols)
	for y := 0; y < src.rows; y++ {
		for x := 0; x < src.cols; x++ {
			var s float64
			for t := 0; t < ksize; t++ {
				s += k[t] * src.at(y, x+t-r)
			}
			tmp.data[tmp.idx(y, x)] = s
		}
	}
	out := newGrid(src.rows, src.cols)
	for y := 0; y < src.rows; y++ {
		for x := 0; x < src.cols; x++ {
			var s float64
			for t := 0; t < ksize; t++ {
				s += k[t] * tmp.at(y+t-r, x)
			}
			out.data[out.idx(y, x)] = s
		}
	}
	return out
}

// conv3 correlates src with a 3×3 kernel (row-major, nine weights) using
// replicate borders and returns the unclamped float response.
func conv3(src grid, k [9]float64) grid {
	out := newGrid(src.rows, src.cols)
	for y := 0; y < src.rows; y++ {
		for x := 0; x < src.cols; x++ {
			var s float64
			ki := 0
			for dy := -1; dy <= 1; dy++ {
				for dx := -1; dx <= 1; dx++ {
					s += k[ki] * src.at(y+dy, x+dx)
					ki++
				}
			}
			out.data[out.idx(y, x)] = s
		}
	}
	return out
}

// downsample2 halves each dimension by averaging non-overlapping 2×2 blocks,
// the low-pass-and-subsample step of the multi-scale SSIM pyramid. Dimensions
// are floored; a degenerate result is at least 1×1.
func downsample2(src grid) grid {
	r := src.rows / 2
	c := src.cols / 2
	if r < 1 {
		r = 1
	}
	if c < 1 {
		c = 1
	}
	out := newGrid(r, c)
	for y := 0; y < r; y++ {
		for x := 0; x < c; x++ {
			sum := src.at(2*y, 2*x) + src.at(2*y, 2*x+1) +
				src.at(2*y+1, 2*x) + src.at(2*y+1, 2*x+1)
			out.data[out.idx(y, x)] = sum / 4
		}
	}
	return out
}

// meanOf returns the arithmetic mean of xs. It panics on an empty slice.
func meanOf(xs []float64) float64 {
	if len(xs) == 0 {
		panic("quality: mean of empty slice")
	}
	var s float64
	for _, v := range xs {
		s += v
	}
	return s / float64(len(xs))
}

// popStdDev returns the population standard deviation of xs.
func popStdDev(xs []float64) float64 {
	m := meanOf(xs)
	var s float64
	for _, v := range xs {
		d := v - m
		s += d * d
	}
	return math.Sqrt(s / float64(len(xs)))
}

// Sobel 3×3 kernels (row-major) shared by the gradient-based metrics.
var (
	sobelX = [9]float64{-1, 0, 1, -2, 0, 2, -1, 0, 1}
	sobelY = [9]float64{-1, -2, -1, 0, 0, 0, 1, 2, 1}
	// Prewitt kernels scaled by 1/3, as used by the GMSD reference.
	prewittX = [9]float64{
		1.0 / 3, 0, -1.0 / 3,
		1.0 / 3, 0, -1.0 / 3,
		1.0 / 3, 0, -1.0 / 3,
	}
	prewittY = [9]float64{
		1.0 / 3, 1.0 / 3, 1.0 / 3,
		0, 0, 0,
		-1.0 / 3, -1.0 / 3, -1.0 / 3,
	}
	// laplacian4 is the classic 4-neighbour Laplacian stencil.
	laplacian4 = [9]float64{0, 1, 0, 1, -4, 1, 0, 1, 0}
)

// gradientMag returns the per-pixel gradient magnitude sqrt(gx²+gy²) of src
// computed with the supplied 3×3 x/y derivative kernels.
func gradientMag(src grid, kx, ky [9]float64) grid {
	gx := conv3(src, kx)
	gy := conv3(src, ky)
	out := newGrid(src.rows, src.cols)
	for i := range out.data {
		out.data[i] = math.Hypot(gx.data[i], gy.data[i])
	}
	return out
}

// grayMapToMat renders a single-channel float grid to a [cv.Mat], clamping each
// sample into [0, 255] after adding the standard rounding bias.
func grayMapToMat(g grid) *cv.Mat {
	out := cv.NewMat(g.rows, g.cols, 1)
	for i, v := range g.data {
		out.Data[i] = clampToUint8(v + 0.5)
	}
	return out
}

// clampToUint8 rounds toward zero (the caller adds any bias) and clamps to
// [0, 255], mirroring the root package's helper of the same name.
func clampToUint8(v float64) uint8 {
	if v <= 0 {
		return 0
	}
	if v >= 255 {
		return 255
	}
	return uint8(v)
}
