package cv

import (
	"fmt"
	"math"
	"sort"
)

// Kernel is a 2-D convolution kernel stored row-major with Rows*Cols float
// weights. Build one with [NewKernel] or the specialised helpers.
type Kernel struct {
	Rows int
	Cols int
	Data []float64
}

// NewKernel builds a Kernel from row-major data. It panics if len(data) does
// not equal rows*cols or a dimension is not positive.
func NewKernel(rows, cols int, data []float64) Kernel {
	if rows <= 0 || cols <= 0 {
		panic("cv: NewKernel requires positive dimensions")
	}
	if len(data) != rows*cols {
		panic(fmt.Sprintf("cv: NewKernel data length %d != %d*%d", len(data), rows, cols))
	}
	cp := make([]float64, len(data))
	copy(cp, data)
	return Kernel{Rows: rows, Cols: cols, Data: cp}
}

// Filter2D convolves each channel of src with kernel and returns a new Mat of
// the same shape. Borders are handled by edge replication. The delta is added
// to every filtered sample before rounding and clamping to [0,255].
//
// Following OpenCV, this performs correlation with the kernel anchored at its
// centre; for symmetric kernels correlation and true convolution coincide.
func Filter2D(src *Mat, kernel Kernel, delta float64) *Mat {
	dst := NewMat(src.Rows, src.Cols, src.Channels)
	ay := kernel.Rows / 2
	ax := kernel.Cols / 2
	for y := 0; y < src.Rows; y++ {
		for x := 0; x < src.Cols; x++ {
			for c := 0; c < src.Channels; c++ {
				var sum float64
				ki := 0
				for ky := 0; ky < kernel.Rows; ky++ {
					sy := y + ky - ay
					for kx := 0; kx < kernel.Cols; kx++ {
						sx := x + kx - ax
						sum += kernel.Data[ki] * float64(src.atReplicate(sy, sx, c))
						ki++
					}
				}
				dst.Data[dst.index(y, x)+c] = clampToUint8(sum + delta + 0.5)
			}
		}
	}
	return dst
}

// filter2DFloat convolves src into a per-channel float result without clamping,
// used internally where the intermediate range exceeds [0,255] (e.g. Sobel).
func filter2DFloat(src *Mat, kernel Kernel) [][]float64 {
	out := make([][]float64, src.Channels)
	for c := range out {
		out[c] = make([]float64, src.Total())
	}
	ay := kernel.Rows / 2
	ax := kernel.Cols / 2
	for y := 0; y < src.Rows; y++ {
		for x := 0; x < src.Cols; x++ {
			for c := 0; c < src.Channels; c++ {
				var sum float64
				ki := 0
				for ky := 0; ky < kernel.Rows; ky++ {
					sy := y + ky - ay
					for kx := 0; kx < kernel.Cols; kx++ {
						sx := x + kx - ax
						sum += kernel.Data[ki] * float64(src.atReplicate(sy, sx, c))
						ki++
					}
				}
				out[c][y*src.Cols+x] = sum
			}
		}
	}
	return out
}

// sepFilterFloat applies separable filtering with a horizontal kernel kx and a
// vertical kernel ky (each a 1-D weight slice) and returns per-channel floats.
func sepFilterFloat(src *Mat, kx, ky []float64) [][]float64 {
	ax := len(kx) / 2
	ay := len(ky) / 2
	tmp := make([][]float64, src.Channels)
	for c := range tmp {
		tmp[c] = make([]float64, src.Total())
	}
	// Horizontal pass over source samples.
	for y := 0; y < src.Rows; y++ {
		for x := 0; x < src.Cols; x++ {
			for c := 0; c < src.Channels; c++ {
				var sum float64
				for k := 0; k < len(kx); k++ {
					sx := x + k - ax
					sum += kx[k] * float64(src.atReplicate(y, sx, c))
				}
				tmp[c][y*src.Cols+x] = sum
			}
		}
	}
	// Vertical pass over the intermediate, replicating borders manually.
	out := make([][]float64, src.Channels)
	for c := range out {
		out[c] = make([]float64, src.Total())
	}
	clampRow := func(y int) int {
		if y < 0 {
			return 0
		}
		if y >= src.Rows {
			return src.Rows - 1
		}
		return y
	}
	for y := 0; y < src.Rows; y++ {
		for x := 0; x < src.Cols; x++ {
			for c := 0; c < src.Channels; c++ {
				var sum float64
				for k := 0; k < len(ky); k++ {
					sy := clampRow(y + k - ay)
					sum += ky[k] * tmp[c][sy*src.Cols+x]
				}
				out[c][y*src.Cols+x] = sum
			}
		}
	}
	return out
}

// floatChannelsToMat clamps per-channel floats (with rounding) into a Mat.
func floatChannelsToMat(rows, cols int, chans [][]float64) *Mat {
	out := NewMat(rows, cols, len(chans))
	for c := range chans {
		for i, v := range chans[c] {
			out.Data[i*len(chans)+c] = clampToUint8(v + 0.5)
		}
	}
	return out
}

// BoxFilter smooths src with a normalised or unnormalised ksize×ksize averaging
// kernel. When normalize is true the kernel sums to one (a mean filter);
// otherwise it is a plain sum. ksize must be a positive odd integer.
func BoxFilter(src *Mat, ksize int, normalize bool) *Mat {
	requireOdd(ksize, "BoxFilter")
	w := 1.0
	if normalize {
		w = 1.0 / float64(ksize*ksize)
	}
	data := make([]float64, ksize*ksize)
	for i := range data {
		data[i] = w
	}
	return Filter2D(src, NewKernel(ksize, ksize, data), 0)
}

// Blur smooths src with a normalised ksize×ksize box (mean) filter. It is a
// convenience wrapper over [BoxFilter] with normalize set to true.
func Blur(src *Mat, ksize int) *Mat {
	return BoxFilter(src, ksize, true)
}

// GaussianBlur convolves src with a separable Gaussian kernel of size
// ksize×ksize. sigma is the standard deviation; when sigma <= 0 it is derived
// from the kernel size as OpenCV does: 0.3*((ksize-1)*0.5 - 1) + 0.8. ksize
// must be a positive odd integer.
func GaussianBlur(src *Mat, ksize int, sigma float64) *Mat {
	requireOdd(ksize, "GaussianBlur")
	k := GaussianKernel1D(ksize, sigma)
	chans := sepFilterFloat(src, k, k)
	return floatChannelsToMat(src.Rows, src.Cols, chans)
}

// GaussianKernel1D returns a normalised 1-D Gaussian of length ksize. When
// sigma <= 0 it is derived from ksize (see [GaussianBlur]).
func GaussianKernel1D(ksize int, sigma float64) []float64 {
	requireOdd(ksize, "GaussianKernel1D")
	if sigma <= 0 {
		sigma = 0.3*(float64(ksize-1)*0.5-1) + 0.8
	}
	k := make([]float64, ksize)
	c := ksize / 2
	var sum float64
	for i := 0; i < ksize; i++ {
		d := float64(i - c)
		k[i] = math.Exp(-(d * d) / (2 * sigma * sigma))
		sum += k[i]
	}
	for i := range k {
		k[i] /= sum
	}
	return k
}

// MedianBlur replaces each sample with the median of its ksize×ksize
// neighbourhood, an effective remedy for salt-and-pepper noise. ksize must be
// a positive odd integer. Borders are replicated.
func MedianBlur(src *Mat, ksize int) *Mat {
	requireOdd(ksize, "MedianBlur")
	dst := NewMat(src.Rows, src.Cols, src.Channels)
	a := ksize / 2
	window := make([]uint8, 0, ksize*ksize)
	for y := 0; y < src.Rows; y++ {
		for x := 0; x < src.Cols; x++ {
			for c := 0; c < src.Channels; c++ {
				window = window[:0]
				for ky := -a; ky <= a; ky++ {
					for kx := -a; kx <= a; kx++ {
						window = append(window, src.atReplicate(y+ky, x+kx, c))
					}
				}
				sort.Slice(window, func(i, j int) bool { return window[i] < window[j] })
				dst.Data[dst.index(y, x)+c] = window[len(window)/2]
			}
		}
	}
	return dst
}

// Sobel computes an image derivative using the Sobel operator. dx and dy are
// the derivative orders (0, 1 or 2, not both zero) and ksize is the aperture
// (1 or 3). The gradient can be signed; results are scaled by scale, offset by
// delta and then clamped to [0,255], so callers wanting the true magnitude
// should combine the x and y results themselves. Use [SobelFloat] for
// unclamped output.
func Sobel(src *Mat, dx, dy, ksize int, scale, delta float64) *Mat {
	chans := SobelFloat(src, dx, dy, ksize)
	out := NewMat(src.Rows, src.Cols, src.Channels)
	for c := range chans {
		for i, v := range chans[c] {
			out.Data[i*src.Channels+c] = clampToUint8(v*scale + delta + 0.5)
		}
	}
	return out
}

// SobelFloat computes the Sobel derivative and returns unclamped per-channel
// float results, preserving sign and magnitude. See [Sobel] for the parameter
// meaning.
func SobelFloat(src *Mat, dx, dy, ksize int) [][]float64 {
	if dx < 0 || dy < 0 || dx+dy == 0 {
		panic("cv: Sobel requires dx,dy >= 0 and dx+dy > 0")
	}
	if ksize != 1 && ksize != 3 {
		panic("cv: Sobel supports ksize 1 or 3")
	}
	kx := sobelKernel1D(dx, ksize)
	ky := sobelKernel1D(dy, ksize)
	return sepFilterFloat(src, kx, ky)
}

// sobelKernel1D returns the 1-D smoothing or derivative kernel for the given
// order and aperture, matching OpenCV's getDerivKernels for Sobel.
func sobelKernel1D(order, ksize int) []float64 {
	if ksize == 1 {
		switch order {
		case 0:
			return []float64{1}
		default:
			return []float64{-1, 0, 1}
		}
	}
	// ksize == 3
	switch order {
	case 0:
		return []float64{1, 2, 1}
	case 1:
		return []float64{-1, 0, 1}
	default: // order 2
		return []float64{1, -2, 1}
	}
}

// Scharr computes an image derivative with the 3×3 Scharr operator, which has
// better rotational symmetry than a 3×3 Sobel. Exactly one of dx, dy must be 1
// and the other 0. Results are scaled, offset by delta and clamped to [0,255].
func Scharr(src *Mat, dx, dy int, scale, delta float64) *Mat {
	if !((dx == 1 && dy == 0) || (dx == 0 && dy == 1)) {
		panic("cv: Scharr requires exactly one of dx,dy to be 1")
	}
	deriv := []float64{-1, 0, 1}
	smooth := []float64{3, 10, 3}
	var kx, ky []float64
	if dx == 1 {
		kx, ky = deriv, smooth
	} else {
		kx, ky = smooth, deriv
	}
	chans := sepFilterFloat(src, kx, ky)
	out := NewMat(src.Rows, src.Cols, src.Channels)
	for c := range chans {
		for i, v := range chans[c] {
			out.Data[i*src.Channels+c] = clampToUint8(v*scale + delta + 0.5)
		}
	}
	return out
}

// Laplacian computes the Laplacian (sum of second spatial derivatives) of src.
// ksize 1 uses the classic 4-neighbour stencil; ksize 3 uses the Sobel-based
// second-derivative sum. Results are scaled, offset by delta and clamped.
func Laplacian(src *Mat, ksize int, scale, delta float64) *Mat {
	var chans [][]float64
	if ksize == 1 {
		k := NewKernel(3, 3, []float64{
			0, 1, 0,
			1, -4, 1,
			0, 1, 0,
		})
		chans = filter2DFloat(src, k)
	} else if ksize == 3 {
		xx := SobelFloat(src, 2, 0, 3)
		yy := SobelFloat(src, 0, 2, 3)
		chans = make([][]float64, src.Channels)
		for c := range chans {
			chans[c] = make([]float64, src.Total())
			for i := range chans[c] {
				chans[c][i] = xx[c][i] + yy[c][i]
			}
		}
	} else {
		panic("cv: Laplacian supports ksize 1 or 3")
	}
	out := NewMat(src.Rows, src.Cols, src.Channels)
	for c := range chans {
		for i, v := range chans[c] {
			out.Data[i*src.Channels+c] = clampToUint8(v*scale + delta + 0.5)
		}
	}
	return out
}

func requireOdd(ksize int, name string) {
	if ksize <= 0 || ksize%2 == 0 {
		panic(fmt.Sprintf("cv: %s requires a positive odd ksize, got %d", name, ksize))
	}
}
