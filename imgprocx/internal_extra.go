package imgprocx

import cv "github.com/malcolmston/opencv"

// clampIndex clamps i to the valid index range [0, n) by replicating the
// nearest border, the border rule shared by the derivative helpers.
func clampIndex(i, n int) int {
	if i < 0 {
		return 0
	}
	if i >= n {
		return n - 1
	}
	return i
}

// sepCorrelate applies the separable filter with horizontal taps kx and
// vertical taps ky to the single-channel plane gray (row-major, rows×cols),
// returning a fresh float plane of the same size. Both kernels are anchored at
// their centre and out-of-range samples are taken from the nearest border
// (BORDER_REPLICATE), matching the convention used by the root package's
// [cv.Filter2D].
func sepCorrelate(gray []float64, rows, cols int, kx, ky []float64) []float64 {
	ax := len(kx) / 2
	ay := len(ky) / 2
	tmp := make([]float64, rows*cols)
	for y := 0; y < rows; y++ {
		base := y * cols
		for x := 0; x < cols; x++ {
			var s float64
			for j := 0; j < len(kx); j++ {
				sx := clampIndex(x+j-ax, cols)
				s += kx[j] * gray[base+sx]
			}
			tmp[base+x] = s
		}
	}
	out := make([]float64, rows*cols)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			var s float64
			for i := 0; i < len(ky); i++ {
				sy := clampIndex(y+i-ay, rows)
				s += ky[i] * tmp[sy*cols+x]
			}
			out[y*cols+x] = s
		}
	}
	return out
}

// derivPlane computes the (dx, dy)-order image derivative of the single-channel
// plane gray using Sobel kernels of aperture ksize, returning the response as a
// fresh float plane. It is the shared work-horse behind [SpatialGradient],
// [CornerEigenValsAndVecs], [CornerMinEigenVal] and [PreCornerDetect].
func derivPlane(gray []float64, rows, cols, dx, dy, ksize int) []float64 {
	kx, ky := GetDerivKernels(dx, dy, ksize, false)
	return sepCorrelate(gray, rows, cols, kx, ky)
}

// requireSingleChannel panics unless m is a single-channel image.
func requireSingleChannel(m *cv.Mat, name string) {
	if m.Channels != 1 {
		panic("imgprocx: " + name + " requires a single-channel image")
	}
}
