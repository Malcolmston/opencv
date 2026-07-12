package imgprocx

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// SpatialGradient computes the first-order image derivatives of a single-channel
// image with a 3×3 Sobel operator, mirroring cv2.spatialGradient. It returns two
// [cv.FloatMat] planes the same size as src: dx holds the horizontal derivative
// (∂/∂x) and dy the vertical derivative (∂/∂y). Borders are handled by edge
// replication. It panics if src is not single-channel.
//
// The response at each pixel is the correlation of the neighbourhood with the
// Sobel kernels
//
//	Kx = [-1 0 1; -2 0 2; -1 0 1]   Ky = [-1 -2 -1; 0 0 0; 1 2 1]
//
// so a horizontal intensity ramp of unit slope yields dx = 8 in its interior.
func SpatialGradient(src *cv.Mat) (dx, dy *cv.FloatMat) {
	requireSingleChannel(src, "SpatialGradient")
	gray, rows, cols := toGrayPlane(src)
	gx := derivPlane(gray, rows, cols, 1, 0, 3)
	gy := derivPlane(gray, rows, cols, 0, 1, 3)
	dx = &cv.FloatMat{Rows: rows, Cols: cols, Data: gx}
	dy = &cv.FloatMat{Rows: rows, Cols: cols, Data: gy}
	return dx, dy
}

// CreateHanningWindow returns a rows×cols separable Hann (raised-cosine) window
// as a [cv.FloatMat], mirroring cv2.createHanningWindow. The window is the outer
// product of two 1-D Hann windows,
//
//	w(i, j) = wr(i)·wc(j),   wr(i) = 0.5·(1 - cos(2π·i/(rows-1)))
//
// (and likewise for the columns), so it tapers smoothly to zero at every edge.
// A dimension of length one contributes a unit factor. It panics unless both
// dimensions are positive.
//
// Multiplying an image by this window before a Fourier transform (for example
// ahead of [PhaseCorrelate]) suppresses the edge discontinuities that otherwise
// leak spectral energy across all frequencies.
func CreateHanningWindow(rows, cols int) *cv.FloatMat {
	if rows <= 0 || cols <= 0 {
		panic("imgprocx: CreateHanningWindow requires positive dimensions")
	}
	wr := hann1D(rows)
	wc := hann1D(cols)
	out := cv.NewFloatMat(rows, cols)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			out.Data[y*cols+x] = wr[y] * wc[x]
		}
	}
	return out
}

// hann1D returns the 1-D Hann window of length n, using a unit factor for n==1.
func hann1D(n int) []float64 {
	w := make([]float64, n)
	if n == 1 {
		w[0] = 1
		return w
	}
	denom := float64(n - 1)
	for i := 0; i < n; i++ {
		w[i] = 0.5 * (1 - math.Cos(2*math.Pi*float64(i)/denom))
	}
	return w
}
