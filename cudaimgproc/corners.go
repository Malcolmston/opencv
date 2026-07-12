package cudaimgproc

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// CornernessCriteria is the common interface of the CPU-backed corner response
// operators [HarrisCorner] and [MinEigenValCorner], mirroring
// cv::cuda::CornernessCriteria. Compute returns a per-pixel response map as a
// [cv.FloatMat].
type CornernessCriteria interface {
	// Compute evaluates the corner response of a single-channel GpuMat.
	Compute(src GpuMat, streams ...Stream) *cv.FloatMat
}

// HarrisCorner is a CPU-backed Harris corner-response operator, mirroring the
// object returned by cv::cuda::createHarrisCorner. Create one with
// [CreateHarrisCorner] and evaluate it with [HarrisCorner.Compute].
type HarrisCorner struct {
	blockSize int
	ksize     int
	k         float64
}

// CreateHarrisCorner returns a [HarrisCorner] operator, mirroring
// cuda::createHarrisCorner. blockSize is the neighbourhood size over which the
// structure tensor is summed, ksize the Sobel aperture, and k the Harris
// sensitivity parameter (typically 0.04–0.06).
func CreateHarrisCorner(blockSize, ksize int, k float64) *HarrisCorner {
	return &HarrisCorner{blockSize: blockSize, ksize: ksize, k: k}
}

// Compute returns the Harris response R = det(M) - k·trace(M)² for each pixel
// of a single-channel GpuMat. The trailing Stream argument is accepted and
// ignored. It panics unless src is single-channel.
func (h *HarrisCorner) Compute(src GpuMat, streams ...Stream) *cv.FloatMat {
	_ = firstStream(streams)
	m := src.requireHost("HarrisCorner.Compute")
	return cv.CornerHarris(m, h.blockSize, h.ksize, h.k)
}

// MinEigenValCorner is a CPU-backed Shi–Tomasi corner-response operator (the
// minimum eigenvalue of the structure tensor), mirroring the object returned by
// cv::cuda::createMinEigenValCorner. Create one with [CreateMinEigenValCorner]
// and evaluate it with [MinEigenValCorner.Compute].
type MinEigenValCorner struct {
	blockSize int
	ksize     int
}

// CreateMinEigenValCorner returns a [MinEigenValCorner] operator, mirroring
// cuda::createMinEigenValCorner. blockSize is the structure-tensor window size
// and ksize the Sobel aperture.
func CreateMinEigenValCorner(blockSize, ksize int) *MinEigenValCorner {
	return &MinEigenValCorner{blockSize: blockSize, ksize: ksize}
}

// Compute returns the smaller eigenvalue of the windowed structure tensor for
// each pixel of a single-channel GpuMat — the Shi–Tomasi corner measure. The
// trailing Stream argument is accepted and ignored. It panics unless src is
// single-channel.
func (mvc *MinEigenValCorner) Compute(src GpuMat, streams ...Stream) *cv.FloatMat {
	_ = firstStream(streams)
	m := src.requireHost("MinEigenValCorner.Compute")
	sxx, syy, sxy := structureTensor(m, mvc.blockSize, mvc.ksize)
	res := cv.NewFloatMat(m.Rows, m.Cols)
	for i := range res.Data {
		a, b, c := sxx[i], sxy[i], syy[i]
		tr := a + c
		disc := math.Sqrt((a-c)*(a-c) + 4*b*b)
		res.Data[i] = (tr - disc) / 2
	}
	return res
}

// structureTensor computes the windowed sums of gradient products (Sxx, Syy,
// Sxy) from the Sobel derivatives of src, replicating the border. It mirrors
// the structure tensor used by the root package's corner detectors but is
// re-derived here from the exported [cv.SobelFloat] so this package depends only
// on cv's public API.
func structureTensor(m *cv.Mat, blockSize, ksize int) (sxx, syy, sxy []float64) {
	ix := cv.SobelFloat(m, 1, 0, ksize)[0]
	iy := cv.SobelFloat(m, 0, 1, ksize)[0]
	n := m.Rows * m.Cols
	ixx := make([]float64, n)
	iyy := make([]float64, n)
	ixy := make([]float64, n)
	for i := 0; i < n; i++ {
		ixx[i] = ix[i] * ix[i]
		iyy[i] = iy[i] * iy[i]
		ixy[i] = ix[i] * iy[i]
	}
	sxx = boxSum(ixx, m.Rows, m.Cols, blockSize)
	syy = boxSum(iyy, m.Rows, m.Cols, blockSize)
	sxy = boxSum(ixy, m.Rows, m.Cols, blockSize)
	return
}

// boxSum sums a per-pixel field over a blockSize×blockSize window centred on
// each pixel, clamping (replicating) at the border.
func boxSum(field []float64, rows, cols, blockSize int) []float64 {
	out := make([]float64, rows*cols)
	a := blockSize / 2
	clamp := func(v, hi int) int {
		if v < 0 {
			return 0
		}
		if v >= hi {
			return hi - 1
		}
		return v
	}
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			var s float64
			for dy := -a; dy <= a; dy++ {
				ry := clamp(y+dy, rows)
				for dx := -a; dx <= a; dx++ {
					rx := clamp(x+dx, cols)
					s += field[ry*cols+rx]
				}
			}
			out[y*cols+x] = s
		}
	}
	return out
}
