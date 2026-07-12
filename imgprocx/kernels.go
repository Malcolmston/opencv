package imgprocx

import "math"

// smallGaussianTables holds the fixed separable Gaussian coefficients OpenCV
// uses for small apertures (ksize 1, 3, 5, 7) when no explicit sigma is given.
// They are the exact tables from cv::getGaussianKernel.
var smallGaussianTables = map[int][]float64{
	1: {1},
	3: {0.25, 0.5, 0.25},
	5: {0.0625, 0.25, 0.375, 0.25, 0.0625},
	7: {0.03125, 0.109375, 0.21875, 0.28125, 0.21875, 0.109375, 0.03125},
}

// GetGaussianKernel returns the ksize normalised 1-D Gaussian smoothing
// coefficients, mirroring cv2.getGaussianKernel. ksize must be a positive odd
// integer; it panics otherwise.
//
// When sigma is positive it is used directly. When sigma is not positive an
// aperture-derived value
//
//	σ = 0.3·((ksize-1)·0.5 - 1) + 0.8
//
// is used, matching OpenCV; additionally, for the small apertures 1, 3, 5 and 7
// the exact fixed tables OpenCV ships are returned instead of the sampled
// Gaussian. In every case the returned coefficients sum to one, so convolving
// with them preserves a constant image. The i-th coefficient corresponds to the
// offset i-(ksize-1)/2 from the kernel centre.
//
// The full 2-D Gaussian used by [cv.GaussianBlur] is the outer product of this
// kernel with itself; returning the 1-D factor lets callers filter separably.
func GetGaussianKernel(ksize int, sigma float64) []float64 {
	if ksize <= 0 || ksize%2 == 0 {
		panic("imgprocx: GetGaussianKernel requires a positive odd ksize")
	}
	if sigma <= 0 {
		if tbl, ok := smallGaussianTables[ksize]; ok {
			out := make([]float64, ksize)
			copy(out, tbl)
			return out
		}
		sigma = 0.3*((float64(ksize)-1)*0.5-1) + 0.8
	}
	scale2X := -0.5 / (sigma * sigma)
	center := float64(ksize-1) * 0.5
	out := make([]float64, ksize)
	var sum float64
	for i := 0; i < ksize; i++ {
		x := float64(i) - center
		v := math.Exp(scale2X * x * x)
		out[i] = v
		sum += v
	}
	inv := 1.0 / sum
	for i := range out {
		out[i] *= inv
	}
	return out
}

// GetDerivKernels returns the pair of 1-D filter kernels (kx applied along the
// columns, ky along the rows) whose separable application yields the image
// derivative of order dx in x and dy in y, mirroring cv2.getDerivKernels for
// Sobel apertures. ksize must be an odd value in {1, 3, 5, 7, ...} and strictly
// greater than the corresponding derivative order; an aperture of 1 is promoted
// to 3 for a positive order, as in OpenCV. It panics on invalid arguments.
//
// The kernels are built the way OpenCV builds Sobel operators: repeated
// convolution of the seed [1] with the smoothing kernel [1,1] (a row of
// Pascal's triangle) for the non-differentiated passes, followed by convolution
// with the difference kernel [1,-1] once per derivative order. When normalize is
// true each kernel is divided by 2^(ksize-order-1) so that filtering a linear
// ramp reproduces the analytic derivative; otherwise the raw integer taps are
// returned.
func GetDerivKernels(dx, dy, ksize int, normalize bool) (kx, ky []float64) {
	if dx < 0 || dy < 0 {
		panic("imgprocx: GetDerivKernels requires dx>=0 and dy>=0")
	}
	kx = sobelKernel1D(dx, ksize, normalize)
	ky = sobelKernel1D(dy, ksize, normalize)
	return kx, ky
}

// sobelKernel1D builds the 1-D Sobel coefficient vector for the given
// derivative order and aperture, optionally normalised. See [GetDerivKernels].
func sobelKernel1D(order, ksize int, normalize bool) []float64 {
	if ksize == 1 && order > 0 {
		ksize = 3
	}
	if ksize <= 0 || ksize%2 == 0 {
		panic("imgprocx: GetDerivKernels requires a positive odd ksize")
	}
	if order >= ksize {
		panic("imgprocx: GetDerivKernels requires ksize > derivative order")
	}
	// ker has one element of scratch space beyond the kernel proper so the
	// smoothing/difference recurrences can read one past the end.
	ker := make([]int, ksize+1)
	if ksize == 1 {
		ker[0] = 1
	} else {
		ker[0] = 1
		for i := 0; i < ksize-order-1; i++ {
			oldval := ker[0]
			for j := 1; j <= ksize; j++ {
				newval := ker[j] + ker[j-1]
				ker[j-1] = oldval
				oldval = newval
			}
		}
		for i := 0; i < order; i++ {
			oldval := -ker[0]
			for j := 1; j <= ksize; j++ {
				newval := ker[j-1] - ker[j]
				ker[j-1] = oldval
				oldval = newval
			}
		}
	}
	scale := 1.0
	if normalize {
		scale = 1.0 / float64(int(1)<<(ksize-order-1))
	}
	out := make([]float64, ksize)
	for i := 0; i < ksize; i++ {
		out[i] = float64(ker[i]) * scale
	}
	return out
}
