package filters2

import (
	cv "github.com/malcolmston/opencv"
)

// LaplacianOfGaussian convolves a single-channel image with a
// Laplacian-of-Gaussian kernel of the given standard deviation and returns the
// signed band-pass response as a [FloatImage]. The kernel size is chosen from
// sigma unless overridden by a positive size. It panics on multi-channel or
// empty input or a non-positive sigma.
func LaplacianOfGaussian(src *cv.Mat, sigma float64, size int) *FloatImage {
	requireGray(src, "LaplacianOfGaussian")
	if size <= 0 {
		size = logKernelSize(sigma)
	}
	return ConvolveMat(src, LaplacianOfGaussianKernel(size, sigma))
}

// DifferenceOfGaussians convolves a single-channel image with the difference of
// two Gaussians of standard deviations sigma1 and sigma2 (with sigma1 < sigma2
// giving a band-pass response) and returns the signed result as a [FloatImage].
// The kernel size is chosen from the larger sigma unless overridden by a
// positive size. It panics on multi-channel or empty input or a non-positive
// sigma.
func DifferenceOfGaussians(src *cv.Mat, sigma1, sigma2 float64, size int) *FloatImage {
	requireGray(src, "DifferenceOfGaussians")
	if size <= 0 {
		s := sigma1
		if sigma2 > s {
			s = sigma2
		}
		size = logKernelSize(s)
	}
	return ConvolveMat(src, DifferenceOfGaussiansKernel(size, sigma1, sigma2))
}

// MarrHildrethEdges detects edges by the Marr-Hildreth method: it computes the
// Laplacian of Gaussian of a single-channel image and marks the zero crossings
// of the response whose local slope exceeds threshold, returning a binary
// [cv.Mat] with 255 on edges and 0 elsewhere. A threshold of 0 keeps every
// crossing. It panics on multi-channel or empty input or a non-positive sigma.
func MarrHildrethEdges(src *cv.Mat, sigma, threshold float64) *cv.Mat {
	requireGray(src, "MarrHildrethEdges")
	log := LaplacianOfGaussian(src, sigma, 0)
	rows, cols := log.Rows, log.Cols
	out := cv.NewMat(rows, cols, 1)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			c := log.At(y, x)
			// A zero crossing exists if a neighbour has the opposite sign and
			// the local variation across the crossing is large enough.
			edge := false
			// Check the right and bottom neighbours (and diagonal) to avoid
			// double counting.
			neighbours := [3][2]int{{0, 1}, {1, 0}, {1, 1}}
			for _, n := range neighbours {
				ny, nx := y+n[0], x+n[1]
				if ny >= rows || nx >= cols {
					continue
				}
				v := log.At(ny, nx)
				if (c > 0 && v < 0) || (c < 0 && v > 0) {
					if diff := c - v; diff >= threshold || -diff >= threshold {
						edge = true
						break
					}
				}
			}
			if edge {
				out.Data[y*cols+x] = 255
			}
		}
	}
	return out
}

// logKernelSize returns an odd kernel side big enough to hold a
// Laplacian-of-Gaussian of the given sigma (about 3 sigma either side).
func logKernelSize(sigma float64) int {
	r := gaussianRadius(sigma)
	return 2*r + 1
}
