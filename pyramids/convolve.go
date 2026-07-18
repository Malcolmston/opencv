package pyramids

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// BinomialKernel returns the normalised 5-tap binomial kernel
// [1 4 6 4 1]/16, the standard separable approximation to a Gaussian used for
// pyramid reduction and expansion.
func BinomialKernel() []float64 {
	return []float64{1.0 / 16, 4.0 / 16, 6.0 / 16, 4.0 / 16, 1.0 / 16}
}

// GaussianKernel returns a normalised 1-D Gaussian kernel with standard
// deviation sigma. The radius is ceil(3*sigma) so the kernel captures more than
// 99% of the mass; the returned slice therefore has length 2*ceil(3*sigma)+1.
// It panics if sigma is not positive.
func GaussianKernel(sigma float64) []float64 {
	if sigma <= 0 {
		panic("pyramids: GaussianKernel: sigma must be positive")
	}
	r := int(math.Ceil(3 * sigma))
	k := make([]float64, 2*r+1)
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

// GaussianDerivativeKernel returns a 1-D first-derivative-of-Gaussian kernel
// with standard deviation sigma, sampled over radius ceil(3*sigma). It is the
// analytic derivative g'(x) = -(x/sigma^2) g(x) and integrates to zero, making
// it suitable as an oriented edge filter. It panics if sigma is not positive.
func GaussianDerivativeKernel(sigma float64) []float64 {
	if sigma <= 0 {
		panic("pyramids: GaussianDerivativeKernel: sigma must be positive")
	}
	r := int(math.Ceil(3 * sigma))
	k := make([]float64, 2*r+1)
	var norm float64
	for i := -r; i <= r; i++ {
		norm += math.Exp(-float64(i*i) / (2 * sigma * sigma))
	}
	for i := -r; i <= r; i++ {
		x := float64(i)
		g := math.Exp(-(x*x)/(2*sigma*sigma)) / norm
		k[i+r] = -(x / (sigma * sigma)) * g
	}
	return k
}

// GaussianSecondDerivativeKernel returns a 1-D second-derivative-of-Gaussian
// kernel with standard deviation sigma over radius ceil(3*sigma). It is the
// analytic g”(x) = ((x^2/sigma^2) - 1)/sigma^2 * g(x) and integrates to zero.
// It panics if sigma is not positive.
func GaussianSecondDerivativeKernel(sigma float64) []float64 {
	if sigma <= 0 {
		panic("pyramids: GaussianSecondDerivativeKernel: sigma must be positive")
	}
	r := int(math.Ceil(3 * sigma))
	k := make([]float64, 2*r+1)
	var norm float64
	for i := -r; i <= r; i++ {
		norm += math.Exp(-float64(i*i) / (2 * sigma * sigma))
	}
	s2 := sigma * sigma
	var sum float64
	for i := -r; i <= r; i++ {
		x := float64(i)
		g := math.Exp(-(x*x)/(2*s2)) / norm
		k[i+r] = ((x*x)/s2 - 1) / s2 * g
		sum += k[i+r]
	}
	// The discretely sampled second derivative has a small non-zero sum;
	// subtract its mean so the kernel is exactly DC-free (zero response on a
	// constant signal), matching the analytic filter.
	mean := sum / float64(len(k))
	for i := range k {
		k[i] -= mean
	}
	return k
}

// convolveRows convolves each row of f with the 1-D kernel k (anchored at its
// centre), replicating the border, and returns a new grid.
func convolveRows(f *cv.FloatMat, k []float64) *cv.FloatMat {
	r := len(k) / 2
	out := cv.NewFloatMat(f.Rows, f.Cols)
	for y := 0; y < f.Rows; y++ {
		base := y * f.Cols
		for x := 0; x < f.Cols; x++ {
			var s float64
			for t := -r; t <= r; t++ {
				xx := x + t
				if xx < 0 {
					xx = 0
				} else if xx >= f.Cols {
					xx = f.Cols - 1
				}
				s += k[t+r] * f.Data[base+xx]
			}
			out.Data[base+x] = s
		}
	}
	return out
}

// convolveCols convolves each column of f with the 1-D kernel k (anchored at
// its centre), replicating the border, and returns a new grid.
func convolveCols(f *cv.FloatMat, k []float64) *cv.FloatMat {
	r := len(k) / 2
	out := cv.NewFloatMat(f.Rows, f.Cols)
	for y := 0; y < f.Rows; y++ {
		for x := 0; x < f.Cols; x++ {
			var s float64
			for t := -r; t <= r; t++ {
				yy := y + t
				if yy < 0 {
					yy = 0
				} else if yy >= f.Rows {
					yy = f.Rows - 1
				}
				s += k[t+r] * f.Data[yy*f.Cols+x]
			}
			out.Data[y*f.Cols+x] = s
		}
	}
	return out
}

// ConvolveSeparable convolves f with the separable kernel formed by the outer
// product of the horizontal kernel kx and the vertical kernel ky, replicating
// the border. Passing kx and ky both equal to a 1-D Gaussian yields a Gaussian
// blur; mixing a derivative kernel with a smoothing kernel yields an oriented
// derivative. Both kernels must have odd length.
func ConvolveSeparable(f *cv.FloatMat, kx, ky []float64) *cv.FloatMat {
	pyramidsRequire(f, "ConvolveSeparable")
	if len(kx)%2 == 0 || len(ky)%2 == 0 {
		panic("pyramids: ConvolveSeparable: kernels must have odd length")
	}
	return convolveCols(convolveRows(f, kx), ky)
}

// Convolve2D convolves f with an arbitrary 2-D kernel anchored at its centre,
// replicating the border, and returns a new grid. The kernel is a [cv.FloatMat]
// with odd dimensions. This performs correlation; for symmetric kernels it
// coincides with true convolution.
func Convolve2D(f *cv.FloatMat, kernel *cv.FloatMat) *cv.FloatMat {
	pyramidsRequire(f, "Convolve2D")
	pyramidsRequire(kernel, "Convolve2D")
	if kernel.Rows%2 == 0 || kernel.Cols%2 == 0 {
		panic("pyramids: Convolve2D: kernel must have odd dimensions")
	}
	ay := kernel.Rows / 2
	ax := kernel.Cols / 2
	out := cv.NewFloatMat(f.Rows, f.Cols)
	for y := 0; y < f.Rows; y++ {
		for x := 0; x < f.Cols; x++ {
			var s float64
			ki := 0
			for ky := 0; ky < kernel.Rows; ky++ {
				sy := y + ky - ay
				for kx := 0; kx < kernel.Cols; kx++ {
					sx := x + kx - ax
					s += kernel.Data[ki] * pyramidsAt(f, sy, sx)
					ki++
				}
			}
			out.Data[y*f.Cols+x] = s
		}
	}
	return out
}

// GaussianBlurFloat blurs f with a separable Gaussian of standard deviation
// sigma, replicating the border, and returns a new grid. It panics if sigma is
// not positive.
func GaussianBlurFloat(f *cv.FloatMat, sigma float64) *cv.FloatMat {
	pyramidsRequire(f, "GaussianBlurFloat")
	k := GaussianKernel(sigma)
	return ConvolveSeparable(f, k, k)
}

// LaplacianOfGaussianKernel returns a square, odd-sized 2-D Laplacian-of-
// Gaussian (Mexican-hat) kernel with standard deviation sigma. The kernel sums
// to zero so it responds to blobs and edges but not to flat regions. The size
// is 2*ceil(3*sigma)+1. It panics if sigma is not positive.
func LaplacianOfGaussianKernel(sigma float64) *cv.FloatMat {
	if sigma <= 0 {
		panic("pyramids: LaplacianOfGaussianKernel: sigma must be positive")
	}
	r := int(math.Ceil(3 * sigma))
	n := 2*r + 1
	k := cv.NewFloatMat(n, n)
	s2 := sigma * sigma
	var sum float64
	for y := -r; y <= r; y++ {
		for x := -r; x <= r; x++ {
			rr := float64(x*x + y*y)
			g := math.Exp(-rr/(2*s2)) * (rr/s2 - 2) / (s2)
			k.Data[(y+r)*n+(x+r)] = g
			sum += g
		}
	}
	// Force an exact zero sum so a constant image yields a zero response.
	mean := sum / float64(n*n)
	for i := range k.Data {
		k.Data[i] -= mean
	}
	return k
}
