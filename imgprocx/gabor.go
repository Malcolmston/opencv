package imgprocx

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// GetGaborKernel builds a Gabor filter kernel — a planar sinusoid modulated by a
// Gaussian envelope — of size ksize×ksize, mirroring cv2.getGaborKernel. The
// result is a [cv.Kernel] ready for convolution with [cv.Filter2D].
//
// The kernel value at offset (x, y) from the centre is
//
//	exp(-(x'² + γ²·y'²) / (2σ²)) · cos(2π·x'/λ + ψ)
//
// where (x', y') is (x, y) rotated by theta:
//
//	x' =  x·cosθ + y·sinθ
//	y' = -x·sinθ + y·cosθ
//
// The parameters are the Gaussian standard deviation sigma (σ), the orientation
// theta (θ) in radians, the sinusoid wavelength lambda (λ) in pixels, the
// spatial aspect ratio gamma (γ) governing the ellipticity of the envelope, and
// the phase offset psi (ψ) in radians. ksize must be a positive odd integer so
// the kernel has a well-defined centre; it panics otherwise or if lambda or
// sigma is not positive.
//
// Because the cosine carrier oscillates under the Gaussian envelope, a Gabor
// kernel with several cycles across its support has a near-zero mean, making it
// an oriented band-pass filter.
func GetGaborKernel(ksize int, sigma, theta, lambda, gamma, psi float64) cv.Kernel {
	if ksize <= 0 || ksize%2 == 0 {
		panic("imgprocx: GetGaborKernel requires a positive odd ksize")
	}
	if sigma <= 0 {
		panic("imgprocx: GetGaborKernel requires sigma > 0")
	}
	if lambda <= 0 {
		panic("imgprocx: GetGaborKernel requires lambda > 0")
	}
	sigmaX := sigma
	sigmaY := sigma / gamma
	c := math.Cos(theta)
	s := math.Sin(theta)
	ex := -0.5 / (sigmaX * sigmaX)
	ey := -0.5 / (sigmaY * sigmaY)
	cscale := 2 * math.Pi / lambda
	half := ksize / 2
	data := make([]float64, ksize*ksize)
	i := 0
	for y := -half; y <= half; y++ {
		for x := -half; x <= half; x++ {
			xr := float64(x)*c + float64(y)*s
			yr := -float64(x)*s + float64(y)*c
			data[i] = math.Exp(ex*xr*xr+ey*yr*yr) * math.Cos(cscale*xr+psi)
			i++
		}
	}
	return cv.NewKernel(ksize, ksize, data)
}
