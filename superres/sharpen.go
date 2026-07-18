package superres

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// GaussianKernel1D returns a normalised 1-D Gaussian kernel for the given
// standard deviation sigma. The kernel is truncated at radius ceil(3*sigma)
// and its taps sum to one. It panics if sigma is not positive.
func GaussianKernel1D(sigma float64) []float64 {
	if sigma <= 0 {
		panic("superres: GaussianKernel1D requires sigma > 0")
	}
	radius := int(math.Ceil(3 * sigma))
	if radius < 1 {
		radius = 1
	}
	k := make([]float64, 2*radius+1)
	var sum float64
	inv := 1.0 / (2 * sigma * sigma)
	for i := -radius; i <= radius; i++ {
		v := math.Exp(-float64(i*i) * inv)
		k[i+radius] = v
		sum += v
	}
	for i := range k {
		k[i] /= sum
	}
	return k
}

// superresPlaneBlur applies a separable 1-D kernel to a float plane with
// border replication, returning a new plane.
func superresPlaneBlur(p *superresPlane, k []float64) *superresPlane {
	radius := len(k) / 2
	// Horizontal pass.
	h := newSuperresPlane(p.rows, p.cols)
	for y := 0; y < p.rows; y++ {
		for x := 0; x < p.cols; x++ {
			var acc float64
			for i := -radius; i <= radius; i++ {
				acc += k[i+radius] * p.at(y, x+i)
			}
			h.set(y, x, acc)
		}
	}
	// Vertical pass.
	v := newSuperresPlane(p.rows, p.cols)
	for y := 0; y < p.rows; y++ {
		for x := 0; x < p.cols; x++ {
			var acc float64
			for i := -radius; i <= radius; i++ {
				acc += k[i+radius] * h.at(y+i, x)
			}
			v.set(y, x, acc)
		}
	}
	return v
}

// GaussianBlur convolves src with a separable Gaussian of standard deviation
// sigma, replicating the border. Each channel is blurred independently. It
// panics if sigma is not positive.
func GaussianBlur(src *cv.Mat, sigma float64) *cv.Mat {
	k := GaussianKernel1D(sigma)
	planes := superresSplitPlanes(src)
	for i, p := range planes {
		planes[i] = superresPlaneBlur(p, k)
	}
	return superresMergePlanes(planes)
}

// UnsharpMask sharpens src with the classic unsharp-mask operator: the output
// is src + amount·(src − Gaussian(src, sigma)), applied per channel. amount is
// the sharpening strength (0 leaves the image unchanged; 1 roughly doubles
// local contrast). threshold suppresses sharpening where the absolute
// high-pass response is at or below it, which avoids amplifying noise in flat
// regions; pass 0 to sharpen everywhere. It panics if sigma is not positive.
func UnsharpMask(src *cv.Mat, sigma, amount, threshold float64) *cv.Mat {
	k := GaussianKernel1D(sigma)
	planes := superresSplitPlanes(src)
	out := make([]*superresPlane, len(planes))
	for i, p := range planes {
		blur := superresPlaneBlur(p, k)
		res := newSuperresPlane(p.rows, p.cols)
		for j := range p.data {
			high := p.data[j] - blur.data[j]
			if math.Abs(high) <= threshold {
				res.data[j] = p.data[j]
			} else {
				res.data[j] = p.data[j] + amount*high
			}
		}
		out[i] = res
	}
	return superresMergePlanes(out)
}

// SharpenLaplacian sharpens src by subtracting a scaled discrete Laplacian:
// out = src − strength·∇²src, using the 4-neighbour Laplacian. strength around
// 0.5–1 gives a crisp result; larger values over-sharpen. Each channel is
// processed independently with border replication.
func SharpenLaplacian(src *cv.Mat, strength float64) *cv.Mat {
	planes := superresSplitPlanes(src)
	out := make([]*superresPlane, len(planes))
	for i, p := range planes {
		res := newSuperresPlane(p.rows, p.cols)
		for y := 0; y < p.rows; y++ {
			for x := 0; x < p.cols; x++ {
				lap := p.at(y-1, x) + p.at(y+1, x) + p.at(y, x-1) + p.at(y, x+1) - 4*p.at(y, x)
				res.set(y, x, p.at(y, x)-strength*lap)
			}
		}
		out[i] = res
	}
	return superresMergePlanes(out)
}

// AdaptiveUnsharpMask is like [UnsharpMask] but scales the sharpening amount by
// the local gradient magnitude, so strong edges are sharpened more than weak
// texture. edgeScale controls how quickly the amount ramps toward its maximum
// with gradient magnitude (larger edgeScale reaches full strength sooner). It
// panics if sigma is not positive.
func AdaptiveUnsharpMask(src *cv.Mat, sigma, amount, edgeScale float64) *cv.Mat {
	if edgeScale <= 0 {
		edgeScale = 1
	}
	k := GaussianKernel1D(sigma)
	planes := superresSplitPlanes(src)
	out := make([]*superresPlane, len(planes))
	for i, p := range planes {
		blur := superresPlaneBlur(p, k)
		res := newSuperresPlane(p.rows, p.cols)
		for y := 0; y < p.rows; y++ {
			for x := 0; x < p.cols; x++ {
				gx := p.at(y, x+1) - p.at(y, x-1)
				gy := p.at(y+1, x) - p.at(y-1, x)
				mag := math.Hypot(gx, gy)
				gain := amount * (1 - math.Exp(-mag/edgeScale))
				high := p.atRaw(y, x) - blur.atRaw(y, x)
				res.set(y, x, p.atRaw(y, x)+gain*high)
			}
		}
		out[i] = res
	}
	return superresMergePlanes(out)
}

// UpscaleAndSharpen enlarges src to width×height with the named interpolation
// and then applies an unsharp mask, the usual remedy for the softness that any
// linear upscaler introduces. It is a convenience wrapper over [Resize] and
// [UnsharpMask]. It panics if sigma is not positive.
func UpscaleAndSharpen(src *cv.Mat, width, height int, interp Interpolation, sigma, amount float64) *cv.Mat {
	up := Resize(src, width, height, interp)
	return UnsharpMask(up, sigma, amount, 0)
}
