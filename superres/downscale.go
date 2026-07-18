package superres

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// BoxDownscale shrinks src by an integer factor using box (average) pooling:
// each output sample is the mean of a factor×factor block of source samples.
// Any partial block at the right/bottom edge is averaged over its available
// samples. It panics if factor is not positive.
func BoxDownscale(src *cv.Mat, factor int) *cv.Mat {
	if factor <= 0 {
		panic("superres: BoxDownscale requires a positive factor")
	}
	if factor == 1 {
		return src.Clone()
	}
	ch := src.Channels
	outH := (src.Rows + factor - 1) / factor
	outW := (src.Cols + factor - 1) / factor
	dst := cv.NewMat(outH, outW, ch)
	for oy := 0; oy < outH; oy++ {
		for ox := 0; ox < outW; ox++ {
			for c := 0; c < ch; c++ {
				var sum float64
				var n int
				for dy := 0; dy < factor; dy++ {
					sy := oy*factor + dy
					if sy >= src.Rows {
						break
					}
					for dx := 0; dx < factor; dx++ {
						sx := ox*factor + dx
						if sx >= src.Cols {
							break
						}
						sum += float64(src.Data[(sy*src.Cols+sx)*ch+c])
						n++
					}
				}
				dst.Data[(oy*outW+ox)*ch+c] = superresClamp8(sum / float64(n))
			}
		}
	}
	return dst
}

// GaussianDownscale shrinks src by an integer factor after a Gaussian
// anti-aliasing blur, the correct way to downsample without introducing
// aliasing. If sigma is not positive a default of 0.5·factor (roughly matching
// the new Nyquist limit) is used. It panics if factor is not positive.
func GaussianDownscale(src *cv.Mat, factor int, sigma float64) *cv.Mat {
	if factor <= 0 {
		panic("superres: GaussianDownscale requires a positive factor")
	}
	if factor == 1 {
		return src.Clone()
	}
	if sigma <= 0 {
		sigma = 0.5 * float64(factor)
	}
	blurred := GaussianBlur(src, sigma)
	ch := src.Channels
	outH := src.Rows / factor
	outW := src.Cols / factor
	if outH < 1 {
		outH = 1
	}
	if outW < 1 {
		outW = 1
	}
	dst := cv.NewMat(outH, outW, ch)
	for oy := 0; oy < outH; oy++ {
		sy := superresClampInt(oy*factor+factor/2, 0, blurred.Rows-1)
		for ox := 0; ox < outW; ox++ {
			sx := superresClampInt(ox*factor+factor/2, 0, blurred.Cols-1)
			for c := 0; c < ch; c++ {
				dst.Data[(oy*outW+ox)*ch+c] = blurred.Data[(sy*blurred.Cols+sx)*ch+c]
			}
		}
	}
	return dst
}

// superresPlaneResize resamples a float plane to width×height with kernel k,
// keeping full precision (used by the iterative routines). It mirrors
// [ResizeKernel] but neither reads nor writes uint8 samples.
func superresPlaneResize(p *superresPlane, width, height int, k ResampleKernel) *superresPlane {
	xw := superresAxisWeights(width, p.cols, k)
	tmp := newSuperresPlane(p.rows, width)
	for y := 0; y < p.rows; y++ {
		for dx := 0; dx < width; dx++ {
			var acc float64
			for _, t := range xw[dx] {
				acc += t.weight * p.atRaw(y, t.index)
			}
			tmp.set(y, dx, acc)
		}
	}
	yw := superresAxisWeights(height, p.rows, k)
	out := newSuperresPlane(height, width)
	for dy := 0; dy < height; dy++ {
		for x := 0; x < width; x++ {
			var acc float64
			for _, t := range yw[dy] {
				acc += t.weight * tmp.atRaw(t.index, x)
			}
			out.set(dy, x, acc)
		}
	}
	return out
}

// superresPlaneGaussianBlur is a convenience wrapper computing the kernel once.
func superresPlaneGaussianBlur(p *superresPlane, sigma float64) *superresPlane {
	return superresPlaneBlur(p, GaussianKernel1D(sigma))
}

// superresGaussianSigmaFor returns a reasonable blur sigma for the given
// integer upscale factor.
func superresGaussianSigmaFor(scale int) float64 {
	return math.Max(0.5, 0.5*float64(scale))
}
