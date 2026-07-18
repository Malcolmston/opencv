package superres

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// GradientProfileSharpen sharpens the edges of an (already upscaled) image by
// steepening its gradient profile, the core idea behind gradient-profile-prior
// super-resolution. Around every pixel it estimates the local edge direction
// from the image gradient and pushes each sample toward the value found a short
// step up-gradient / down-gradient, which narrows blurred edge transitions
// without amplifying flat-region noise. strength scales the effect (0 leaves
// the image unchanged; 0.5–1 is typical). Each channel is processed
// independently. It panics if strength is negative.
func GradientProfileSharpen(src *cv.Mat, strength float64) *cv.Mat {
	if strength < 0 {
		panic("superres: GradientProfileSharpen requires strength >= 0")
	}
	planes := superresSplitPlanes(src)
	out := make([]*superresPlane, len(planes))
	for i, p := range planes {
		res := newSuperresPlane(p.rows, p.cols)
		for y := 0; y < p.rows; y++ {
			for x := 0; x < p.cols; x++ {
				gx := 0.5 * (p.at(y, x+1) - p.at(y, x-1))
				gy := 0.5 * (p.at(y+1, x) - p.at(y-1, x))
				mag := math.Hypot(gx, gy)
				center := p.atRaw(y, x)
				if mag < 1e-6 {
					res.set(y, x, center)
					continue
				}
				// Unit step along the gradient direction.
				ux := gx / mag
				uy := gy / mag
				up := superresBilinearPlane(p, float64(x)+ux, float64(y)+uy)
				down := superresBilinearPlane(p, float64(x)-ux, float64(y)-uy)
				// Second difference along the edge normal is the un-sharp
				// signal; subtracting it steepens the profile.
				lap := up + down - 2*center
				res.set(y, x, center-strength*lap)
			}
		}
		out[i] = res
	}
	return superresMergePlanes(out)
}

// superresBilinearPlane samples a float plane at fractional (x, y) with border
// replication.
func superresBilinearPlane(p *superresPlane, x, y float64) float64 {
	ix := int(math.Floor(x))
	iy := int(math.Floor(y))
	fx := x - float64(ix)
	fy := y - float64(iy)
	v00 := p.at(iy, ix)
	v01 := p.at(iy, ix+1)
	v10 := p.at(iy+1, ix)
	v11 := p.at(iy+1, ix+1)
	top := v00 + (v01-v00)*fx
	bot := v10 + (v11-v10)*fx
	return top + (bot-top)*fy
}

// ProgressiveUpscale enlarges src to the target integer scale in small
// geometric steps rather than a single jump, sharpening after each step. Taking
// several modest bicubic steps (each about 1.5×) with intermediate sharpening
// preserves detail better than one large resize, a standard "self-example-free"
// single-image enlargement trick. stepSharpen is the unsharp-mask amount
// applied after every step (0 disables it). It panics if scale < 1.
func ProgressiveUpscale(src *cv.Mat, scale float64, stepSharpen float64) *cv.Mat {
	if scale < 1 {
		panic("superres: ProgressiveUpscale requires scale >= 1")
	}
	const stepFactor = 1.5
	out := src
	current := 1.0
	for current*stepFactor < scale {
		next := current * stepFactor
		w := int(math.Round(float64(src.Cols) * next))
		h := int(math.Round(float64(src.Rows) * next))
		out = BicubicResize(out, w, h)
		if stepSharpen > 0 {
			out = UnsharpMask(out, 1.0, stepSharpen, 0)
		}
		current = next
	}
	// Final step to the exact scale.
	w := int(math.Round(float64(src.Cols) * scale))
	h := int(math.Round(float64(src.Rows) * scale))
	if w < 1 {
		w = 1
	}
	if h < 1 {
		h = 1
	}
	out = BicubicResize(out, w, h)
	if stepSharpen > 0 {
		out = UnsharpMask(out, 1.0, stepSharpen, 0)
	}
	return out
}

// ExampleFreeSR performs single-image, example-free super-resolution at an
// integer scale. It combines the strengths of the other routines in this
// package: an edge-directed base upscale ([NEDI] for power-of-two scales,
// bicubic otherwise), iterative back-projection to enforce consistency with the
// low-resolution input, and a final gradient-profile sharpening pass to crisp
// the reconstructed edges. No external training data or example dictionary is
// used. iterations controls the back-projection refinement (0 skips it). It
// panics if scale < 2 or iterations < 0.
func ExampleFreeSR(low *cv.Mat, scale, iterations int) *cv.Mat {
	if scale < 2 {
		panic("superres: ExampleFreeSR requires scale >= 2")
	}
	if iterations < 0 {
		panic("superres: ExampleFreeSR requires iterations >= 0")
	}
	// Base estimate: edge-directed when the scale is a power of two.
	var base *cv.Mat
	if scale&(scale-1) == 0 {
		base = NEDI(low, scale)
	} else {
		base = BicubicResize(low, low.Cols*scale, low.Rows*scale)
	}
	if iterations > 0 {
		base = refineByBackProjection(low, base, scale, iterations)
	}
	return GradientProfileSharpen(base, 0.5)
}

// refineByBackProjection applies iterative back-projection starting from an
// existing high-resolution estimate rather than a fresh bicubic upscale, so it
// can polish the output of any base upscaler.
func refineByBackProjection(low, hrEstimate *cv.Mat, scale, iterations int) *cv.Mat {
	sigma := superresGaussianSigmaFor(scale)
	cubic := CatmullRomKernel()
	lowH, lowW := low.Rows, low.Cols
	hiH, hiW := hrEstimate.Rows, hrEstimate.Cols
	lowPlanes := superresSplitPlanes(low)
	hrPlanes := superresSplitPlanes(hrEstimate)
	for c := range hrPlanes {
		hr := hrPlanes[c]
		lp := lowPlanes[c]
		for it := 0; it < iterations; it++ {
			blurred := superresPlaneGaussianBlur(hr, sigma)
			sim := superresPlaneResize(blurred, lowW, lowH, cubic)
			res := newSuperresPlane(lowH, lowW)
			for i := range res.data {
				res.data[i] = lp.data[i] - sim.data[i]
			}
			resUp := superresPlaneResize(res, hiW, hiH, cubic)
			for i := range hr.data {
				hr.data[i] += resUp.data[i]
			}
		}
	}
	return superresMergePlanes(hrPlanes)
}
