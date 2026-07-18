package superres

import cv "github.com/malcolmston/opencv"

// BackProjectionParams configures [IterativeBackProjection]. Its zero value is
// not valid; obtain a sensible starting point from [DefaultBackProjectionParams]
// and adjust as needed.
type BackProjectionParams struct {
	// Scale is the integer super-resolution factor (>= 2).
	Scale int
	// Iterations is the number of back-projection updates to perform.
	Iterations int
	// Lambda is the step size applied to the back-projected residual each
	// iteration (0 < Lambda <= 1 is typical). Smaller values converge more
	// slowly but more stably.
	Lambda float64
	// BlurSigma is the standard deviation of the Gaussian point-spread
	// function used to model the imaging blur; if <= 0 a scale-dependent
	// default is used.
	BlurSigma float64
}

// DefaultBackProjectionParams returns default back-projection parameters for
// the given integer scale: eight iterations, a step size of 1, and a
// scale-matched Gaussian blur.
func DefaultBackProjectionParams(scale int) BackProjectionParams {
	return BackProjectionParams{
		Scale:      scale,
		Iterations: 8,
		Lambda:     1.0,
		BlurSigma:  superresGaussianSigmaFor(scale),
	}
}

// IterativeBackProjection performs Irani-Peleg iterative back-projection
// super-resolution on a single low-resolution image. It forms an initial
// high-resolution estimate by bicubic upscaling, then repeatedly simulates the
// imaging process (Gaussian blur followed by decimation), compares the
// simulated low-resolution image with the observed input, and back-projects the
// upscaled residual into the estimate. The result is sharper and more faithful
// to the imaging model than plain interpolation. It panics if params.Scale < 2
// or params.Iterations < 0.
func IterativeBackProjection(low *cv.Mat, params BackProjectionParams) *cv.Mat {
	if params.Scale < 2 {
		panic("superres: IterativeBackProjection requires Scale >= 2")
	}
	if params.Iterations < 0 {
		panic("superres: IterativeBackProjection requires Iterations >= 0")
	}
	sigma := params.BlurSigma
	if sigma <= 0 {
		sigma = superresGaussianSigmaFor(params.Scale)
	}
	lambda := params.Lambda
	if lambda <= 0 {
		lambda = 1
	}
	scale := params.Scale
	cubic := CatmullRomKernel()
	lowH, lowW := low.Rows, low.Cols
	hiH, hiW := lowH*scale, lowW*scale

	lowPlanes := superresSplitPlanes(low)
	outPlanes := make([]*superresPlane, len(lowPlanes))
	for c, lp := range lowPlanes {
		// Initial estimate: bicubic upscale.
		hr := superresPlaneResize(lp, hiW, hiH, cubic)
		for it := 0; it < params.Iterations; it++ {
			// Simulate imaging: blur then decimate to low resolution.
			blurred := superresPlaneGaussianBlur(hr, sigma)
			sim := superresPlaneResize(blurred, lowW, lowH, cubic)
			// Residual in low resolution.
			res := newSuperresPlane(lowH, lowW)
			for i := range res.data {
				res.data[i] = lp.data[i] - sim.data[i]
			}
			// Back-project: upscale the residual and add.
			resUp := superresPlaneResize(res, hiW, hiH, cubic)
			for i := range hr.data {
				hr.data[i] += lambda * resUp.data[i]
			}
		}
		outPlanes[c] = hr
	}
	return superresMergePlanes(outPlanes)
}

// BackProjectionSR is a convenience wrapper that runs [IterativeBackProjection]
// with [DefaultBackProjectionParams] for the given integer scale and iteration
// count. It panics if scale < 2 or iterations < 0.
func BackProjectionSR(low *cv.Mat, scale, iterations int) *cv.Mat {
	p := DefaultBackProjectionParams(scale)
	p.Iterations = iterations
	return IterativeBackProjection(low, p)
}
