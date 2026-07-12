package fuzzy

import (
	"fmt"

	cv "github.com/malcolmston/opencv"
)

// FilterLinear smooths img with the degree-0 F-transform over a triangular
// ([LinearBasis]) fuzzy partition of the given radius. It is the named
// convenience form of Filter(img, LinearBasis, radius) — the LINEAR variant of
// OpenCV's ft smoothing — and works on grayscale or multi-channel images.
func FilterLinear(img *cv.Mat, radius int) *cv.Mat {
	return Filter(img, LinearBasis, radius)
}

// FilterSinus smooths img with the degree-0 F-transform over a raised-cosine
// ([SinusBasis]) fuzzy partition of the given radius — the SINUS variant of
// OpenCV's ft smoothing. The smoother basis reduces the faceting a triangular
// basis can leave on gently curved content.
func FilterSinus(img *cv.Mat, radius int) *cv.Mat {
	return Filter(img, SinusBasis, radius)
}

// FilterDegree1 smooths img with the degree-1 (linear-polynomial) F-transform
// over a fuzzy partition of the given basis function and radius. Because each
// node reconstructs a local sloped plane rather than a constant, gradients and
// ramps survive the round trip almost perfectly, so at equal radius it preserves
// structure far better than the degree-0 [Filter] while still suppressing
// high-frequency noise. It is a convenience wrapper around [FT12DProcess].
func FilterDegree1(img *cv.Mat, function BasisFunction, radius int) *cv.Mat {
	if img == nil || img.Empty() {
		panic("fuzzy: FilterDegree1 given an empty image")
	}
	if radius < 1 {
		panic(fmt.Sprintf("fuzzy: FilterDegree1 radius must be >= 1, got %d", radius))
	}
	kernel := CreateKernel(function, radius)
	c := FT12DComponents(img, kernel, nil)
	c.Function = function
	return FT12DInverse(c)
}

// FilterMultiRadius applies the degree-0 F-transform smoother at several radii
// and returns one reconstruction per radius, in the same order. It is a
// convenience for building a scale pyramid or sweeping the smoothing strength
// (each larger radius removes more high-frequency detail) without repeating the
// kernel-and-process boilerplate. radii must be non-empty and every entry must be
// >= 1.
func FilterMultiRadius(img *cv.Mat, function BasisFunction, radii []int) []*cv.Mat {
	if img == nil || img.Empty() {
		panic("fuzzy: FilterMultiRadius given an empty image")
	}
	if len(radii) == 0 {
		panic("fuzzy: FilterMultiRadius requires at least one radius")
	}
	out := make([]*cv.Mat, len(radii))
	for i, r := range radii {
		if r < 1 {
			panic(fmt.Sprintf("fuzzy: FilterMultiRadius radius[%d] must be >= 1, got %d", i, r))
		}
		out[i] = Filter(img, function, r)
	}
	return out
}
