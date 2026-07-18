package morph2

import cv "github.com/malcolmston/opencv"

// PatternSpectrum holds the result of a granulometric analysis: how the image
// volume is distributed across structuring-element sizes. It is produced by
// [Granulometry].
type PatternSpectrum struct {
	// Sizes are the structuring-element radii examined, in increasing order,
	// starting at 0 (the identity opening).
	Sizes []int
	// Volumes[i] is the total sample volume (sum of all samples) of the opening
	// of the source by the element of radius Sizes[i]; it is non-increasing.
	Volumes []float64
	// Spectrum[i] is the normalised volume lost between radius Sizes[i] and
	// Sizes[i+1], i.e. the fraction of image content removed by that size class.
	// It has length len(Sizes)-1 and sums to at most 1.
	Spectrum []float64
}

// Volume returns the total sample volume of the unopened source (Volumes[0]).
func (ps *PatternSpectrum) Volume() float64 {
	if len(ps.Volumes) == 0 {
		return 0
	}
	return ps.Volumes[0]
}

// Mean returns the volume-weighted mean structuring-element size of the pattern
// spectrum, a scalar summary of the characteristic object size in the image. It
// returns 0 when the spectrum is empty.
func (ps *PatternSpectrum) Mean() float64 {
	var num, den float64
	for i, s := range ps.Spectrum {
		size := float64(ps.Sizes[i+1])
		num += size * s
		den += s
	}
	if den == 0 {
		return 0
	}
	return num / den
}

// Granulometry performs a granulometric analysis of a grey-scale or binary
// image by opening it with structuring elements of the given shape and
// increasing radius, from 0 up to maxRadius in the given step, and recording
// the residual volume at each size. The differences between successive volumes
// form the pattern spectrum, which characterises the size distribution of the
// bright structures in the image.
//
// maxRadius must be >= 1 and step >= 1. It panics on multi-channel input or an
// invalid parameter.
func Granulometry(src *cv.Mat, shape Shape, maxRadius, step int) *PatternSpectrum {
	requireGray(src)
	if maxRadius < 1 {
		panic("morph2: Granulometry requires maxRadius >= 1")
	}
	if step < 1 {
		panic("morph2: Granulometry requires step >= 1")
	}
	var sizes []int
	for r := 0; r <= maxRadius; r += step {
		sizes = append(sizes, r)
	}
	vols := make([]float64, len(sizes))
	for i, r := range sizes {
		e := NewElement(shape, 2*r+1, 2*r+1)
		opened := Open(src, e)
		vols[i] = volume(opened)
	}
	spec := make([]float64, 0, len(sizes)-1)
	base := vols[0]
	for i := 0; i+1 < len(sizes); i++ {
		diff := vols[i] - vols[i+1]
		if base > 0 {
			spec = append(spec, diff/base)
		} else {
			spec = append(spec, 0)
		}
	}
	return &PatternSpectrum{Sizes: sizes, Volumes: vols, Spectrum: spec}
}

// volume returns the sum of all samples of a single-channel image.
func volume(m *cv.Mat) float64 {
	var s float64
	for _, v := range m.Data {
		s += float64(v)
	}
	return s
}
