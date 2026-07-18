package histogram2

import cv "github.com/malcolmston/opencv"

// BuildSpecificationLUT builds a lookup table that maps each bin of srcHist to
// an output intensity in [0,255] such that applying it drives the source
// distribution toward the shape of refHist. It works by cumulative-distribution
// matching: for each source bin it finds the reference bin whose cumulative
// mass first reaches the source bin's cumulative mass. The returned slice has
// length srcHist.BinCount. Empty histograms yield an identity-like ramp.
func BuildSpecificationLUT(srcHist, refHist *Histogram1D) []uint8 {
	srcCDF := CumulativeDistribution(srcHist)
	refCDF := CumulativeDistribution(refHist)
	nr := refHist.BinCount

	lut := make([]uint8, srcHist.BinCount)
	// Precompute the output intensity for each reference bin.
	refVal := func(j int) uint8 {
		if nr <= 1 {
			return 0
		}
		return histogram2clampByte(float64(j) * 255 / float64(nr-1))
	}

	j := 0
	for i := 0; i < srcHist.BinCount; i++ {
		target := srcCDF[i]
		// Advance the reference bin until its CDF reaches the target.
		for j < nr-1 && refCDF[j] < target {
			j++
		}
		lut[i] = refVal(j)
	}
	return lut
}

// SpecifyHistogram remaps a single-channel image so that its intensity
// distribution approximates the target histogram (histogram specification). It
// returns [ErrEmptyImage] if src is empty and [ErrChannelRange] if src is not
// single-channel.
func SpecifyHistogram(src *cv.Mat, target *Histogram1D) (*cv.Mat, error) {
	if src.Empty() {
		return nil, ErrEmptyImage
	}
	if src.Channels != 1 {
		return nil, ErrChannelRange
	}
	srcHist, err := CalcHist1D(src, 0, 256)
	if err != nil {
		return nil, err
	}
	lut := BuildSpecificationLUT(srcHist, target)
	dst := cv.NewMat(src.Rows, src.Cols, 1)
	for i, s := range src.Data {
		dst.Data[i] = lut[s]
	}
	return dst, nil
}

// MatchHistograms remaps a single-channel source image so that its intensity
// distribution matches that of a single-channel reference image. It returns
// [ErrEmptyImage] if either image is empty and [ErrChannelRange] if either is
// not single-channel.
func MatchHistograms(src, reference *cv.Mat) (*cv.Mat, error) {
	if src.Empty() || reference.Empty() {
		return nil, ErrEmptyImage
	}
	if src.Channels != 1 || reference.Channels != 1 {
		return nil, ErrChannelRange
	}
	refHist, err := CalcHist1D(reference, 0, 256)
	if err != nil {
		return nil, err
	}
	return SpecifyHistogram(src, refHist)
}

// MatchHistogramsColor remaps each channel of a source image so that its
// per-channel intensity distribution matches the corresponding channel of a
// reference image. The two images must have the same channel count. It returns
// [ErrEmptyImage] if either image is empty and [ErrSizeMismatch] if the channel
// counts differ.
func MatchHistogramsColor(src, reference *cv.Mat) (*cv.Mat, error) {
	if src.Empty() || reference.Empty() {
		return nil, ErrEmptyImage
	}
	if src.Channels != reference.Channels {
		return nil, ErrSizeMismatch
	}
	ch := src.Channels
	luts := make([][]uint8, ch)
	for c := 0; c < ch; c++ {
		srcHist, err := CalcHist1D(src, c, 256)
		if err != nil {
			return nil, err
		}
		refHist, err := CalcHist1D(reference, c, 256)
		if err != nil {
			return nil, err
		}
		luts[c] = BuildSpecificationLUT(srcHist, refHist)
	}
	dst := cv.NewMat(src.Rows, src.Cols, ch)
	for p := 0; p < src.Total(); p++ {
		base := p * ch
		for c := 0; c < ch; c++ {
			dst.Data[base+c] = luts[c][src.Data[base+c]]
		}
	}
	return dst, nil
}
