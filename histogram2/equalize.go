package histogram2

import cv "github.com/malcolmston/opencv"

// histogram2clampByte rounds and clamps a floating value to the [0,255] byte
// range.
func histogram2clampByte(v float64) uint8 {
	v += 0.5
	if v < 0 {
		return 0
	}
	if v > 255 {
		return 255
	}
	return uint8(v)
}

// histogram2equalizeLUT builds the 256-entry equalisation lookup table for a
// 256-bin intensity histogram covering total pixels, using the standard
// CDF-min normalisation. A degenerate (single-value) image yields the identity
// mapping.
func histogram2equalizeLUT(hist [256]int, total int) [256]uint8 {
	var cdf [256]int
	acc := 0
	cdfMin := 0
	for i := 0; i < 256; i++ {
		acc += hist[i]
		cdf[i] = acc
		if cdfMin == 0 && acc > 0 {
			cdfMin = acc
		}
	}
	var lut [256]uint8
	denom := total - cdfMin
	if denom <= 0 {
		for i := range lut {
			lut[i] = uint8(i)
		}
		return lut
	}
	for i := 0; i < 256; i++ {
		v := float64(cdf[i]-cdfMin) / float64(denom) * 255
		lut[i] = histogram2clampByte(v)
	}
	return lut
}

// histogram2channelHist counts the 256-bin intensity histogram of a single
// channel of src.
func histogram2channelHist(src *cv.Mat, channel int) [256]int {
	var hist [256]int
	total := src.Total()
	for p := 0; p < total; p++ {
		hist[src.Data[p*src.Channels+channel]]++
	}
	return hist
}

// EqualizeHist performs global histogram equalisation on a single-channel
// image, spreading its intensities across the full [0,255] range to improve
// contrast. It returns [ErrEmptyImage] if src is empty and [ErrChannelRange]
// if src is not single-channel.
func EqualizeHist(src *cv.Mat) (*cv.Mat, error) {
	if src.Empty() {
		return nil, ErrEmptyImage
	}
	if src.Channels != 1 {
		return nil, ErrChannelRange
	}
	lut := histogram2equalizeLUT(histogram2channelHist(src, 0), src.Total())
	dst := cv.NewMat(src.Rows, src.Cols, 1)
	for i, s := range src.Data {
		dst.Data[i] = lut[s]
	}
	return dst, nil
}

// EqualizeHistPerChannel equalises every channel of src independently and
// returns an image of the same shape. For a colour image this alters hue as a
// side effect; use [EqualizeHistLuminance] to preserve colour. It returns
// [ErrEmptyImage] if src is empty.
func EqualizeHistPerChannel(src *cv.Mat) (*cv.Mat, error) {
	if src.Empty() {
		return nil, ErrEmptyImage
	}
	ch := src.Channels
	luts := make([][256]uint8, ch)
	for c := 0; c < ch; c++ {
		luts[c] = histogram2equalizeLUT(histogram2channelHist(src, c), src.Total())
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

// EqualizeHistLuminance equalises the luminance of a three-channel RGB image
// while preserving its colour ratios: it computes the ITU-R luma of each pixel,
// equalises that single channel and rescales the R, G and B samples by the
// ratio of the new to the old luma. It returns [ErrEmptyImage] if src is empty
// and [ErrChannelRange] if src does not have three channels.
func EqualizeHistLuminance(src *cv.Mat) (*cv.Mat, error) {
	if src.Empty() {
		return nil, ErrEmptyImage
	}
	if src.Channels != 3 {
		return nil, ErrChannelRange
	}
	total := src.Total()
	luma := make([]uint8, total)
	var hist [256]int
	for p := 0; p < total; p++ {
		base := p * 3
		r := float64(src.Data[base])
		g := float64(src.Data[base+1])
		b := float64(src.Data[base+2])
		y := histogram2clampByte(0.299*r + 0.587*g + 0.114*b)
		luma[p] = y
		hist[y]++
	}
	lut := histogram2equalizeLUT(hist, total)
	dst := cv.NewMat(src.Rows, src.Cols, 3)
	for p := 0; p < total; p++ {
		base := p * 3
		oldY := luma[p]
		newY := lut[oldY]
		if oldY == 0 {
			// Achromatic black: assign the new luma to all channels.
			dst.Data[base] = newY
			dst.Data[base+1] = newY
			dst.Data[base+2] = newY
			continue
		}
		scale := float64(newY) / float64(oldY)
		dst.Data[base] = histogram2clampByte(float64(src.Data[base]) * scale)
		dst.Data[base+1] = histogram2clampByte(float64(src.Data[base+1]) * scale)
		dst.Data[base+2] = histogram2clampByte(float64(src.Data[base+2]) * scale)
	}
	return dst, nil
}
