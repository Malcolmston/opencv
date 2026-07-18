package threshold2

import (
	"errors"

	cv "github.com/malcolmston/opencv"
)

// PerChannelOtsu thresholds each channel of src independently with Otsu's
// method and returns an image of the same shape whose samples are 0 or 255,
// together with the per-channel thresholds. For a three-channel image the
// result is a three-channel mask, one Otsu threshold per channel.
func PerChannelOtsu(src *cv.Mat, p Polarity) (*cv.Mat, []int, error) {
	if src.Empty() {
		return nil, nil, ErrEmpty
	}
	ch := src.Channels
	thresholds := make([]int, ch)
	for c := 0; c < ch; c++ {
		var bins [256]int
		for px := 0; px < src.Rows*src.Cols; px++ {
			bins[src.Data[px*ch+c]]++
		}
		thresholds[c] = otsuFromBins(bins, src.Rows*src.Cols)
	}
	dst := threshold2applyPerChannel(src, thresholds, p)
	return dst, thresholds, nil
}

// PerChannelThreshold binarizes each channel of src with its own grey level
// from the thresholds slice, whose length must equal the channel count. The
// result has the same shape as src with samples 0 or 255. Foreground selection
// follows p.
func PerChannelThreshold(src *cv.Mat, thresholds []int, p Polarity) (*cv.Mat, error) {
	if src.Empty() {
		return nil, ErrEmpty
	}
	if len(thresholds) != src.Channels {
		return nil, errors.New("threshold2: PerChannelThreshold needs one threshold per channel")
	}
	return threshold2applyPerChannel(src, thresholds, p), nil
}

// threshold2applyPerChannel binarizes each channel of src using thresholds[c].
func threshold2applyPerChannel(src *cv.Mat, thresholds []int, p Polarity) *cv.Mat {
	ch := src.Channels
	dst := cv.NewMat(src.Rows, src.Cols, ch)
	for px := 0; px < src.Rows*src.Cols; px++ {
		for c := 0; c < ch; c++ {
			i := px*ch + c
			fg := int(src.Data[i]) > thresholds[c]
			if p == ObjectDark {
				fg = int(src.Data[i]) <= thresholds[c]
			}
			if fg {
				dst.Data[i] = 255
			}
		}
	}
	return dst
}

// InRange produces a single-channel mask that is 255 where every channel of
// src lies within the inclusive range [lower[c], upper[c]] and 0 elsewhere,
// mirroring OpenCV's cv::inRange. Both bound slices must have one entry per
// channel.
func InRange(src *cv.Mat, lower, upper []uint8) (*cv.Mat, error) {
	if src.Empty() {
		return nil, ErrEmpty
	}
	ch := src.Channels
	if len(lower) != ch || len(upper) != ch {
		return nil, errors.New("threshold2: InRange bounds must have one entry per channel")
	}
	dst := cv.NewMat(src.Rows, src.Cols, 1)
	for px := 0; px < src.Rows*src.Cols; px++ {
		inside := true
		for c := 0; c < ch; c++ {
			v := src.Data[px*ch+c]
			if v < lower[c] || v > upper[c] {
				inside = false
				break
			}
		}
		if inside {
			dst.Data[px] = 255
		}
	}
	return dst, nil
}
