package histogram2

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// histogram2stretchLUT builds a 256-entry LUT that linearly maps the input
// range [lo, hi] onto [0, 255], clamping values outside the range. A
// degenerate range (hi <= lo) yields a threshold-like step at lo.
func histogram2stretchLUT(lo, hi int) [256]uint8 {
	var lut [256]uint8
	if hi <= lo {
		for i := 0; i < 256; i++ {
			if i < lo {
				lut[i] = 0
			} else {
				lut[i] = 255
			}
		}
		return lut
	}
	scale := 255.0 / float64(hi-lo)
	for i := 0; i < 256; i++ {
		if i <= lo {
			lut[i] = 0
		} else if i >= hi {
			lut[i] = 255
		} else {
			lut[i] = histogram2clampByte(float64(i-lo) * scale)
		}
	}
	return lut
}

// ContrastStretchRange linearly rescales every channel of src so that the input
// intensity range [inLow, inHigh] is mapped onto the full [0,255] range, with
// values outside the range clamped. It returns [ErrEmptyImage] if src is empty
// and [ErrInvalidArgument] if inHigh is not greater than inLow.
func ContrastStretchRange(src *cv.Mat, inLow, inHigh uint8) (*cv.Mat, error) {
	if src.Empty() {
		return nil, ErrEmptyImage
	}
	if inHigh <= inLow {
		return nil, ErrInvalidArgument
	}
	lut := histogram2stretchLUT(int(inLow), int(inHigh))
	dst := cv.NewMat(src.Rows, src.Cols, src.Channels)
	for i, s := range src.Data {
		dst.Data[i] = lut[s]
	}
	return dst, nil
}

// MinMaxStretch linearly rescales each channel of src independently so that its
// observed minimum maps to 0 and its maximum to 255 (full-scale contrast
// stretching). It returns [ErrEmptyImage] if src is empty.
func MinMaxStretch(src *cv.Mat) (*cv.Mat, error) {
	if src.Empty() {
		return nil, ErrEmptyImage
	}
	ch := src.Channels
	dst := cv.NewMat(src.Rows, src.Cols, ch)
	total := src.Total()
	for c := 0; c < ch; c++ {
		lo, hi := 255, 0
		for p := 0; p < total; p++ {
			v := int(src.Data[p*ch+c])
			if v < lo {
				lo = v
			}
			if v > hi {
				hi = v
			}
		}
		lut := histogram2stretchLUT(lo, hi)
		for p := 0; p < total; p++ {
			dst.Data[p*ch+c] = lut[src.Data[p*ch+c]]
		}
	}
	return dst, nil
}

// ContrastStretch rescales each channel of src independently so that the
// intensity at the lowPct percentile maps to 0 and the intensity at the highPct
// percentile maps to 255, clamping the tails. Percentiles are given in [0,100]
// and lowPct must be strictly below highPct. This is robust percentile-based
// contrast stretching. It returns [ErrEmptyImage] if src is empty and
// [ErrInvalidArgument] if the percentiles are out of range or misordered.
func ContrastStretch(src *cv.Mat, lowPct, highPct float64) (*cv.Mat, error) {
	if src.Empty() {
		return nil, ErrEmptyImage
	}
	if lowPct < 0 || highPct > 100 || !(lowPct < highPct) {
		return nil, ErrInvalidArgument
	}
	ch := src.Channels
	total := src.Total()
	dst := cv.NewMat(src.Rows, src.Cols, ch)
	for c := 0; c < ch; c++ {
		var hist [256]int
		for p := 0; p < total; p++ {
			hist[src.Data[p*ch+c]]++
		}
		loCount := lowPct / 100 * float64(total)
		hiCount := highPct / 100 * float64(total)
		lo, hi := 0, 255
		acc := 0
		foundLo := false
		for i := 0; i < 256; i++ {
			acc += hist[i]
			// lo is the first level whose cumulative count exceeds the low
			// threshold, so a 0% percentile selects the minimum present value.
			if !foundLo && float64(acc) > loCount {
				lo = i
				foundLo = true
			}
			if float64(acc) >= hiCount {
				hi = i
				break
			}
		}
		lut := histogram2stretchLUT(lo, hi)
		for p := 0; p < total; p++ {
			dst.Data[p*ch+c] = lut[src.Data[p*ch+c]]
		}
	}
	return dst, nil
}

// GammaCorrect applies a power-law (gamma) transform to every sample of src:
// out = 255 * (in/255)^gamma. Gamma below 1 brightens, above 1 darkens. It
// returns [ErrEmptyImage] if src is empty and [ErrInvalidArgument] if gamma is
// not positive.
func GammaCorrect(src *cv.Mat, gamma float64) (*cv.Mat, error) {
	if src.Empty() {
		return nil, ErrEmptyImage
	}
	if gamma <= 0 {
		return nil, ErrInvalidArgument
	}
	var lut [256]uint8
	for i := 0; i < 256; i++ {
		lut[i] = histogram2clampByte(math.Pow(float64(i)/255, gamma) * 255)
	}
	dst := cv.NewMat(src.Rows, src.Cols, src.Channels)
	for i, s := range src.Data {
		dst.Data[i] = lut[s]
	}
	return dst, nil
}
