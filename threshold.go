package cv

import "fmt"

// ThresholdType selects the behaviour of [Threshold]. It may be combined with
// [ThreshOtsu] using a bitwise OR to have the threshold chosen automatically.
type ThresholdType int

const (
	// ThreshBinary sets samples above the threshold to maxval, others to 0.
	ThreshBinary ThresholdType = iota
	// ThreshBinaryInv sets samples above the threshold to 0, others to maxval.
	ThreshBinaryInv
	// ThreshTrunc clamps samples above the threshold to the threshold value.
	ThreshTrunc
	// ThreshToZero zeroes samples at or below the threshold, keeps the rest.
	ThreshToZero
	// ThreshToZeroInv zeroes samples above the threshold, keeps the rest.
	ThreshToZeroInv
)

// ThreshOtsu is an OR-able flag that makes [Threshold] ignore the supplied
// level and compute an optimal global threshold with Otsu's method. It is only
// meaningful for single-channel input.
const ThreshOtsu ThresholdType = 8

// Threshold applies a fixed-level threshold to a single-channel Mat and returns
// the result together with the threshold that was used. When the [ThreshOtsu]
// flag is OR-ed into typ the level argument is ignored and Otsu's optimal value
// is returned. It panics if src is not single-channel.
func Threshold(src *Mat, thresh, maxval float64, typ ThresholdType) (*Mat, float64) {
	requireChannels(src, 1, "Threshold")
	useOtsu := typ&ThreshOtsu != 0
	base := typ &^ ThreshOtsu
	if useOtsu {
		thresh = otsuLevel(src)
	}
	dst := NewMat(src.Rows, src.Cols, 1)
	mv := clampToUint8(maxval)
	t := thresh
	for i, s := range src.Data {
		v := float64(s)
		switch base {
		case ThreshBinary:
			if v > t {
				dst.Data[i] = mv
			} else {
				dst.Data[i] = 0
			}
		case ThreshBinaryInv:
			if v > t {
				dst.Data[i] = 0
			} else {
				dst.Data[i] = mv
			}
		case ThreshTrunc:
			if v > t {
				dst.Data[i] = clampToUint8(t)
			} else {
				dst.Data[i] = s
			}
		case ThreshToZero:
			if v > t {
				dst.Data[i] = s
			} else {
				dst.Data[i] = 0
			}
		case ThreshToZeroInv:
			if v > t {
				dst.Data[i] = 0
			} else {
				dst.Data[i] = s
			}
		default:
			panic(fmt.Sprintf("cv: Threshold unknown type %d", base))
		}
	}
	return dst, thresh
}

// otsuLevel computes the Otsu threshold that maximises inter-class variance of
// a single-channel image's histogram.
func otsuLevel(src *Mat) float64 {
	var hist [256]int
	for _, s := range src.Data {
		hist[s]++
	}
	total := len(src.Data)
	var sum float64
	for i := 0; i < 256; i++ {
		sum += float64(i) * float64(hist[i])
	}
	var sumB, wB float64
	var maxVar float64
	best := 0
	for t := 0; t < 256; t++ {
		wB += float64(hist[t])
		if wB == 0 {
			continue
		}
		wF := float64(total) - wB
		if wF == 0 {
			break
		}
		sumB += float64(t) * float64(hist[t])
		mB := sumB / wB
		mF := (sum - sumB) / wF
		between := wB * wF * (mB - mF) * (mB - mF)
		if between > maxVar {
			maxVar = between
			best = t
		}
	}
	return float64(best)
}

// AdaptiveMethod selects how [AdaptiveThreshold] computes the local threshold.
type AdaptiveMethod int

const (
	// AdaptiveThreshMeanC uses the mean of the blockSize×blockSize
	// neighbourhood, minus C, as the local threshold.
	AdaptiveThreshMeanC AdaptiveMethod = iota
	// AdaptiveThreshGaussianC uses a Gaussian-weighted neighbourhood mean,
	// minus C, as the local threshold.
	AdaptiveThreshGaussianC
)

// AdaptiveThreshold thresholds a single-channel Mat using a threshold computed
// per pixel from its neighbourhood. blockSize is the odd neighbourhood size and
// C is a constant subtracted from the local mean. typ must be [ThreshBinary] or
// [ThreshBinaryInv]. It panics on invalid arguments.
func AdaptiveThreshold(src *Mat, maxval float64, method AdaptiveMethod, typ ThresholdType, blockSize int, c float64) *Mat {
	requireChannels(src, 1, "AdaptiveThreshold")
	requireOdd(blockSize, "AdaptiveThreshold")
	if typ != ThreshBinary && typ != ThreshBinaryInv {
		panic("cv: AdaptiveThreshold type must be ThreshBinary or ThreshBinaryInv")
	}
	var local [][]float64
	switch method {
	case AdaptiveThreshMeanC:
		w := 1.0 / float64(blockSize*blockSize)
		data := make([]float64, blockSize*blockSize)
		for i := range data {
			data[i] = w
		}
		local = filter2DFloat(src, NewKernel(blockSize, blockSize, data))
	case AdaptiveThreshGaussianC:
		k := GaussianKernel1D(blockSize, 0)
		local = sepFilterFloat(src, k, k)
	default:
		panic(fmt.Sprintf("cv: AdaptiveThreshold unknown method %d", method))
	}
	dst := NewMat(src.Rows, src.Cols, 1)
	mv := clampToUint8(maxval)
	for i, s := range src.Data {
		t := local[0][i] - c
		above := float64(s) > t
		if typ == ThreshBinaryInv {
			above = !above
		}
		if above {
			dst.Data[i] = mv
		} else {
			dst.Data[i] = 0
		}
	}
	return dst
}

// InRange produces a single-channel mask that is 255 where every channel of src
// lies within the inclusive [lo, hi] band and 0 elsewhere. lo and hi must have
// one entry per channel. It is handy for colour segmentation, e.g. on an HSV
// image. It panics if the bounds do not match the channel count.
func InRange(src *Mat, lo, hi []uint8) *Mat {
	if len(lo) != src.Channels || len(hi) != src.Channels {
		panic("cv: InRange bounds must have one entry per channel")
	}
	dst := NewMat(src.Rows, src.Cols, 1)
	for p := 0; p < src.Total(); p++ {
		base := p * src.Channels
		in := true
		for c := 0; c < src.Channels; c++ {
			v := src.Data[base+c]
			if v < lo[c] || v > hi[c] {
				in = false
				break
			}
		}
		if in {
			dst.Data[p] = 255
		}
	}
	return dst
}
