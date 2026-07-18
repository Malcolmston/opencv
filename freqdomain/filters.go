package freqdomain

import (
	"fmt"
	"math"

	cv "github.com/malcolmston/opencv"
)

// centerDistance returns the Euclidean distance from pixel (y,x) to the centre
// (rows/2, cols/2) of a rows×cols centred spectrum.
func centerDistance(y, x, rows, cols int) float64 {
	cy := float64(rows / 2)
	cx := float64(cols / 2)
	dy := float64(y) - cy
	dx := float64(x) - cx
	return math.Hypot(dy, dx)
}

// buildFilter allocates a centred rows×cols transfer function whose value at
// each pixel is fn(D), where D is the distance from the spectrum centre.
func buildFilter(rows, cols int, fn func(d float64) float64) *cv.FloatMat {
	if rows <= 0 || cols <= 0 {
		panic(fmt.Sprintf("freqdomain: filter requires positive size, got %dx%d", rows, cols))
	}
	out := cv.NewFloatMat(rows, cols)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			out.Data[y*cols+x] = fn(centerDistance(y, x, rows, cols))
		}
	}
	return out
}

// IdealLowPass returns a centred ideal low-pass transfer function of size
// rows×cols: 1 where the distance from the centre is at most cutoff and 0
// beyond it. It panics if cutoff is negative.
func IdealLowPass(rows, cols int, cutoff float64) *cv.FloatMat {
	requireCutoff(cutoff, "IdealLowPass")
	return buildFilter(rows, cols, func(d float64) float64 {
		if d <= cutoff {
			return 1
		}
		return 0
	})
}

// IdealHighPass returns a centred ideal high-pass transfer function, the
// complement of [IdealLowPass]: 0 within cutoff of the centre and 1 beyond it.
// It panics if cutoff is negative.
func IdealHighPass(rows, cols int, cutoff float64) *cv.FloatMat {
	requireCutoff(cutoff, "IdealHighPass")
	return buildFilter(rows, cols, func(d float64) float64 {
		if d <= cutoff {
			return 0
		}
		return 1
	})
}

// IdealBandPass returns a centred ideal band-pass transfer function: 1 where the
// distance from the centre lies in [low, high] and 0 elsewhere. It panics
// unless 0 <= low <= high.
func IdealBandPass(rows, cols int, low, high float64) *cv.FloatMat {
	requireBand(low, high, "IdealBandPass")
	return buildFilter(rows, cols, func(d float64) float64 {
		if d >= low && d <= high {
			return 1
		}
		return 0
	})
}

// IdealBandReject returns a centred ideal band-reject (band-stop) transfer
// function, the complement of [IdealBandPass]: 0 where the distance lies in
// [low, high] and 1 elsewhere. It panics unless 0 <= low <= high.
func IdealBandReject(rows, cols int, low, high float64) *cv.FloatMat {
	requireBand(low, high, "IdealBandReject")
	return buildFilter(rows, cols, func(d float64) float64 {
		if d >= low && d <= high {
			return 0
		}
		return 1
	})
}

// ButterworthLowPass returns a centred Butterworth low-pass transfer function
// H(D) = 1 / (1 + (D/cutoff)^(2·order)). Larger order sharpens the transition
// toward the ideal filter. It panics unless cutoff > 0 and order > 0.
func ButterworthLowPass(rows, cols int, cutoff float64, order int) *cv.FloatMat {
	requirePositiveCutoff(cutoff, "ButterworthLowPass")
	requireOrder(order, "ButterworthLowPass")
	n2 := 2.0 * float64(order)
	return buildFilter(rows, cols, func(d float64) float64 {
		return 1 / (1 + math.Pow(d/cutoff, n2))
	})
}

// ButterworthHighPass returns a centred Butterworth high-pass transfer function
// H(D) = 1 / (1 + (cutoff/D)^(2·order)), with H(0)=0. It panics unless
// cutoff > 0 and order > 0.
func ButterworthHighPass(rows, cols int, cutoff float64, order int) *cv.FloatMat {
	requirePositiveCutoff(cutoff, "ButterworthHighPass")
	requireOrder(order, "ButterworthHighPass")
	n2 := 2.0 * float64(order)
	return buildFilter(rows, cols, func(d float64) float64 {
		if d == 0 {
			return 0
		}
		return 1 / (1 + math.Pow(cutoff/d, n2))
	})
}

// ButterworthBandReject returns a centred Butterworth band-reject transfer
// function of centre frequency center and bandwidth width, following Gonzalez:
// H(D) = 1 / (1 + ((D·width) / (D²−center²))^(2·order)). It panics unless
// center > 0, width > 0 and order > 0.
func ButterworthBandReject(rows, cols int, center, width float64, order int) *cv.FloatMat {
	requirePositiveCutoff(center, "ButterworthBandReject")
	requirePositiveCutoff(width, "ButterworthBandReject")
	requireOrder(order, "ButterworthBandReject")
	n2 := 2.0 * float64(order)
	return buildFilter(rows, cols, func(d float64) float64 {
		den := d*d - center*center
		if den == 0 {
			return 0
		}
		return 1 / (1 + math.Pow(d*width/den, n2))
	})
}

// ButterworthBandPass returns a centred Butterworth band-pass transfer function,
// the complement of [ButterworthBandReject]. It panics unless center > 0,
// width > 0 and order > 0.
func ButterworthBandPass(rows, cols int, center, width float64, order int) *cv.FloatMat {
	rej := ButterworthBandReject(rows, cols, center, width, order)
	return complementFilter(rej)
}

// GaussianLowPass returns a centred Gaussian low-pass transfer function
// H(D) = exp(−D² / (2·cutoff²)). The cutoff equals the standard deviation of
// the Gaussian in the frequency domain. It panics unless cutoff > 0.
func GaussianLowPass(rows, cols int, cutoff float64) *cv.FloatMat {
	requirePositiveCutoff(cutoff, "GaussianLowPass")
	denom := 2 * cutoff * cutoff
	return buildFilter(rows, cols, func(d float64) float64 {
		return math.Exp(-(d * d) / denom)
	})
}

// GaussianHighPass returns a centred Gaussian high-pass transfer function
// H(D) = 1 − exp(−D² / (2·cutoff²)), the complement of [GaussianLowPass]. It
// panics unless cutoff > 0.
func GaussianHighPass(rows, cols int, cutoff float64) *cv.FloatMat {
	requirePositiveCutoff(cutoff, "GaussianHighPass")
	denom := 2 * cutoff * cutoff
	return buildFilter(rows, cols, func(d float64) float64 {
		return 1 - math.Exp(-(d*d)/denom)
	})
}

// GaussianBandReject returns a centred Gaussian band-reject transfer function of
// centre frequency center and bandwidth width, following Gonzalez:
// H(D) = 1 − exp(−((D²−center²) / (D·width))²), with H(0)=1. It panics unless
// center > 0 and width > 0.
func GaussianBandReject(rows, cols int, center, width float64) *cv.FloatMat {
	requirePositiveCutoff(center, "GaussianBandReject")
	requirePositiveCutoff(width, "GaussianBandReject")
	return buildFilter(rows, cols, func(d float64) float64 {
		if d == 0 {
			return 1
		}
		t := (d*d - center*center) / (d * width)
		return 1 - math.Exp(-t*t)
	})
}

// GaussianBandPass returns a centred Gaussian band-pass transfer function, the
// complement of [GaussianBandReject]. It panics unless center > 0 and
// width > 0.
func GaussianBandPass(rows, cols int, center, width float64) *cv.FloatMat {
	rej := GaussianBandReject(rows, cols, center, width)
	return complementFilter(rej)
}

// complementFilter returns 1−H for a real transfer function H.
func complementFilter(h *cv.FloatMat) *cv.FloatMat {
	out := cv.NewFloatMat(h.Rows, h.Cols)
	for i, v := range h.Data {
		out.Data[i] = 1 - v
	}
	return out
}

func requireCutoff(cutoff float64, name string) {
	if cutoff < 0 {
		panic("freqdomain: " + name + ": cutoff must be non-negative")
	}
}

func requirePositiveCutoff(cutoff float64, name string) {
	if cutoff <= 0 {
		panic("freqdomain: " + name + ": cutoff must be positive")
	}
}

func requireOrder(order int, name string) {
	if order <= 0 {
		panic("freqdomain: " + name + ": order must be positive")
	}
}

func requireBand(low, high float64, name string) {
	if low < 0 || high < low {
		panic("freqdomain: " + name + ": require 0 <= low <= high")
	}
}
