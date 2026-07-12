package quality

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// RMSE returns the root-mean-squared error between a and b, one value per
// channel. It is simply the square root of [MSE]: lower is better and identical
// images score exactly zero. It panics unless the two images share a size and
// channel count.
func RMSE(a, b *cv.Mat) []float64 {
	requireComparable(a, b, "RMSE")
	out := MSE(a, b)
	for c := range out {
		out[c] = math.Sqrt(out[c])
	}
	return out
}

// NRMSE returns the normalised root-mean-squared error between a and b, one
// value per channel. Each channel's [RMSE] is divided by the dynamic range of
// the reference channel a (its max minus min sample value), which makes the
// score scale-independent and, for most images, bounded in [0, 1]. A channel
// whose reference is perfectly flat is normalised by the peak value
// (dynamicRange) so the result stays finite. Lower is better; identical images
// score zero. It panics unless the two images share a size and channel count.
func NRMSE(a, b *cv.Mat) []float64 {
	requireComparable(a, b, "NRMSE")
	ch := a.Channels
	rmse := RMSE(a, b)
	n := a.Total()
	for c := 0; c < ch; c++ {
		mn, mx := 255.0, 0.0
		for p := 0; p < n; p++ {
			v := float64(a.Data[p*ch+c])
			if v < mn {
				mn = v
			}
			if v > mx {
				mx = v
			}
		}
		rng := mx - mn
		if rng <= 0 {
			rng = dynamicRange
		}
		rmse[c] /= rng
	}
	return rmse
}

// SNR returns the signal-to-noise ratio between reference a and candidate b in
// decibels, pooling over every channel. The signal power is the mean squared
// reference sample and the noise power is the pooled [MSE]; SNR is ten times
// their log-ratio. Higher is better and identical images yield +Inf (zero
// noise). It panics unless the two images share a size and channel count.
func SNR(a, b *cv.Mat) float64 {
	requireComparable(a, b, "SNR")
	ch := a.Channels
	n := a.Total()
	var signal, noise float64
	for p := 0; p < n; p++ {
		base := p * ch
		for c := 0; c < ch; c++ {
			av := float64(a.Data[base+c])
			d := av - float64(b.Data[base+c])
			signal += av * av
			noise += d * d
		}
	}
	if noise == 0 {
		return math.Inf(1)
	}
	if signal == 0 {
		return math.Inf(-1)
	}
	return 10 * math.Log10(signal/noise)
}
