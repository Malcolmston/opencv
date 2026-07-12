package xphoto

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// WhiteBalancer is the common interface implemented by every white-balance
// algorithm in this package, mirroring OpenCV's cv::xphoto::WhiteBalancer.
// BalanceWhite returns a new colour-corrected copy of src; it never mutates the
// input.
type WhiteBalancer interface {
	// BalanceWhite returns a white-balanced copy of src. Implementations
	// require a three-channel RGB image and panic otherwise.
	BalanceWhite(src *cv.Mat) *cv.Mat
}

// Compile-time assertions that the concrete types satisfy the interface.
var (
	_ WhiteBalancer = (*SimpleWB)(nil)
	_ WhiteBalancer = (*GrayworldWB)(nil)
	_ WhiteBalancer = (*LearningBasedWB)(nil)
)

// SimpleWB implements a simple per-channel white balance: each channel is
// stretched independently so that, after discarding the darkest and brightest
// P percent of its samples, the surviving range [inputMin,inputMax] maps
// linearly onto [outputMin,outputMax]. It ports cv::xphoto::SimpleWB.
//
// The zero value is not recommended; construct with [NewSimpleWB], which fills
// in OpenCV's defaults, then override fields as needed.
type SimpleWB struct {
	// InputMin and InputMax bound the samples considered when building each
	// channel histogram. Samples outside the range are ignored.
	InputMin, InputMax float64
	// OutputMin and OutputMax are the target range the stretched channel is
	// mapped into.
	OutputMin, OutputMax float64
	// P is the percentage (0..50) of the lowest and highest samples clipped
	// from each channel before computing its stretch bounds.
	P float64
}

// NewSimpleWB returns a SimpleWB with OpenCV's default parameters: input and
// output ranges of [0,255] and a 2% clip on each tail.
func NewSimpleWB() *SimpleWB {
	return &SimpleWB{InputMin: 0, InputMax: 255, OutputMin: 0, OutputMax: 255, P: 2.0}
}

// BalanceWhite stretches each channel of src independently. src must be a
// three-channel RGB image.
func (s *SimpleWB) BalanceWhite(src *cv.Mat) *cv.Mat {
	requireNonEmpty(src, "SimpleWB.BalanceWhite")
	requireChannels(src, 3, "SimpleWB.BalanceWhite")

	inMin, inMax := s.InputMin, s.InputMax
	if inMax <= inMin {
		inMin, inMax = 0, 255
	}
	outMin, outMax := s.OutputMin, s.OutputMax
	if outMax == outMin {
		outMin, outMax = 0, 255
	}
	p := s.P
	if p < 0 {
		p = 0
	}
	if p > 50 {
		p = 50
	}

	dst := cv.NewMat(src.Rows, src.Cols, 3)
	total := src.Total()
	for c := 0; c < 3; c++ {
		// Build a 256-bin histogram over the input range.
		var hist [256]int
		var n int
		for p2 := 0; p2 < total; p2++ {
			v := float64(src.Data[p2*3+c])
			if v < inMin || v > inMax {
				continue
			}
			hist[src.Data[p2*3+c]]++
			n++
		}
		lo, hi := 0.0, 255.0
		if n > 0 {
			clip := int(math.Round(p / 100.0 * float64(n)))
			lo = percentile(&hist, clip, true)
			hi = percentile(&hist, clip, false)
		}
		if hi <= lo {
			// Degenerate (near-constant) channel: fall back to the input range
			// so the mapping stays well defined.
			lo, hi = inMin, inMax
			if hi <= lo {
				hi = lo + 1
			}
		}
		scale := (outMax - outMin) / (hi - lo)
		for p2 := 0; p2 < total; p2++ {
			v := float64(src.Data[p2*3+c])
			out := (v-lo)*scale + outMin
			dst.Data[p2*3+c] = clampU8(out)
		}
	}
	return dst
}

// percentile walks a 256-bin histogram and returns the bin index at which the
// cumulative count first exceeds clip samples. When fromLow is true it scans
// upward from bin 0 (lower bound); otherwise it scans downward from bin 255
// (upper bound).
func percentile(hist *[256]int, clip int, fromLow bool) float64 {
	acc := 0
	if fromLow {
		for i := 0; i < 256; i++ {
			acc += hist[i]
			if acc > clip {
				return float64(i)
			}
		}
		return 0
	}
	for i := 255; i >= 0; i-- {
		acc += hist[i]
		if acc > clip {
			return float64(i)
		}
	}
	return 255
}

// GrayworldWB implements the gray-world white-balance assumption: the average
// colour of a scene is achromatic (gray), so each channel is scaled by
// grayMean/channelMean to equalise the channel means. Pixels that are close to
// saturation are excluded from the statistics to avoid clipped highlights
// biasing the estimate. It ports cv::xphoto::GrayworldWB.
type GrayworldWB struct {
	// SaturationThreshold is in (0,1]. A pixel is excluded from the mean
	// computation when its per-pixel saturation (max-min)/max exceeds this
	// value. A value of 1 includes every pixel.
	SaturationThreshold float64
}

// NewGrayworldWB returns a GrayworldWB with a default saturation threshold of
// 0.98 (OpenCV's default).
func NewGrayworldWB() *GrayworldWB {
	return &GrayworldWB{SaturationThreshold: 0.98}
}

// BalanceWhite scales each channel of src so their means match. src must be a
// three-channel RGB image.
func (g *GrayworldWB) BalanceWhite(src *cv.Mat) *cv.Mat {
	requireNonEmpty(src, "GrayworldWB.BalanceWhite")
	requireChannels(src, 3, "GrayworldWB.BalanceWhite")

	thresh := g.SaturationThreshold
	if thresh <= 0 || thresh > 1 {
		thresh = 1
	}
	var sum [3]float64
	var count int
	total := src.Total()
	for p := 0; p < total; p++ {
		r := float64(src.Data[p*3+0])
		gg := float64(src.Data[p*3+1])
		b := float64(src.Data[p*3+2])
		mx := math.Max(r, math.Max(gg, b))
		mn := math.Min(r, math.Min(gg, b))
		if mx > 0 {
			if (mx-mn)/mx > thresh {
				continue
			}
		}
		sum[0] += r
		sum[1] += gg
		sum[2] += b
		count++
	}
	if count == 0 {
		return src.Clone()
	}
	mean := [3]float64{sum[0] / float64(count), sum[1] / float64(count), sum[2] / float64(count)}
	grayMean := (mean[0] + mean[1] + mean[2]) / 3
	var gains [3]float64
	for c := 0; c < 3; c++ {
		if mean[c] > 1e-6 {
			gains[c] = grayMean / mean[c]
		} else {
			gains[c] = 1
		}
	}
	return ApplyChannelGains(src, gains[0], gains[1], gains[2])
}

// LearningBasedWB approximates OpenCV's learning-based white balance. The
// OpenCV implementation extracts colour and edge features from the image and
// feeds them to a regression tree trained on a labelled illuminant dataset,
// loaded from a model file. This port has no trained model; instead it
// estimates the scene illuminant directly from the same feature family:
//
//   - a brightness- and edge-weighted mean chromaticity (the dominant colour of
//     bright and textured regions, which tends to reveal the illuminant), and
//   - a gray-edge estimate (the mean of the per-channel gradient magnitudes,
//     the classical first-order gray-edge illuminant estimator).
//
// The two per-channel estimates are combined robustly (their median) to form
// the illuminant, from which neutralising channel gains are derived. See the
// package Deferred note: this is an approximation of the estimation step, not a
// reproduction of the trained regressor.
type LearningBasedWB struct {
	// RangeMaxVal is the maximum sample value (255 for 8-bit data). Samples at
	// or above SaturationThreshold*RangeMaxVal are treated as clipped and
	// excluded from the estimate.
	RangeMaxVal float64
	// SaturationThreshold is in (0,1]; pixels whose maximum channel exceeds
	// this fraction of RangeMaxVal are ignored.
	SaturationThreshold float64
	// HistBinNum controls the quantisation of the internal chromaticity
	// accumulation. It is retained for parity with OpenCV's API and to bound
	// the feature resolution; values <= 0 default to 64.
	HistBinNum int
}

// NewLearningBasedWB returns a LearningBasedWB with defaults matching OpenCV's
// public API (range 255, saturation threshold ~0.98, 64 histogram bins).
func NewLearningBasedWB() *LearningBasedWB {
	return &LearningBasedWB{RangeMaxVal: 255, SaturationThreshold: 0.98, HistBinNum: 64}
}

// BalanceWhite estimates the illuminant of src and applies neutralising gains.
// src must be a three-channel RGB image.
func (l *LearningBasedWB) BalanceWhite(src *cv.Mat) *cv.Mat {
	requireNonEmpty(src, "LearningBasedWB.BalanceWhite")
	requireChannels(src, 3, "LearningBasedWB.BalanceWhite")

	rangeMax := l.RangeMaxVal
	if rangeMax <= 0 {
		rangeMax = 255
	}
	satT := l.SaturationThreshold
	if satT <= 0 || satT > 1 {
		satT = 0.98
	}
	satLevel := satT * rangeMax
	bins := l.HistBinNum
	if bins <= 0 {
		bins = 64
	}

	// Feature 1: brightness- and edge-weighted mean colour. Edge weighting uses
	// a per-pixel gradient magnitude of luma so textured pixels (which carry
	// more illuminant information) count more, mirroring the edge features used
	// by the learning-based method.
	rows, cols := src.Rows, src.Cols
	var colorAcc [3]float64
	var colorW float64
	// Feature 2: gray-edge accumulator (mean per-channel gradient magnitude).
	var edgeAcc [3]float64
	var edgeN float64
	quant := rangeMax / float64(bins)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			r := float64(src.At(y, x, 0))
			g := float64(src.At(y, x, 1))
			b := float64(src.At(y, x, 2))
			mx := math.Max(r, math.Max(g, b))
			// Per-channel Sobel gradient magnitude for the gray-edge feature.
			gr := channelGradient(src, y, x, 0)
			gg := channelGradient(src, y, x, 1)
			gb := channelGradient(src, y, x, 2)
			if mx < satLevel {
				edgeAcc[0] += gr
				edgeAcc[1] += gg
				edgeAcc[2] += gb
				edgeN++
			}
			sum := r + g + b
			if sum <= 0 || mx >= satLevel {
				continue
			}
			edgeMag := gr + gg + gb
			// Quantise brightness to the requested bin resolution; brighter,
			// edgier pixels are weighted higher.
			bright := math.Floor(mx/quant) * quant
			w := bright * (1 + edgeMag)
			colorAcc[0] += w * r
			colorAcc[1] += w * g
			colorAcc[2] += w * b
			colorW += w
		}
	}

	// Assemble two illuminant estimates, each normalised to unit mean so they
	// are comparable, then combine per channel by their median (robust to one
	// estimate misfiring on a strongly textured or flat image).
	est1 := normalizeIllum([3]float64{colorAcc[0], colorAcc[1], colorAcc[2]}, colorW)
	est2 := normalizeIllum(edgeAcc, edgeN)
	// A third anchor: the neutral illuminant, so that when both estimates agree
	// the median stays stable and a single wild estimate cannot dominate.
	neutral := [3]float64{1, 1, 1}

	var illum [3]float64
	for c := 0; c < 3; c++ {
		vals := []float64{est1[c], est2[c], neutral[c]}
		illum[c] = median(vals)
	}
	illMean := (illum[0] + illum[1] + illum[2]) / 3
	var gains [3]float64
	for c := 0; c < 3; c++ {
		if illum[c] > 1e-6 {
			gains[c] = illMean / illum[c]
		} else {
			gains[c] = 1
		}
	}
	return ApplyChannelGains(src, gains[0], gains[1], gains[2])
}

// normalizeIllum turns a channel accumulator into a unit-mean illuminant
// vector. A zero weight yields the neutral illuminant.
func normalizeIllum(acc [3]float64, w float64) [3]float64 {
	if w <= 0 {
		return [3]float64{1, 1, 1}
	}
	m := [3]float64{acc[0] / w, acc[1] / w, acc[2] / w}
	mean := (m[0] + m[1] + m[2]) / 3
	if mean <= 1e-9 {
		return [3]float64{1, 1, 1}
	}
	return [3]float64{m[0] / mean, m[1] / mean, m[2] / mean}
}

// channelGradient returns the 3x3 Sobel gradient magnitude of a single channel
// at (y,x) with replicated borders.
func channelGradient(m *cv.Mat, y, x, c int) float64 {
	p := func(dy, dx int) float64 { return float64(atRep(m, y+dy, x+dx, c)) }
	gx := (p(-1, 1) + 2*p(0, 1) + p(1, 1)) - (p(-1, -1) + 2*p(0, -1) + p(1, -1))
	gy := (p(1, -1) + 2*p(1, 0) + p(1, 1)) - (p(-1, -1) + 2*p(-1, 0) + p(-1, 1))
	return math.Hypot(gx, gy)
}

// ApplyChannelGains multiplies the red, green and blue channels of src by
// gainR, gainG and gainB respectively, rounding and clamping each result into
// [0,255]. It ports cv::xphoto::applyChannelGains and is the operation the
// white balancers use to apply their estimated correction. src must be a
// three-channel RGB image.
func ApplyChannelGains(src *cv.Mat, gainR, gainG, gainB float64) *cv.Mat {
	requireNonEmpty(src, "ApplyChannelGains")
	requireChannels(src, 3, "ApplyChannelGains")
	gains := [3]float64{gainR, gainG, gainB}
	dst := cv.NewMat(src.Rows, src.Cols, 3)
	total := src.Total()
	for p := 0; p < total; p++ {
		for c := 0; c < 3; c++ {
			dst.Data[p*3+c] = clampU8(float64(src.Data[p*3+c]) * gains[c])
		}
	}
	return dst
}
