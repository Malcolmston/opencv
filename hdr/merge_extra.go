package hdr

import (
	"errors"
	"math"

	cv "github.com/malcolmston/opencv"
)

// WeightFunc maps an 8-bit pixel value to the confidence weight it receives when
// several exposures are combined into a radiance estimate. A good weight is
// large for well-exposed mid-tones and small near the black and white clipping
// points. Weights must be non-negative.
type WeightFunc func(z int) float64

// HatWeight is Debevec & Malik's triangular ("hat") weighting: it rises
// linearly from the extremes to a peak at the middle of the 8-bit range and is
// never zero, so no sample is discarded outright.
func HatWeight(z int) float64 { return hat(z) }

// TentWeight is the symmetric tent min(z, 255-z): it reaches zero at both
// clipping points, fully rejecting saturated and black pixels.
func TentWeight(z int) float64 {
	if z < 255-z {
		return float64(z)
	}
	return float64(255 - z)
}

// GaussianWeight is a Gaussian centred on mid-grey (127.5) with a standard
// deviation of a quarter of the range, giving a smooth roll-off that strongly
// down-weights the tonal extremes. This is a robust choice in the presence of
// noise and clipping.
func GaussianWeight(z int) float64 {
	const mu = 127.5
	const sigma = 63.75 // 255/4
	d := (float64(z) - mu) / sigma
	return math.Exp(-0.5 * d * d)
}

// UniformWeight gives every unsaturated pixel equal weight and rejects the two
// clipping points outright.
func UniformWeight(z int) float64 {
	if z == 0 || z == 255 {
		return 0
	}
	return 1
}

// weightTable materialises a WeightFunc into a 256-entry lookup, defaulting to
// [HatWeight] when w is nil.
func weightTable(w WeightFunc) []float64 {
	if w == nil {
		w = HatWeight
	}
	t := make([]float64, 256)
	for z := 0; z < 256; z++ {
		v := w(z)
		if v < 0 {
			v = 0
		}
		t[z] = v
	}
	return t
}

// MergeDebevecFunc is [MergeDebevec] with a caller-supplied robustness weighting
// instead of the fixed hat weight. Passing a nil WeightFunc reproduces
// [MergeDebevec] exactly. Use [GaussianWeight] or [TentWeight] to reject clipped
// samples more aggressively when the bracket contains heavy over- or
// under-exposure.
func MergeDebevecFunc(images []*cv.Mat, times []float64, resp *CameraResponse, w WeightFunc) (*Radiance, error) {
	if err := validateStack(images, times); err != nil {
		return nil, err
	}
	if resp == nil {
		return nil, errors.New("hdr: nil camera response")
	}
	rows, cols, ch := images[0].Rows, images[0].Cols, images[0].Channels
	if resp.Channels != ch {
		return nil, errors.New("hdr: response channel count does not match images")
	}
	wt := weightTable(w)
	logTimes := make([]float64, len(times))
	for j, t := range times {
		logTimes[j] = math.Log(t)
	}
	logCurve := make([][]float64, ch)
	for c := 0; c < ch; c++ {
		logCurve[c] = resp.logCurve(c)
	}
	out := NewRadiance(rows, cols, ch)
	nImg := len(images)
	total := rows * cols
	for p := 0; p < total; p++ {
		for c := 0; c < ch; c++ {
			var num, den float64
			for j := 0; j < nImg; j++ {
				z := int(images[j].Data[p*ch+c])
				wv := wt[z]
				num += wv * (logCurve[c][z] - logTimes[j])
				den += wv
			}
			if den > 0 {
				out.Data[p*ch+c] = math.Exp(num / den)
			}
		}
	}
	return out, nil
}

// MergeRobertson merges an LDR bracket into a linear radiance map with
// Robertson's estimator, the counterpart of OpenCV's createMergeRobertson. Given
// the inverse camera response resp (the linear radiance for each pixel value)
// and the exposure times, each pixel's radiance is the weighted least-squares
// solution E = Σ w·t·I(z) / Σ w·t², which is optimal under the assumption of
// additive noise proportional to exposure time. Pair it with
// [CalibrateRobertson], whose curve is exactly this inverse response.
func MergeRobertson(images []*cv.Mat, times []float64, resp *CameraResponse) (*Radiance, error) {
	if err := validateStack(images, times); err != nil {
		return nil, err
	}
	if resp == nil {
		return nil, errors.New("hdr: nil camera response")
	}
	rows, cols, ch := images[0].Rows, images[0].Cols, images[0].Channels
	if resp.Channels != ch {
		return nil, errors.New("hdr: response channel count does not match images")
	}
	out := NewRadiance(rows, cols, ch)
	nImg := len(images)
	total := rows * cols
	for p := 0; p < total; p++ {
		for c := 0; c < ch; c++ {
			var num, den float64
			for j := 0; j < nImg; j++ {
				z := int(images[j].Data[p*ch+c])
				wv := hat(z)
				num += wv * times[j] * resp.Curve[c][z]
				den += wv * times[j] * times[j]
			}
			if den > 0 {
				out.Data[p*ch+c] = num / den
			}
		}
	}
	return out, nil
}

// MergeMertensProcessor is a reusable, configurable Mertens exposure-fusion
// operator. It wraps the [MergeMertens] function with a stored
// [MergeMertensParams] so the same weighting can be applied to many brackets,
// and it mirrors OpenCV's MergeMertens object, which offers both a plain
// process and a process-with-exposure-times overload.
type MergeMertensProcessor struct {
	// Params holds the contrast, saturation and well-exposedness exponents.
	Params MergeMertensParams
}

// NewMergeMertensProcessor returns a processor with the given weighting
// exponents. Non-positive fields fall back to their defaults inside
// [MergeMertens].
func NewMergeMertensProcessor(params MergeMertensParams) *MergeMertensProcessor {
	return &MergeMertensProcessor{Params: params}
}

// Process fuses the bracket with the stored parameters. It is equivalent to
// calling [MergeMertens] directly.
func (m *MergeMertensProcessor) Process(images []*cv.Mat) (*cv.Mat, error) {
	return MergeMertens(images, m.Params)
}

// ProcessWithExposures fuses the bracket, accepting per-image exposure times for
// API parity with OpenCV. Mertens fusion is purely intensity-based, so the
// times do not affect the result; they are only validated for length. Pass a
// nil times slice to skip the check.
func (m *MergeMertensProcessor) ProcessWithExposures(images []*cv.Mat, times []float64) (*cv.Mat, error) {
	if times != nil && len(times) != len(images) {
		return nil, errors.New("hdr: number of images and exposure times differ")
	}
	return MergeMertens(images, m.Params)
}
