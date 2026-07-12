package fuzzy

import (
	"fmt"
	"math"

	cv "github.com/malcolmston/opencv"
)

// ErrorStats summarises the per-sample discrepancy between a reference image and
// an F-transform reconstruction of it. It lets callers report the quality of a
// smoothing, inverse, or inpainting result numerically instead of eyeballing it.
type ErrorStats struct {
	// MAE is the mean absolute error across every sample and channel.
	MAE float64
	// RMSE is the root-mean-square error across every sample and channel.
	RMSE float64
	// MaxAbs is the largest absolute single-sample error.
	MaxAbs float64
	// PSNR is the peak signal-to-noise ratio in decibels, computed against the
	// 8-bit peak of 255. It is +Inf when the images are identical.
	PSNR float64
	// Samples is the number of samples (Rows*Cols*Channels) compared.
	Samples int
}

// String renders the statistics compactly, e.g. "MAE=0.42 RMSE=0.71 Max=3 PSNR=51.1dB".
func (e ErrorStats) String() string {
	psnr := fmt.Sprintf("%.1f", e.PSNR)
	if math.IsInf(e.PSNR, 1) {
		psnr = "inf"
	}
	return fmt.Sprintf("MAE=%.3f RMSE=%.3f Max=%.0f PSNR=%sdB", e.MAE, e.RMSE, e.MaxAbs, psnr)
}

// TransformError compares reference against reconstructed sample by sample and
// returns the aggregate [ErrorStats]. The two images must have identical
// dimensions and channel counts. It is the general-purpose quality/error report
// for any routine in this package: pass the original image and, for example, the
// output of [Filter], [FT12DProcess] or [InpaintMultiStep].
func TransformError(reference, reconstructed *cv.Mat) ErrorStats {
	if reference == nil || reconstructed == nil {
		panic("fuzzy: TransformError given a nil image")
	}
	if reference.Rows != reconstructed.Rows || reference.Cols != reconstructed.Cols ||
		reference.Channels != reconstructed.Channels {
		panic(fmt.Sprintf("fuzzy: TransformError size/channel mismatch %dx%dx%d vs %dx%dx%d",
			reference.Rows, reference.Cols, reference.Channels,
			reconstructed.Rows, reconstructed.Cols, reconstructed.Channels))
	}
	n := len(reference.Data)
	if n == 0 {
		panic("fuzzy: TransformError given an empty image")
	}
	var sumAbs, sumSq, maxAbs float64
	for i := 0; i < n; i++ {
		d := math.Abs(float64(int(reference.Data[i]) - int(reconstructed.Data[i])))
		sumAbs += d
		sumSq += d * d
		if d > maxAbs {
			maxAbs = d
		}
	}
	mse := sumSq / float64(n)
	psnr := math.Inf(1)
	if mse > 0 {
		psnr = 10 * math.Log10(255*255/mse)
	}
	return ErrorStats{
		MAE:     sumAbs / float64(n),
		RMSE:    math.Sqrt(mse),
		MaxAbs:  maxAbs,
		PSNR:    psnr,
		Samples: n,
	}
}

// MaskedError is like [TransformError] but restricts the comparison to the pixels
// where mask is non-zero (in its first channel). It is the natural way to score
// an inpainting result, which should only be judged on the reconstructed hole:
// pass the ground-truth image, the inpainted output and the hole mask. reference
// and reconstructed must share dimensions and channels; mask must match their
// pixel dimensions.
func MaskedError(reference, reconstructed, mask *cv.Mat) ErrorStats {
	if reference == nil || reconstructed == nil || mask == nil {
		panic("fuzzy: MaskedError given a nil image")
	}
	if reference.Rows != reconstructed.Rows || reference.Cols != reconstructed.Cols ||
		reference.Channels != reconstructed.Channels {
		panic(fmt.Sprintf("fuzzy: MaskedError size/channel mismatch %dx%dx%d vs %dx%dx%d",
			reference.Rows, reference.Cols, reference.Channels,
			reconstructed.Rows, reconstructed.Cols, reconstructed.Channels))
	}
	if mask.Rows != reference.Rows || mask.Cols != reference.Cols {
		panic(fmt.Sprintf("fuzzy: MaskedError mask %dx%d does not match image %dx%d",
			mask.Rows, mask.Cols, reference.Rows, reference.Cols))
	}
	rows, cols, ch := reference.Rows, reference.Cols, reference.Channels
	var sumAbs, sumSq, maxAbs float64
	var n int
	for p := 0; p < rows*cols; p++ {
		if mask.Data[p*mask.Channels] == 0 {
			continue
		}
		for c := 0; c < ch; c++ {
			d := math.Abs(float64(int(reference.Data[p*ch+c]) - int(reconstructed.Data[p*ch+c])))
			sumAbs += d
			sumSq += d * d
			if d > maxAbs {
				maxAbs = d
			}
			n++
		}
	}
	if n == 0 {
		return ErrorStats{PSNR: math.Inf(1)}
	}
	mse := sumSq / float64(n)
	psnr := math.Inf(1)
	if mse > 0 {
		psnr = 10 * math.Log10(255*255/mse)
	}
	return ErrorStats{
		MAE:     sumAbs / float64(n),
		RMSE:    math.Sqrt(mse),
		MaxAbs:  maxAbs,
		PSNR:    psnr,
		Samples: n,
	}
}
