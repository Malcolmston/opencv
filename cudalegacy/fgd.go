package cudalegacy

import (
	"math"

	cv "github.com/malcolmston/opencv"
	"github.com/malcolmston/opencv/bgsegm"
)

// BackgroundSubtractorFGD is a CPU-backed mirror of the detector returned by
// OpenCV's cv::cuda::createBackgroundSubtractorFGD — the "foreground object
// detection" model of Liyuan Li et al. ("Foreground Object Detection from Videos
// Containing Complex Background", ACM MM 2003).
//
// This is a genuine implementation, not a delegating shim. It maintains, per
// pixel and per colour channel, an adaptive single-Gaussian background model
// (running mean and variance). A pixel is provisionally foreground when its
// summed per-channel Mahalanobis distance to the model exceeds
// [BackgroundSubtractorFGD.Threshold]. The provisional mask is then cleaned the
// way FGD cleans it: a morphological opening removes isolated speckle and a
// connected-component filter drops blobs smaller than
// [BackgroundSubtractorFGD.MinArea]. Background pixels update the model toward
// the observation at rate [BackgroundSubtractorFGD.Alpha]; detected foreground
// pixels update far more slowly (at Alpha scaled by
// [BackgroundSubtractorFGD.ForegroundAlphaScale]) so a stopped object is not
// immediately absorbed.
//
// Mask samples use the [github.com/malcolmston/opencv/bgsegm] convention:
// [bgsegm.BackgroundValue] and [bgsegm.ForegroundValue]. FGD does not classify
// shadows. Construct one with [CreateBackgroundSubtractorFGD]; the zero value is
// not usable.
type BackgroundSubtractorFGD struct {
	// Alpha is the background-model learning rate in (0,1]: each frame a
	// background pixel's mean and variance move this fraction of the way toward
	// the new observation.
	Alpha float64
	// ForegroundAlphaScale multiplies Alpha for pixels currently classified as
	// foreground, slowing (or, at 0, freezing) their absorption into the model.
	ForegroundAlphaScale float64
	// Threshold is the summed per-channel squared Mahalanobis distance above
	// which a pixel is flagged foreground. With one channel this is the squared
	// number of standard deviations.
	Threshold float64
	// InitVar is the initial per-channel variance assigned when a pixel is first
	// observed. It also serves as a variance floor to avoid division by zero.
	InitVar float64
	// OpenKernel, when greater than 1, morphologically opens the mask at that
	// odd size before regions are filtered (see [bgsegm.CleanupMask]).
	OpenKernel int
	// MinArea drops connected foreground components smaller than this many
	// pixels. A value <= 1 disables the size filter.
	MinArea int

	rows, cols, channels int
	mu, variance         []float64 // length rows*cols*channels
	inited               bool
}

// CreateBackgroundSubtractorFGD creates an FGD detector, mirroring
// cv::cuda::createBackgroundSubtractorFGD. history sets the adaptation speed:
// Alpha becomes 1/history (history <= 0 falls back to 100, i.e. Alpha 0.01).
// The remaining parameters take FGD-typical defaults — Threshold 16 (four
// standard deviations squared), InitVar 400, MinArea 15, OpenKernel 3,
// ForegroundAlphaScale 0.1 — and may be overridden on the returned value before
// the first [BackgroundSubtractorFGD.Apply].
func CreateBackgroundSubtractorFGD(history int) *BackgroundSubtractorFGD {
	if history <= 0 {
		history = 100
	}
	return &BackgroundSubtractorFGD{
		Alpha:                1.0 / float64(history),
		ForegroundAlphaScale: 0.1,
		Threshold:            16.0,
		InitVar:              400.0,
		OpenKernel:           3,
		MinArea:              15,
	}
}

func (b *BackgroundSubtractorFGD) init(frame *cv.Mat) {
	b.rows, b.cols, b.channels = frame.Rows, frame.Cols, frame.Channels
	n := len(frame.Data)
	b.mu = make([]float64, n)
	b.variance = make([]float64, n)
	for i, s := range frame.Data {
		b.mu[i] = float64(s)
		b.variance[i] = b.InitVar
	}
	b.inited = true
}

// Apply learns from or classifies frame and returns the foreground mask wrapped
// in a fresh [GpuMat]. The first frame only initialises the model and yields an
// all-background mask. It mirrors
// cv::cuda::BackgroundSubtractorFGD::apply(image, fgmask, learningRate, stream).
//
// learningRate is accepted for signature compatibility; the model uses its
// configured [BackgroundSubtractorFGD.Alpha] unless a non-negative override is
// supplied, in which case that value is used as the per-frame background rate.
// The stream is a no-op. Apply panics on a nil or empty frame, or on a frame
// whose size or channel count differs from the first frame applied.
func (b *BackgroundSubtractorFGD) Apply(frame *GpuMat, learningRate float64, stream *Stream) *GpuMat {
	_ = stream
	m := requireMat(frame, "BackgroundSubtractorFGD.Apply")
	if !b.inited {
		b.init(m)
		return GpuMatFromMat(cv.NewMat(b.rows, b.cols, 1))
	}
	if m.Rows != b.rows || m.Cols != b.cols || m.Channels != b.channels {
		panic("cudalegacy: BackgroundSubtractorFGD.Apply frame shape does not match model")
	}

	alpha := b.Alpha
	if learningRate >= 0 {
		alpha = learningRate
	}
	floor := b.InitVar * 1e-3
	if floor <= 0 {
		floor = 1e-3
	}

	mask := cv.NewMat(b.rows, b.cols, 1)
	total := b.rows * b.cols
	for p := 0; p < total; p++ {
		base := p * b.channels
		dist := 0.0
		for c := 0; c < b.channels; c++ {
			idx := base + c
			d := float64(m.Data[idx]) - b.mu[idx]
			v := b.variance[idx]
			if v < floor {
				v = floor
			}
			dist += (d * d) / v
		}
		if dist > b.Threshold {
			mask.Data[p] = bgsegm.ForegroundValue
		}
	}

	// FGD-style post-processing: open, then discard small blobs.
	mask = bgsegm.CleanupMask(mask, b.OpenKernel)
	if b.MinArea > 1 {
		mask = b.filterSmall(mask)
	}

	// Update the model. Foreground pixels adapt far more slowly.
	for p := 0; p < total; p++ {
		rate := alpha
		if mask.Data[p] == bgsegm.ForegroundValue {
			rate = alpha * b.ForegroundAlphaScale
		}
		if rate <= 0 {
			continue
		}
		base := p * b.channels
		for c := 0; c < b.channels; c++ {
			idx := base + c
			x := float64(m.Data[idx])
			d := x - b.mu[idx]
			b.mu[idx] += rate * d
			b.variance[idx] += rate * (d*d - b.variance[idx])
			if b.variance[idx] < floor {
				b.variance[idx] = floor
			}
		}
	}
	return GpuMatFromMat(mask)
}

// filterSmall zeroes every connected foreground component whose area is below
// MinArea, mirroring FGD's minimum-region gate.
func (b *BackgroundSubtractorFGD) filterSmall(mask *cv.Mat) *cv.Mat {
	labels, _, stats := cv.ConnectedComponentsWithStats(mask, cv.Connectivity8)
	out := cv.NewMat(mask.Rows, mask.Cols, 1)
	for i, l := range labels {
		if l == 0 {
			continue
		}
		if stats[l].Area >= b.MinArea {
			out.Data[i] = bgsegm.ForegroundValue
		}
	}
	return out
}

// GetBackgroundImage returns the model's current per-pixel mean as a fresh
// [GpuMat] the same shape as the learned frames, mirroring
// getBackgroundImage. It returns an empty GpuMat before the first
// [BackgroundSubtractorFGD.Apply]. The stream is a no-op.
func (b *BackgroundSubtractorFGD) GetBackgroundImage(stream *Stream) *GpuMat {
	_ = stream
	if !b.inited {
		return NewGpuMat()
	}
	out := cv.NewMat(b.rows, b.cols, b.channels)
	for i, v := range b.mu {
		out.Data[i] = clampByte(math.Round(v))
	}
	return GpuMatFromMat(out)
}

// clampByte rounds and clamps v into the uint8 range.
func clampByte(v float64) uint8 {
	if v <= 0 {
		return 0
	}
	if v >= 255 {
		return 255
	}
	return uint8(v)
}
