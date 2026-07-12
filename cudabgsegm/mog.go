package cudabgsegm

import (
	"github.com/malcolmston/opencv/bgsegm"
)

// BackgroundSubtractorMOG is a CPU-backed mirror of the subtractor returned by
// OpenCV's cv::cuda::createBackgroundSubtractorMOG — the original
// KaewTraKulPong–Bowden adaptive Gaussian-mixture model. It wraps a
// [bgsegm.BackgroundSubtractorMOG] and exchanges frames and masks as [GpuMat]
// values.
//
// Construct one with [CreateBackgroundSubtractorMOG]; the zero value is not
// usable.
type BackgroundSubtractorMOG struct {
	impl *bgsegm.BackgroundSubtractorMOG
}

// CreateBackgroundSubtractorMOG creates a CUDA-style MOG subtractor, mirroring
// cv::cuda::createBackgroundSubtractorMOG(history, nmixtures, backgroundRatio,
// noiseSigma). history and nmixtures fall back to the OpenCV defaults (200 and 5)
// when non-positive. A non-positive backgroundRatio or noiseSigma leaves the CPU
// model's own defaults (0.7 and 15) in place, matching OpenCV's use of 0 as the
// "auto noise" sentinel.
func CreateBackgroundSubtractorMOG(history, nmixtures int, backgroundRatio, noiseSigma float64) *BackgroundSubtractorMOG {
	impl := bgsegm.NewBackgroundSubtractorMOG(history, nmixtures, false)
	if backgroundRatio > 0 {
		impl.BackgroundRatio = backgroundRatio
	}
	if noiseSigma > 0 {
		impl.NoiseSigma = noiseSigma
	}
	return &BackgroundSubtractorMOG{impl: impl}
}

// Apply classifies frame against the model, updates the model and returns the
// foreground mask wrapped in a fresh [GpuMat]. It mirrors
// cv::cuda::BackgroundSubtractorMOG::apply(image, fgmask, learningRate, stream).
//
// learningRate is accepted for signature compatibility. The CPU model derives
// its own adaptive rate from the configured history and the number of frames seen
// so far, so any value other than the OpenCV auto sentinel (-1) is ignored. The
// stream is a no-op. Apply panics on a nil or empty frame, or on a frame whose
// size differs from the first frame applied.
func (b *BackgroundSubtractorMOG) Apply(frame *GpuMat, learningRate float64, stream *Stream) *GpuMat {
	_ = learningRate
	_ = stream
	mask := b.impl.Apply(requireFrame(frame))
	return GpuMatFromMat(mask)
}

// GetBackgroundImage returns the model's current background estimate wrapped in a
// [GpuMat], mirroring cv::cuda::BackgroundSubtractorMOG::getBackgroundImage. The
// returned GpuMat is empty before the first [BackgroundSubtractorMOG.Apply]. The
// stream is a no-op.
func (b *BackgroundSubtractorMOG) GetBackgroundImage(stream *Stream) *GpuMat {
	_ = stream
	return GpuMatFromMat(b.impl.GetBackgroundImage())
}

// GetHistory returns the model's history length. Mirrors
// cv::cuda::BackgroundSubtractorMOG::getHistory.
func (b *BackgroundSubtractorMOG) GetHistory() int { return b.impl.History }

// SetHistory sets the model's history length. Mirrors
// cv::cuda::BackgroundSubtractorMOG::setHistory.
func (b *BackgroundSubtractorMOG) SetHistory(history int) { b.impl.History = history }

// GetNMixtures returns the number of Gaussians kept per pixel. Mirrors
// cv::cuda::BackgroundSubtractorMOG::getNMixtures.
func (b *BackgroundSubtractorMOG) GetNMixtures() int { return b.impl.NMixtures }

// SetNMixtures sets the number of Gaussians kept per pixel. It only takes effect
// before the first Apply, since the per-pixel mixtures are allocated then.
// Mirrors cv::cuda::BackgroundSubtractorMOG::setNMixtures.
func (b *BackgroundSubtractorMOG) SetNMixtures(n int) { b.impl.NMixtures = n }

// GetBackgroundRatio returns the fraction of mixture weight that defines the
// background. Mirrors cv::cuda::BackgroundSubtractorMOG::getBackgroundRatio.
func (b *BackgroundSubtractorMOG) GetBackgroundRatio() float64 { return b.impl.BackgroundRatio }

// SetBackgroundRatio sets the fraction of mixture weight that defines the
// background. Mirrors cv::cuda::BackgroundSubtractorMOG::setBackgroundRatio.
func (b *BackgroundSubtractorMOG) SetBackgroundRatio(r float64) { b.impl.BackgroundRatio = r }

// GetNoiseSigma returns the standard deviation assigned to a freshly spawned
// Gaussian. Mirrors cv::cuda::BackgroundSubtractorMOG::getNoiseSigma.
func (b *BackgroundSubtractorMOG) GetNoiseSigma() float64 { return b.impl.NoiseSigma }

// SetNoiseSigma sets the standard deviation assigned to a freshly spawned
// Gaussian. Mirrors cv::cuda::BackgroundSubtractorMOG::setNoiseSigma.
func (b *BackgroundSubtractorMOG) SetNoiseSigma(sigma float64) { b.impl.NoiseSigma = sigma }
