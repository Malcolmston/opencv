package cudabgsegm

import (
	"github.com/malcolmston/opencv/bgsegm"
)

// BackgroundSubtractorMOG2 is a CPU-backed mirror of the subtractor returned by
// OpenCV's cv::cuda::createBackgroundSubtractorMOG2 — the Zivkovic adaptive
// Gaussian-mixture model with optional shadow detection. It wraps a
// [bgsegm.BackgroundSubtractorMOG2] and exchanges frames and masks as [GpuMat]
// values.
//
// Construct one with [CreateBackgroundSubtractorMOG2]; the zero value is not
// usable.
type BackgroundSubtractorMOG2 struct {
	impl *bgsegm.BackgroundSubtractorMOG2
}

// CreateBackgroundSubtractorMOG2 creates a CUDA-style MOG2 subtractor, mirroring
// cv::cuda::createBackgroundSubtractorMOG2(history, varThreshold, detectShadows).
// history and varThreshold fall back to the OpenCV defaults (500 and 16) when
// non-positive; detectShadows toggles shadow classification.
func CreateBackgroundSubtractorMOG2(history int, varThreshold float64, detectShadows bool) *BackgroundSubtractorMOG2 {
	impl := bgsegm.NewBackgroundSubtractorMOG2(history, varThreshold, detectShadows)
	return &BackgroundSubtractorMOG2{impl: impl}
}

// Apply classifies frame against the model, updates the model and returns the
// foreground mask wrapped in a fresh [GpuMat]. It mirrors
// cv::cuda::BackgroundSubtractorMOG2::apply(image, fgmask, learningRate, stream).
//
// learningRate is accepted for signature compatibility. The CPU model derives its
// own adaptive rate from the configured history and the number of frames seen so
// far, so any value other than the OpenCV auto sentinel (-1) is ignored. The
// stream is a no-op. Apply panics on a nil or empty frame, or on a frame whose
// size differs from the first frame applied.
func (b *BackgroundSubtractorMOG2) Apply(frame *GpuMat, learningRate float64, stream *Stream) *GpuMat {
	_ = learningRate
	_ = stream
	mask := b.impl.Apply(requireFrame(frame))
	return GpuMatFromMat(mask)
}

// GetBackgroundImage returns the model's current background estimate wrapped in a
// [GpuMat], mirroring cv::cuda::BackgroundSubtractorMOG2::getBackgroundImage. The
// returned GpuMat is empty before the first [BackgroundSubtractorMOG2.Apply]. The
// stream is a no-op.
func (b *BackgroundSubtractorMOG2) GetBackgroundImage(stream *Stream) *GpuMat {
	_ = stream
	return GpuMatFromMat(b.impl.GetBackgroundImage())
}

// GetHistory returns the model's history length. Mirrors
// cv::cuda::BackgroundSubtractorMOG2::getHistory.
func (b *BackgroundSubtractorMOG2) GetHistory() int { return b.impl.History }

// SetHistory sets the model's history length. Mirrors
// cv::cuda::BackgroundSubtractorMOG2::setHistory.
func (b *BackgroundSubtractorMOG2) SetHistory(history int) { b.impl.History = history }

// GetVarThreshold returns the squared-Mahalanobis matching threshold. Mirrors
// cv::cuda::BackgroundSubtractorMOG2::getVarThreshold.
func (b *BackgroundSubtractorMOG2) GetVarThreshold() float64 { return b.impl.VarThreshold }

// SetVarThreshold sets the squared-Mahalanobis matching threshold. Mirrors
// cv::cuda::BackgroundSubtractorMOG2::setVarThreshold.
func (b *BackgroundSubtractorMOG2) SetVarThreshold(t float64) { b.impl.VarThreshold = t }

// GetDetectShadows reports whether shadow classification is enabled. Mirrors
// cv::cuda::BackgroundSubtractorMOG2::getDetectShadows.
func (b *BackgroundSubtractorMOG2) GetDetectShadows() bool { return b.impl.DetectShadows }

// SetDetectShadows enables or disables shadow classification. Mirrors
// cv::cuda::BackgroundSubtractorMOG2::setDetectShadows.
func (b *BackgroundSubtractorMOG2) SetDetectShadows(on bool) { b.impl.DetectShadows = on }
