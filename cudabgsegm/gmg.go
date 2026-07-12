package cudabgsegm

import (
	"github.com/malcolmston/opencv/bgsegm"
)

// BackgroundSubtractorGMG is a CPU-backed mirror of the subtractor returned by
// OpenCV's cv::cuda::createBackgroundSubtractorGMG — the
// Godbehere–Matsukawa–Goldberg Bayesian per-pixel model. It wraps a
// [bgsegm.BackgroundSubtractorGMG] and exchanges frames and masks as [GpuMat]
// values. GMG does not classify shadows.
//
// Construct one with [CreateBackgroundSubtractorGMG]; the zero value is not
// usable.
type BackgroundSubtractorGMG struct {
	impl *bgsegm.BackgroundSubtractorGMG
}

// CreateBackgroundSubtractorGMG creates a CUDA-style GMG subtractor, mirroring
// cv::cuda::createBackgroundSubtractorGMG(initializationFrames,
// decisionThreshold). Both arguments fall back to the OpenCV defaults (20 and
// 0.8) when non-positive.
func CreateBackgroundSubtractorGMG(initializationFrames int, decisionThreshold float64) *BackgroundSubtractorGMG {
	impl := bgsegm.NewBackgroundSubtractorGMG(initializationFrames, decisionThreshold)
	return &BackgroundSubtractorGMG{impl: impl}
}

// Apply learns from or classifies frame and returns the foreground mask wrapped
// in a fresh [GpuMat]. During the initialization period the mask is all
// background. It mirrors
// cv::cuda::BackgroundSubtractorGMG::apply(image, fgmask, learningRate, stream).
//
// learningRate is accepted for signature compatibility; the CPU model uses its
// own configured aging rate and ignores any value other than the OpenCV auto
// sentinel (-1). The stream is a no-op. Apply panics on a nil or empty frame, or
// on a frame whose size differs from the first frame applied.
func (b *BackgroundSubtractorGMG) Apply(frame *GpuMat, learningRate float64, stream *Stream) *GpuMat {
	_ = learningRate
	_ = stream
	mask := b.impl.Apply(requireFrame(frame))
	return GpuMatFromMat(mask)
}

// GetBackgroundImage returns the model's current background estimate wrapped in a
// [GpuMat]. The returned GpuMat is empty before the first
// [BackgroundSubtractorGMG.Apply]. The stream is a no-op.
func (b *BackgroundSubtractorGMG) GetBackgroundImage(stream *Stream) *GpuMat {
	_ = stream
	return GpuMatFromMat(b.impl.GetBackgroundImage())
}

// GetNumFrames returns the length of the initial learning period. Mirrors
// cv::cuda::BackgroundSubtractorGMG::getNumFrames.
func (b *BackgroundSubtractorGMG) GetNumFrames() int { return b.impl.NumInitFrames }

// SetNumFrames sets the length of the initial learning period. It only takes
// effect before the first Apply. Mirrors
// cv::cuda::BackgroundSubtractorGMG::setNumFrames.
func (b *BackgroundSubtractorGMG) SetNumFrames(n int) { b.impl.NumInitFrames = n }

// GetDecisionThreshold returns the foreground-probability threshold above which a
// pixel is classified as foreground. Mirrors
// cv::cuda::BackgroundSubtractorGMG::getDecisionThreshold.
func (b *BackgroundSubtractorGMG) GetDecisionThreshold() float64 { return b.impl.DecisionThreshold }

// SetDecisionThreshold sets the foreground-probability threshold above which a
// pixel is classified as foreground. Mirrors
// cv::cuda::BackgroundSubtractorGMG::setDecisionThreshold.
func (b *BackgroundSubtractorGMG) SetDecisionThreshold(t float64) { b.impl.DecisionThreshold = t }
