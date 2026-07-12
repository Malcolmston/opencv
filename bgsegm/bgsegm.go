package bgsegm

import (
	"fmt"

	cv "github.com/malcolmston/opencv"
)

// Mask sample values produced by every [BackgroundSubtractor]. They are chosen
// so that a mask can be displayed directly as a grayscale image.
const (
	// BackgroundValue marks a pixel that fits the background model.
	BackgroundValue uint8 = 0
	// ForegroundValue marks a pixel that deviates from the background model.
	ForegroundValue uint8 = 255
	// ShadowValue marks a pixel that looks like the moving shadow of a
	// background object. It is only emitted when a model's DetectShadows option
	// is enabled.
	ShadowValue uint8 = 127
)

// BackgroundSubtractor is the common interface implemented by every model in
// this package. A subtractor is stateful: successive calls to Apply feed the
// running background model and return the foreground mask for the frame just
// supplied.
type BackgroundSubtractor interface {
	// Apply classifies frame against the current background model, updates the
	// model with the observation, and returns a fresh single-channel foreground
	// mask the same size as frame. Mask samples are [BackgroundValue],
	// [ForegroundValue] or [ShadowValue]. Every frame after the first must have
	// the same dimensions as the first; Apply panics on a size or shape it
	// cannot handle.
	Apply(frame *cv.Mat) (fgMask *cv.Mat)

	// GetBackgroundImage returns the model's current estimate of the static
	// background as a fresh single-channel [cv.Mat]. It returns nil before the
	// first call to Apply, when no model has been formed yet.
	GetBackgroundImage() *cv.Mat
}

// Compile-time checks that every model satisfies the interface.
var (
	_ BackgroundSubtractor = (*BackgroundSubtractorMOG2)(nil)
	_ BackgroundSubtractor = (*BackgroundSubtractorKNN)(nil)
	_ BackgroundSubtractor = (*RunningAverage)(nil)
	_ BackgroundSubtractor = (*BackgroundSubtractorGMG)(nil)
)

// toIntensity reduces a frame to a per-pixel scalar intensity in [0,255]. A
// single-channel frame is used as-is; a three-channel frame is converted to
// BT.601 luma; any other channel count is averaged. It panics on a nil or empty
// frame. The returned slice has length frame.Total() in row-major order.
func toIntensity(frame *cv.Mat) []float64 {
	if frame == nil || frame.Empty() {
		panic("bgsegm: Apply given a nil or empty frame")
	}
	n := frame.Total()
	out := make([]float64, n)
	switch frame.Channels {
	case 1:
		for p := 0; p < n; p++ {
			out[p] = float64(frame.Data[p])
		}
	case 3:
		for p := 0; p < n; p++ {
			base := p * 3
			r := float64(frame.Data[base+0])
			g := float64(frame.Data[base+1])
			b := float64(frame.Data[base+2])
			out[p] = 0.299*r + 0.587*g + 0.114*b
		}
	default:
		ch := frame.Channels
		for p := 0; p < n; p++ {
			base := p * ch
			sum := 0.0
			for c := 0; c < ch; c++ {
				sum += float64(frame.Data[base+c])
			}
			out[p] = sum / float64(ch)
		}
	}
	return out
}

// checkFrame verifies that a subtractor already sized to rows×cols is being fed
// a frame of matching dimensions. It panics on a mismatch.
func checkFrame(rows, cols int, frame *cv.Mat) {
	if frame.Rows != rows || frame.Cols != cols {
		panic(fmt.Sprintf("bgsegm: frame size %dx%d does not match model size %dx%d",
			frame.Rows, frame.Cols, rows, cols))
	}
}

// newMask returns a zero-filled (all-background) single-channel mask.
func newMask(rows, cols int) *cv.Mat {
	return cv.NewMat(rows, cols, 1)
}

// clampUint8 rounds v and clamps it into the [0,255] range.
func clampUint8(v float64) uint8 {
	if v <= 0 {
		return 0
	}
	if v >= 255 {
		return 255
	}
	return uint8(v + 0.5)
}

// CleanupMask returns a morphologically opened copy of mask: an erosion
// followed by a dilation over a ksize×ksize rectangular structuring element,
// performed with [cv.MorphologyEx] and [cv.MorphOpen]. Opening removes bright
// specks smaller than the kernel while preserving the shape of larger blobs. If
// ksize is not a positive odd number it is rounded up to the next odd value;
// ksize <= 1 returns mask unchanged. The input mask is not modified.
func CleanupMask(mask *cv.Mat, ksize int) *cv.Mat {
	if ksize <= 1 {
		return mask
	}
	if ksize%2 == 0 {
		ksize++
	}
	kernel := cv.GetStructuringElement(cv.MorphRect, ksize, ksize)
	return cv.MorphologyEx(mask, kernel, cv.MorphOpen, 1)
}

// applyCleanup runs CleanupMask when openKernel requests it.
func applyCleanup(mask *cv.Mat, openKernel int) *cv.Mat {
	if openKernel > 1 {
		return CleanupMask(mask, openKernel)
	}
	return mask
}
