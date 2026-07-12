package bgsegm

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// RunningAverage is the simplest background model: it keeps an exponential
// moving average of the frame intensities as the background image and flags a
// pixel as foreground when the current frame differs from that average by more
// than Threshold. The average is updated toward every observation with rate
// Alpha, so a pixel that settles at a new value is absorbed into the background
// with a time constant of 1/Alpha frames.
//
// Construct one with [NewRunningAverage]; the zero value is not usable.
type RunningAverage struct {
	// Alpha is the moving-average learning rate in (0,1]: bg ← bg + Alpha·(v-bg)
	// each frame. Larger values adapt faster.
	Alpha float64
	// Threshold is the absolute intensity difference above which a pixel is
	// foreground.
	Threshold float64
	// OpenKernel, when greater than 1, morphologically opens the mask at that
	// odd size before Apply returns it (see [CleanupMask]).
	OpenKernel int

	rows, cols int
	bg         []float64
	inited     bool
}

// NewRunningAverage creates a running-average subtractor. history sets the
// learning rate to Alpha = 1/history (falling back to 1/500 when non-positive),
// and threshold is the foreground difference threshold (falling back to 25 when
// non-positive). Alpha may be overridden directly before the first Apply.
func NewRunningAverage(history int, threshold float64) *RunningAverage {
	if history <= 0 {
		history = 500
	}
	if threshold <= 0 {
		threshold = 25
	}
	return &RunningAverage{
		Alpha:     1.0 / float64(history),
		Threshold: threshold,
	}
}

// Apply thresholds frame against the running background, updates the background
// and returns the foreground mask. The first frame initialises the background
// and yields an all-background mask. See [BackgroundSubtractor].
func (r *RunningAverage) Apply(frame *cv.Mat) *cv.Mat {
	intensity := toIntensity(frame)
	if !r.inited {
		r.rows, r.cols = frame.Rows, frame.Cols
		r.bg = make([]float64, len(intensity))
		copy(r.bg, intensity)
		r.inited = true
		return newMask(r.rows, r.cols)
	}
	checkFrame(r.rows, r.cols, frame)

	mask := newMask(r.rows, r.cols)
	for p, v := range intensity {
		if math.Abs(v-r.bg[p]) > r.Threshold {
			mask.Data[p] = ForegroundValue
		}
		r.bg[p] += r.Alpha * (v - r.bg[p])
	}
	return applyCleanup(mask, r.OpenKernel)
}

// GetBackgroundImage returns the current running-average background as a
// single-channel image, or nil before the first Apply.
func (r *RunningAverage) GetBackgroundImage() *cv.Mat {
	if !r.inited {
		return nil
	}
	out := cv.NewMat(r.rows, r.cols, 1)
	for p, v := range r.bg {
		out.Data[p] = clampUint8(v)
	}
	return out
}
