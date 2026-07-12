package bgsegm

import (
	"sort"

	cv "github.com/malcolmston/opencv"
)

// BackgroundSubtractorKNN is a non-parametric K-nearest-neighbours background
// model after Zivkovic and van der Heijden ("Efficient adaptive density
// estimation per image pixel for the task of background subtraction", 2006).
// Every pixel keeps a bank of NSamples recent intensity samples. An observation
// is background when at least KNNSamples of the stored samples lie within
// Dist2Threshold (a squared intensity distance) of it — that is, when the local
// sample density around the observation is high enough.
//
// The sample bank is refreshed round-robin: one slot per pixel is overwritten
// with the current observation every max(1, History/NSamples) frames, so the
// whole bank turns over about once every History frames and static changes are
// absorbed over that horizon. This update is deterministic (no random sample
// replacement).
//
// Construct one with [NewBackgroundSubtractorKNN]; the zero value is not usable.
type BackgroundSubtractorKNN struct {
	// History sets how many frames the sample bank takes to fully refresh.
	History int
	// Dist2Threshold is the squared intensity distance within which a stored
	// sample counts as a neighbour of the observation.
	Dist2Threshold float64
	// KNNSamples is the minimum number of neighbours required for an
	// observation to be classified as background.
	KNNSamples int
	// NSamples is the number of samples stored per pixel.
	NSamples int
	// DetectShadows enables classifying darkened background pixels as
	// [ShadowValue].
	DetectShadows bool
	// ShadowThreshold is the darkest relative intensity (value/sample) still
	// considered a shadow, used only when DetectShadows is set.
	ShadowThreshold float64
	// OpenKernel, when greater than 1, morphologically opens the mask at that
	// odd size before Apply returns it (see [CleanupMask]).
	OpenKernel int

	rows, cols int
	samples    [][]float64
	frameCount int
	slot       int
	inited     bool
}

// NewBackgroundSubtractorKNN creates a KNN subtractor. history and
// dist2Threshold fall back to the OpenCV defaults (500 and 400) when
// non-positive; detectShadows toggles shadow classification. KNNSamples and
// NSamples default to 2 and 7 on the returned value and may be overridden
// before the first Apply.
func NewBackgroundSubtractorKNN(history int, dist2Threshold float64, detectShadows bool) *BackgroundSubtractorKNN {
	if history <= 0 {
		history = 500
	}
	if dist2Threshold <= 0 {
		dist2Threshold = 400
	}
	return &BackgroundSubtractorKNN{
		History:         history,
		Dist2Threshold:  dist2Threshold,
		KNNSamples:      2,
		NSamples:        7,
		DetectShadows:   detectShadows,
		ShadowThreshold: 0.5,
	}
}

func (b *BackgroundSubtractorKNN) init(frame *cv.Mat, intensity []float64) {
	b.rows, b.cols = frame.Rows, frame.Cols
	b.samples = make([][]float64, frame.Total())
	for p := range b.samples {
		row := make([]float64, b.NSamples)
		for i := range row {
			row[i] = intensity[p]
		}
		b.samples[p] = row
	}
	b.inited = true
}

// Apply classifies frame, refreshes the sample banks and returns the foreground
// mask. See [BackgroundSubtractor].
func (b *BackgroundSubtractorKNN) Apply(frame *cv.Mat) *cv.Mat {
	intensity := toIntensity(frame)
	if !b.inited {
		b.init(frame, intensity)
	} else {
		checkFrame(b.rows, b.cols, frame)
	}
	b.frameCount++

	mask := newMask(b.rows, b.cols)
	for p := range b.samples {
		mask.Data[p] = b.classify(b.samples[p], intensity[p])
	}

	// Refresh one round-robin slot across all pixels at the History-derived
	// cadence, deterministically.
	period := b.History / b.NSamples
	if period < 1 {
		period = 1
	}
	if b.frameCount%period == 0 {
		for p := range b.samples {
			b.samples[p][b.slot] = intensity[p]
		}
		b.slot = (b.slot + 1) % b.NSamples
	}
	return applyCleanup(mask, b.OpenKernel)
}

// classify returns the mask value for a pixel given its sample bank and the
// current observation.
func (b *BackgroundSubtractorKNN) classify(samples []float64, v float64) uint8 {
	near := 0
	shadow := 0
	for _, s := range samples {
		d := v - s
		if d*d < b.Dist2Threshold {
			near++
		}
		if b.DetectShadows && s > 0 && v <= s && v >= b.ShadowThreshold*s {
			shadow++
		}
	}
	if near >= b.KNNSamples {
		return BackgroundValue
	}
	if b.DetectShadows && shadow >= b.KNNSamples {
		return ShadowValue
	}
	return ForegroundValue
}

// GetBackgroundImage returns the per-pixel median of the sample bank as a
// single-channel image, or nil before the first Apply.
func (b *BackgroundSubtractorKNN) GetBackgroundImage() *cv.Mat {
	if !b.inited {
		return nil
	}
	out := cv.NewMat(b.rows, b.cols, 1)
	buf := make([]float64, b.NSamples)
	for p, s := range b.samples {
		copy(buf, s)
		sort.Float64s(buf)
		out.Data[p] = clampUint8(buf[len(buf)/2])
	}
	return out
}
