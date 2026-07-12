package bgsegm

import (
	cv "github.com/malcolmston/opencv"
)

// BackgroundSubtractorGMG is a Bayesian per-pixel background model after
// Godbehere, Matsukawa and Goldberg ("Visual Tracking of Human Visitors under
// Variable-Lighting Conditions", 2012). Each pixel quantises its intensity into
// NumBins bins and maintains a decaying histogram of how often each bin has
// been observed. During the first NumInitFrames frames the model only learns
// (its output is all background). Thereafter an observation whose posterior
// background probability — the normalised weight of its bin — is low enough that
// its foreground probability exceeds DecisionThreshold is flagged as
// foreground, and the histogram is aged toward the new observation with rate
// LearningRate.
//
// Construct one with [NewBackgroundSubtractorGMG]; the zero value is not usable.
// GMG does not classify shadows.
type BackgroundSubtractorGMG struct {
	// NumInitFrames is the length of the initial learning period during which
	// Apply returns an all-background mask.
	NumInitFrames int
	// NumBins is the number of intensity quantisation bins per pixel.
	NumBins int
	// DecisionThreshold is the foreground-probability (1 - posterior background
	// probability) above which a pixel is classified as foreground.
	DecisionThreshold float64
	// LearningRate is the histogram aging rate in (0,1]: each frame the
	// histogram is scaled by (1-LearningRate) and the observed bin is boosted.
	LearningRate float64
	// OpenKernel, when greater than 1, morphologically opens the mask at that
	// odd size before Apply returns it (see [CleanupMask]).
	OpenKernel int

	rows, cols int
	hist       [][]float64
	frameCount int
	inited     bool
}

// NewBackgroundSubtractorGMG creates a GMG subtractor. numInitFrames and
// decisionThreshold fall back to the OpenCV defaults (20 and 0.8) when
// non-positive. NumBins and LearningRate default to 16 and 0.025 on the
// returned value and may be overridden before the first Apply.
func NewBackgroundSubtractorGMG(numInitFrames int, decisionThreshold float64) *BackgroundSubtractorGMG {
	if numInitFrames <= 0 {
		numInitFrames = 20
	}
	if decisionThreshold <= 0 {
		decisionThreshold = 0.8
	}
	return &BackgroundSubtractorGMG{
		NumInitFrames:     numInitFrames,
		NumBins:           16,
		DecisionThreshold: decisionThreshold,
		LearningRate:      0.025,
	}
}

func (b *BackgroundSubtractorGMG) init(frame *cv.Mat) {
	b.rows, b.cols = frame.Rows, frame.Cols
	b.hist = make([][]float64, frame.Total())
	for i := range b.hist {
		b.hist[i] = make([]float64, b.NumBins)
	}
	b.inited = true
}

// binOf maps an intensity in [0,255] to a histogram bin in [0, NumBins).
func (b *BackgroundSubtractorGMG) binOf(v float64) int {
	bin := int(v) * b.NumBins / 256
	if bin < 0 {
		return 0
	}
	if bin >= b.NumBins {
		return b.NumBins - 1
	}
	return bin
}

// Apply learns from or classifies frame and returns the foreground mask. During
// the first NumInitFrames the mask is all background. See [BackgroundSubtractor].
func (b *BackgroundSubtractorGMG) Apply(frame *cv.Mat) *cv.Mat {
	intensity := toIntensity(frame)
	if !b.inited {
		b.init(frame)
	} else {
		checkFrame(b.rows, b.cols, frame)
	}

	mask := newMask(b.rows, b.cols)
	learning := b.frameCount < b.NumInitFrames
	for p, v := range intensity {
		h := b.hist[p]
		bin := b.binOf(v)
		if learning {
			h[bin]++
			continue
		}
		total := 0.0
		for _, w := range h {
			total += w
		}
		fgProb := 1.0
		if total > 0 {
			fgProb = 1.0 - h[bin]/total
		}
		if fgProb > b.DecisionThreshold {
			mask.Data[p] = ForegroundValue
		}
		// Age the histogram toward the current observation.
		for k := range h {
			h[k] *= 1 - b.LearningRate
		}
		h[bin] += b.LearningRate
	}
	b.frameCount++
	return applyCleanup(mask, b.OpenKernel)
}

// GetBackgroundImage returns the centre intensity of each pixel's most-observed
// histogram bin as a single-channel image, or nil before the first Apply.
func (b *BackgroundSubtractorGMG) GetBackgroundImage() *cv.Mat {
	if !b.inited {
		return nil
	}
	out := cv.NewMat(b.rows, b.cols, 1)
	binWidth := 256.0 / float64(b.NumBins)
	for p, h := range b.hist {
		best := 0
		bestW := -1.0
		for k, w := range h {
			if w > bestW {
				bestW = w
				best = k
			}
		}
		center := (float64(best) + 0.5) * binWidth
		out.Data[p] = clampUint8(center)
	}
	return out
}
