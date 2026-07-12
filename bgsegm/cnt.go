package bgsegm

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// BackgroundSubtractorCNT is the counter-based ("CouNT") background model from
// OpenCV's bgsegm module — the fastest of the family. Instead of a statistical
// distribution it simply counts, per pixel, how many consecutive frames the
// intensity has stayed constant (within Tolerance). Once that run reaches
// MinPixelStability the value is trusted as background and committed; a run
// capped at MaxPixelStability keeps a long-lived background from being forgotten
// too quickly. When UseHistory is set the last trusted value is remembered, so a
// pixel that briefly changes and returns is instantly re-recognised as
// background rather than relearned from scratch.
//
// The model is fully deterministic and stateful per pixel. Construct one with
// [NewBackgroundSubtractorCNT]; the zero value is not usable.
type BackgroundSubtractorCNT struct {
	// ShadowParams supplies the embedded DetectShadows / ShadowValue /
	// ShadowThreshold configuration and their setters.
	ShadowParams

	// MinPixelStability is the number of consecutive stable frames after which a
	// pixel value is accepted as background.
	MinPixelStability int
	// MaxPixelStability caps the stability counter so an established background
	// can still be superseded within a bounded number of frames.
	MaxPixelStability int
	// UseHistory enables remembering the last trusted background value for
	// instant re-recognition.
	UseHistory bool
	// Tolerance is the absolute intensity difference within which two successive
	// observations count as "the same value".
	Tolerance float64
	// OpenKernel, when greater than 1, morphologically opens the mask at that odd
	// size before Apply returns it (see [CleanupMask]).
	OpenKernel int

	rows, cols int
	px         []cntPixel
	inited     bool
}

// cntPixel is the per-pixel counter state.
type cntPixel struct {
	prev      float64 // last observed value
	stability int     // consecutive stable frames of prev
	bg        float64 // last committed background value
	history   int     // stability at the time bg was committed
	committed bool    // whether bg holds a trusted value yet
}

// NewBackgroundSubtractorCNT creates a CNT subtractor. minPixelStability and
// maxPixelStability fall back to the OpenCV defaults (15 and 15·60) when
// non-positive; useHistory toggles background memory. Tolerance defaults to 1 on
// the returned value and may be overridden before the first Apply.
func NewBackgroundSubtractorCNT(minPixelStability int, useHistory bool, maxPixelStability int) *BackgroundSubtractorCNT {
	if minPixelStability <= 0 {
		minPixelStability = 15
	}
	if maxPixelStability <= 0 {
		maxPixelStability = 15 * 60
	}
	if maxPixelStability < minPixelStability {
		maxPixelStability = minPixelStability
	}
	return &BackgroundSubtractorCNT{
		ShadowParams:      defaultShadowParams(),
		MinPixelStability: minPixelStability,
		MaxPixelStability: maxPixelStability,
		UseHistory:        useHistory,
		Tolerance:         1,
	}
}

func (b *BackgroundSubtractorCNT) init(frame *cv.Mat, intensity []float64) {
	b.rows, b.cols = frame.Rows, frame.Cols
	b.px = make([]cntPixel, frame.Total())
	for p := range b.px {
		b.px[p].prev = intensity[p]
	}
	b.inited = true
}

// Apply classifies frame, advances the per-pixel counters and returns the
// foreground mask. See [BackgroundSubtractor].
func (b *BackgroundSubtractorCNT) Apply(frame *cv.Mat) *cv.Mat {
	intensity := toIntensity(frame)
	if !b.inited {
		b.init(frame, intensity)
	} else {
		checkFrame(b.rows, b.cols, frame)
	}

	mask := newMask(b.rows, b.cols)
	for p := range b.px {
		mask.Data[p] = b.updatePixel(&b.px[p], intensity[p])
	}
	return applyCleanup(mask, b.OpenKernel)
}

// updatePixel advances one pixel's counter and returns its mask sample.
func (b *BackgroundSubtractorCNT) updatePixel(s *cntPixel, v float64) uint8 {
	if math.Abs(v-s.prev) <= b.Tolerance {
		if s.stability < b.MaxPixelStability {
			s.stability++
		}
	} else {
		// The value changed: if the interrupted run was long enough, remember it.
		if b.UseHistory && s.stability >= b.MinPixelStability {
			s.bg = s.prev
			s.history = s.stability
			s.committed = true
		}
		s.stability = 0
	}
	s.prev = v

	if s.stability >= b.MinPixelStability {
		// The current value is stable enough to be (or remain) background.
		s.bg = v
		s.history = s.stability
		s.committed = true
		return BackgroundValue
	}
	if b.UseHistory && s.committed && s.history >= b.MinPixelStability &&
		math.Abs(v-s.bg) <= b.Tolerance {
		// Matches the remembered background even though the current run is short.
		return BackgroundValue
	}
	if s.committed && b.isShadowOf(v, s.bg) {
		return b.shadowSample()
	}
	return ForegroundValue
}

// GetBackgroundImage returns the per-pixel trusted background value (or, before
// any value has been trusted, the most recent observation) as a single-channel
// image, or nil before the first Apply.
func (b *BackgroundSubtractorCNT) GetBackgroundImage() *cv.Mat {
	if !b.inited {
		return nil
	}
	out := cv.NewMat(b.rows, b.cols, 1)
	for p := range b.px {
		if b.px[p].committed {
			out.Data[p] = clampUint8(b.px[p].bg)
		} else {
			out.Data[p] = clampUint8(b.px[p].prev)
		}
	}
	return out
}

var _ BackgroundSubtractor = (*BackgroundSubtractorCNT)(nil)
