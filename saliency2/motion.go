package saliency2

import cv "github.com/malcolmston/opencv"

// MotionSaliencyByDifference is a streaming motion-saliency detector based on
// temporal frame differencing. Each incoming frame is compared against the
// previous one; the absolute luminance difference — optionally Gaussian
// smoothed — highlights regions that changed, which for a static camera are the
// moving objects. It implements the [MotionSaliency] interface.
//
// Construct one with [NewMotionSaliencyByDifference]. The first frame of a
// sequence has no predecessor, so its saliency map is all zeros.
type MotionSaliencyByDifference struct {
	// Sigma is the Gaussian smoothing applied to the difference image; 0
	// disables smoothing.
	Sigma float64

	prev *SaliencyMap
}

// NewMotionSaliencyByDifference returns a frame-difference detector with a mild
// sigma-1 smoothing.
func NewMotionSaliencyByDifference() *MotionSaliencyByDifference {
	return &MotionSaliencyByDifference{Sigma: 1.0}
}

// ComputeSaliencyMap ingests the next frame and returns its motion saliency as
// a [SaliencyMap]. The first frame yields a zero map. It panics if frame is nil
// or empty, or if its size differs from earlier frames.
func (m *MotionSaliencyByDifference) ComputeSaliencyMap(frame *cv.Mat) *SaliencyMap {
	cur := saliency2GrayFloat(frame)
	if m.prev == nil {
		m.prev = cur
		return NewSaliencyMap(cur.Rows, cur.Cols)
	}
	if m.prev.Rows != cur.Rows || m.prev.Cols != cur.Cols {
		panic("saliency2: MotionSaliencyByDifference frame size changed; call Reset first")
	}
	diff := NewSaliencyMap(cur.Rows, cur.Cols)
	for i := range diff.Data {
		d := cur.Data[i] - m.prev.Data[i]
		if d < 0 {
			d = -d
		}
		diff.Data[i] = d
	}
	m.prev = cur
	if m.Sigma > 0 {
		return GaussianSmooth(diff, m.Sigma)
	}
	return diff
}

// ComputeSaliency ingests the next frame and returns its motion saliency map as
// an 8-bit single-channel [cv.Mat]. It satisfies the [MotionSaliency]
// interface.
func (m *MotionSaliencyByDifference) ComputeSaliency(frame *cv.Mat) *cv.Mat {
	return m.ComputeSaliencyMap(frame).ToMat()
}

// Reset discards the stored previous frame so the next frame starts a new
// sequence.
func (m *MotionSaliencyByDifference) Reset() {
	m.prev = nil
}

// MotionSaliencyRunningAverage is a streaming motion-saliency detector that
// compares each frame against an exponentially decaying background model rather
// than only the immediately preceding frame. The background is updated as
// bg = (1-Alpha)*bg + Alpha*frame, and saliency is the absolute difference of
// the current frame from that background. Compared with plain differencing it
// suppresses the "ghost" trail a moving object leaves behind and tolerates
// slow illumination change. It implements the [MotionSaliency] interface.
//
// Construct one with [NewMotionSaliencyRunningAverage].
type MotionSaliencyRunningAverage struct {
	// Alpha is the background learning rate in [0,1]; smaller values give a
	// longer memory and a more stable background.
	Alpha float64
	// Sigma is the Gaussian smoothing applied to the saliency output; 0
	// disables smoothing.
	Sigma float64

	bg *SaliencyMap
}

// NewMotionSaliencyRunningAverage returns a running-average detector with a
// 0.05 learning rate and mild sigma-1 smoothing.
func NewMotionSaliencyRunningAverage() *MotionSaliencyRunningAverage {
	return &MotionSaliencyRunningAverage{Alpha: 0.05, Sigma: 1.0}
}

// ComputeSaliencyMap ingests the next frame, updates the background model and
// returns the frame's motion saliency as a [SaliencyMap]. The first frame
// initialises the background and yields a zero map. It panics if frame is nil
// or empty, or if its size differs from earlier frames.
func (m *MotionSaliencyRunningAverage) ComputeSaliencyMap(frame *cv.Mat) *SaliencyMap {
	cur := saliency2GrayFloat(frame)
	if m.bg == nil {
		m.bg = cur.Clone()
		return NewSaliencyMap(cur.Rows, cur.Cols)
	}
	if m.bg.Rows != cur.Rows || m.bg.Cols != cur.Cols {
		panic("saliency2: MotionSaliencyRunningAverage frame size changed; call Reset first")
	}
	alpha := saliency2ClampFloat(m.Alpha, 0, 1)
	diff := NewSaliencyMap(cur.Rows, cur.Cols)
	for i := range diff.Data {
		d := cur.Data[i] - m.bg.Data[i]
		if d < 0 {
			d = -d
		}
		diff.Data[i] = d
		m.bg.Data[i] = (1-alpha)*m.bg.Data[i] + alpha*cur.Data[i]
	}
	if m.Sigma > 0 {
		return GaussianSmooth(diff, m.Sigma)
	}
	return diff
}

// ComputeSaliency ingests the next frame and returns its motion saliency map as
// an 8-bit single-channel [cv.Mat]. It satisfies the [MotionSaliency]
// interface.
func (m *MotionSaliencyRunningAverage) ComputeSaliency(frame *cv.Mat) *cv.Mat {
	return m.ComputeSaliencyMap(frame).ToMat()
}

// Background returns a snapshot of the current background model as an 8-bit
// single-channel [cv.Mat], or nil before any frame has been ingested.
func (m *MotionSaliencyRunningAverage) Background() *cv.Mat {
	if m.bg == nil {
		return nil
	}
	out := cv.NewMat(m.bg.Rows, m.bg.Cols, 1)
	for i, v := range m.bg.Data {
		out.Data[i] = uint8(saliency2ClampFloat(v+0.5, 0, 255))
	}
	return out
}

// Reset discards the background model so the next frame starts a new sequence.
func (m *MotionSaliencyRunningAverage) Reset() {
	m.bg = nil
}
