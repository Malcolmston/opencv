package tracking

import cv "github.com/malcolmston/opencv"

// ConfidenceTracker is a [Tracker] that also reports a continuous per-frame
// confidence score instead of only a boolean flag. The score's scale is
// tracker-specific — a peak-to-sidelobe ratio for [TrackerMOSSE] and
// [TrackerCSRT], a peak correlation response for [TrackerDCF] and
// [TrackerKCFHOG], a classifier margin for [TrackerMIL] and [TrackerBoosting],
// and a template NCC similarity for [TrackerTLD] — but within one tracker a
// higher value always means a more reliable localisation. Use it when the
// application needs to rank frames, trigger re-detection, or fuse trackers.
type ConfidenceTracker interface {
	Tracker
	// UpdateConfidence locates the object and returns its box and confidence.
	UpdateConfidence(frame *cv.Mat) (cv.Rect, float64)
}

// MultiTracker manages a collection of independent single-object trackers,
// mirroring OpenCV's legacy cv::MultiTracker: one call updates every tracked
// object on a shared frame. Trackers may be of different concrete types.
//
// Construct it with [NewMultiTracker], register objects with
// [MultiTracker.Add], and advance them all with [MultiTracker.Update] (boxes and
// per-object flags) or [MultiTracker.UpdateConfidence] (boxes and per-object
// confidences, using [ConfidenceTracker] where a tracker supports it).
type MultiTracker struct {
	trackers []Tracker
	boxes    []cv.Rect
}

// NewMultiTracker returns an empty MultiTracker.
func NewMultiTracker() *MultiTracker { return &MultiTracker{} }

// Add registers tr for a new object, initialising it on frame with bbox. The
// object's index is len(objects)-1 after the call.
func (m *MultiTracker) Add(tr Tracker, frame *cv.Mat, bbox cv.Rect) {
	tr.Init(frame, bbox)
	m.trackers = append(m.trackers, tr)
	m.boxes = append(m.boxes, bbox)
}

// Len returns the number of tracked objects.
func (m *MultiTracker) Len() int { return len(m.trackers) }

// Boxes returns a copy of the most recent bounding box of every object, in
// registration order.
func (m *MultiTracker) Boxes() []cv.Rect {
	out := make([]cv.Rect, len(m.boxes))
	copy(out, m.boxes)
	return out
}

// Update advances every tracker on frame and returns each object's new box and
// confidence flag, in registration order.
func (m *MultiTracker) Update(frame *cv.Mat) ([]cv.Rect, []bool) {
	boxes := make([]cv.Rect, len(m.trackers))
	oks := make([]bool, len(m.trackers))
	for i, tr := range m.trackers {
		box, ok := tr.Update(frame)
		boxes[i] = box
		oks[i] = ok
		m.boxes[i] = box
	}
	return boxes, oks
}

// UpdateConfidence advances every tracker and returns each object's new box and
// a continuous confidence. Trackers implementing [ConfidenceTracker] report
// their native score; for a plain [Tracker] the confidence is 1 when its Update
// flag is true and 0 otherwise.
func (m *MultiTracker) UpdateConfidence(frame *cv.Mat) ([]cv.Rect, []float64) {
	boxes := make([]cv.Rect, len(m.trackers))
	confs := make([]float64, len(m.trackers))
	for i, tr := range m.trackers {
		if ct, ok := tr.(ConfidenceTracker); ok {
			box, c := ct.UpdateConfidence(frame)
			boxes[i] = box
			confs[i] = c
			m.boxes[i] = box
			continue
		}
		box, ok := tr.Update(frame)
		boxes[i] = box
		if ok {
			confs[i] = 1
		}
		m.boxes[i] = box
	}
	return boxes, confs
}

// Compile-time checks that the new trackers satisfy the interfaces.
var (
	_ Tracker = (*TrackerMOSSE)(nil)
	_ Tracker = (*TrackerDCF)(nil)
	_ Tracker = (*TrackerKCFHOG)(nil)
	_ Tracker = (*TrackerCSRT)(nil)
	_ Tracker = (*TrackerMIL)(nil)
	_ Tracker = (*TrackerBoosting)(nil)
	_ Tracker = (*TrackerTLD)(nil)

	_ ConfidenceTracker = (*TrackerMOSSE)(nil)
	_ ConfidenceTracker = (*TrackerDCF)(nil)
	_ ConfidenceTracker = (*TrackerKCFHOG)(nil)
	_ ConfidenceTracker = (*TrackerCSRT)(nil)
	_ ConfidenceTracker = (*TrackerMIL)(nil)
	_ ConfidenceTracker = (*TrackerBoosting)(nil)
	_ ConfidenceTracker = (*TrackerTLD)(nil)
)
