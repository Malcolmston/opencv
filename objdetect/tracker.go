package objdetect

import (
	"sort"

	cv "github.com/malcolmston/opencv"
)

// Detector is anything that can return object rectangles for an image. The
// concrete detectors in this package — [CascadeClassifier], [LBPCascadeClassifier]
// and [SoftCascade] — all satisfy it, so any of them can drive a
// [DetectionBasedTracker].
type Detector interface {
	DetectMultiScale(img *cv.Mat) []cv.Rect
}

// TrackedObject is one object followed across frames by a
// [DetectionBasedTracker]. Its ID is stable for the object's whole lifetime.
type TrackedObject struct {
	// ID is a monotonically assigned identifier, unique and stable per object.
	ID int
	// Rect is the object's most recent bounding box.
	Rect cv.Rect
	// Age is the number of frames since the object first appeared.
	Age int
	// TimeSinceUpdate is the number of consecutive frames the object went
	// undetected; 0 means it was matched in the latest frame.
	TimeSinceUpdate int
	// Hits is the total number of frames in which the object was detected.
	Hits int
}

// DetectionBasedTracker follows detected objects across a sequence of frames,
// giving each a stable identity, in the spirit of OpenCV's
// cv::DetectionBasedTracker. On every [DetectionBasedTracker.Process] call it
// runs its Detector, greedily associates the new detections to existing tracks
// by intersection-over-union, updates matched tracks, spawns tracks for
// unmatched detections and ages tracks that were not matched, dropping them once
// they have been missing for too long. This bridges the gaps and identity swaps
// that a frame-by-frame detector alone would produce.
type DetectionBasedTracker struct {
	// Detector supplies detections for each processed frame. It must be set
	// before calling Process.
	Detector Detector
	// MinIoU is the minimum intersection-over-union for a detection to be
	// associated with an existing track. Values <= 0 default to 0.3.
	MinIoU float64
	// MaxTimeSinceUpdate is how many consecutive missed frames a track may
	// survive before it is dropped. Values < 0 default to 2.
	MaxTimeSinceUpdate int

	nextID int
	tracks []TrackedObject
}

// Reset clears all tracks and identity counters, returning the tracker to its
// initial state while keeping its configuration.
func (t *DetectionBasedTracker) Reset() {
	t.nextID = 0
	t.tracks = nil
}

func (t *DetectionBasedTracker) minIoU() float64 {
	if t.MinIoU <= 0 {
		return 0.3
	}
	return t.MinIoU
}

func (t *DetectionBasedTracker) maxMissing() int {
	if t.MaxTimeSinceUpdate < 0 {
		return 2
	}
	return t.MaxTimeSinceUpdate
}

// Process runs the detector on img and updates the tracker's state for one
// frame. It panics if no Detector has been set.
func (t *DetectionBasedTracker) Process(img *cv.Mat) {
	if t.Detector == nil {
		panic("objdetect: DetectionBasedTracker.Process with nil Detector")
	}
	dets := t.Detector.DetectMultiScale(img)
	t.update(dets)
}

// update associates detections with existing tracks and mutates the track list.
// It is separated from Process so tests can drive it with fixed detections.
func (t *DetectionBasedTracker) update(dets []cv.Rect) {
	type pair struct {
		track, det int
		iou        float64
	}
	var pairs []pair
	for ti := range t.tracks {
		for di := range dets {
			iou := RectIoU(t.tracks[ti].Rect, dets[di])
			if iou >= t.minIoU() {
				pairs = append(pairs, pair{ti, di, iou})
			}
		}
	}
	// Greedy matching: highest IoU first, each track and detection used once.
	sort.SliceStable(pairs, func(a, b int) bool { return pairs[a].iou > pairs[b].iou })
	trackTaken := make([]bool, len(t.tracks))
	detTaken := make([]bool, len(dets))
	for _, p := range pairs {
		if trackTaken[p.track] || detTaken[p.det] {
			continue
		}
		trackTaken[p.track] = true
		detTaken[p.det] = true
		tr := &t.tracks[p.track]
		tr.Rect = dets[p.det]
		tr.TimeSinceUpdate = 0
		tr.Age++
		tr.Hits++
	}

	// Age unmatched tracks; keep those still within the miss budget.
	kept := t.tracks[:0]
	for ti := range t.tracks {
		if trackTaken[ti] {
			kept = append(kept, t.tracks[ti])
			continue
		}
		tr := t.tracks[ti]
		tr.TimeSinceUpdate++
		tr.Age++
		if tr.TimeSinceUpdate <= t.maxMissing() {
			kept = append(kept, tr)
		}
	}
	t.tracks = kept

	// Spawn a track for every unmatched detection.
	for di := range dets {
		if detTaken[di] {
			continue
		}
		t.tracks = append(t.tracks, TrackedObject{
			ID:              t.nextID,
			Rect:            dets[di],
			Age:             1,
			TimeSinceUpdate: 0,
			Hits:            1,
		})
		t.nextID++
	}
}

// Objects returns a snapshot of every live track, ordered by ID. The returned
// slice is a copy; mutating it does not affect the tracker.
func (t *DetectionBasedTracker) Objects() []TrackedObject {
	out := make([]TrackedObject, len(t.tracks))
	copy(out, t.tracks)
	sort.Slice(out, func(a, b int) bool { return out[a].ID < out[b].ID })
	return out
}

// Visible returns only the tracks matched in the most recent frame
// (TimeSinceUpdate == 0), ordered by ID.
func (t *DetectionBasedTracker) Visible() []TrackedObject {
	var out []TrackedObject
	for _, tr := range t.tracks {
		if tr.TimeSinceUpdate == 0 {
			out = append(out, tr)
		}
	}
	sort.Slice(out, func(a, b int) bool { return out[a].ID < out[b].ID })
	return out
}
