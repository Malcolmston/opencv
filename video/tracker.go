package video

import (
	cv "github.com/malcolmston/opencv"
)

// TrackerFeaturePyrLK is a stateful sparse point tracker built on
// [CalcOpticalFlowPyrLK]. It remembers the previous frame and the current set of
// tracked feature points; each [TrackerFeaturePyrLK.Update] advances every point
// to the next frame with pyramidal Lucas-Kanade and drops points whose track was
// lost. It mirrors the role of OpenCV's sparse PyrLK-based feature trackers.
type TrackerFeaturePyrLK struct {
	// WinSize is the Lucas-Kanade window size (odd, in pixels).
	WinSize int
	// MaxLevel is the number of pyramid levels above the base image.
	MaxLevel int

	prev   *cv.Mat // grayscale previous frame
	points []cv.Point
}

// NewTrackerFeaturePyrLK creates a tracker with the given Lucas-Kanade window
// size and pyramid depth. winSize must be >= 1 (rounded up to odd) and maxLevel
// must be >= 0.
func NewTrackerFeaturePyrLK(winSize, maxLevel int) *TrackerFeaturePyrLK {
	if winSize < 1 {
		panic("video: NewTrackerFeaturePyrLK requires winSize >= 1")
	}
	if winSize%2 == 0 {
		winSize++
	}
	if maxLevel < 0 {
		panic("video: NewTrackerFeaturePyrLK requires maxLevel >= 0")
	}
	return &TrackerFeaturePyrLK{WinSize: winSize, MaxLevel: maxLevel}
}

// Init seeds the tracker with the first frame and the feature points to follow.
// The points are copied. A subsequent [TrackerFeaturePyrLK.Update] tracks them
// into the next frame.
func (t *TrackerFeaturePyrLK) Init(frame *cv.Mat, points []cv.Point) {
	if frame == nil || frame.Empty() {
		panic("video: TrackerFeaturePyrLK.Init requires a non-empty frame")
	}
	t.prev = toGray(frame)
	t.points = make([]cv.Point, len(points))
	copy(t.points, points)
}

// Update tracks the current point set from the previous frame into frame. It
// returns the new location of every currently tracked point together with a
// status slice, both aligned with the tracker's point set as it was on entry
// (element i corresponds to point i). Points whose flow was lost (status false)
// are removed from the tracker's internal state afterwards, so the set shrinks
// monotonically across frames until re-seeded with [TrackerFeaturePyrLK.Init].
// frame becomes the new reference frame.
//
// It panics if the tracker has not been initialised. When no points remain the
// returned slices are empty and frame is still stored as the reference.
func (t *TrackerFeaturePyrLK) Update(frame *cv.Mat) (points []cv.Point, status []bool) {
	if t.prev == nil {
		panic("video: TrackerFeaturePyrLK.Update called before Init")
	}
	if frame == nil || frame.Empty() {
		panic("video: TrackerFeaturePyrLK.Update requires a non-empty frame")
	}
	gray := toGray(frame)
	if len(t.points) == 0 {
		t.prev = gray
		return nil, nil
	}
	nextPts, ok, _ := CalcOpticalFlowPyrLK(t.prev, gray, t.points, t.WinSize, t.MaxLevel)
	kept := make([]cv.Point, 0, len(nextPts))
	for i := range nextPts {
		if ok[i] {
			kept = append(kept, nextPts[i])
		}
	}
	t.prev = gray
	t.points = kept
	return nextPts, ok
}

// Points returns a copy of the tracker's current point set.
func (t *TrackerFeaturePyrLK) Points() []cv.Point {
	out := make([]cv.Point, len(t.points))
	copy(out, t.points)
	return out
}
