package objdetect

import (
	"testing"

	cv "github.com/malcolmston/opencv"
)

// scriptedDetector returns a preset list of rectangles per frame, ignoring the
// image, so tracker behaviour can be tested deterministically.
type scriptedDetector struct {
	frames [][]cv.Rect
	i      int
}

func (s *scriptedDetector) DetectMultiScale(_ *cv.Mat) []cv.Rect {
	if s.i >= len(s.frames) {
		return nil
	}
	r := s.frames[s.i]
	s.i++
	return r
}

func TestDetectionBasedTrackerIdentity(t *testing.T) {
	frames := [][]cv.Rect{
		{{X: 10, Y: 10, Width: 20, Height: 20}},                                          // f0: object A appears
		{{X: 12, Y: 10, Width: 20, Height: 20}},                                          // f1: A moves right
		{{X: 14, Y: 10, Width: 20, Height: 20}, {X: 100, Y: 100, Width: 20, Height: 20}}, // f2: A + new B
	}
	det := &scriptedDetector{frames: frames}
	tr := &DetectionBasedTracker{Detector: det, MinIoU: 0.3, MaxTimeSinceUpdate: 2}
	dummy := cv.NewMat(1, 1, 1)

	tr.Process(dummy)
	tr.Process(dummy)
	tr.Process(dummy)

	objs := tr.Objects()
	if len(objs) != 2 {
		t.Fatalf("after 3 frames expected 2 tracks, got %d (%v)", len(objs), objs)
	}
	// Track A keeps ID 0 across the three frames and was hit 3 times.
	if objs[0].ID != 0 || objs[0].Hits != 3 {
		t.Fatalf("track A = %+v, want ID 0 with 3 hits", objs[0])
	}
	if objs[0].Rect.X != 14 {
		t.Fatalf("track A rect not updated to latest position: %+v", objs[0].Rect)
	}
	// Track B is the new object with ID 1.
	if objs[1].ID != 1 || objs[1].Hits != 1 {
		t.Fatalf("track B = %+v, want ID 1 with 1 hit", objs[1])
	}
}

func TestDetectionBasedTrackerDropAfterMisses(t *testing.T) {
	frames := [][]cv.Rect{
		{{X: 10, Y: 10, Width: 20, Height: 20}}, // f0: appears
		nil,                                     // f1: miss 1
		nil,                                     // f2: miss 2
		nil,                                     // f3: miss 3 -> dropped
	}
	det := &scriptedDetector{frames: frames}
	tr := &DetectionBasedTracker{Detector: det, MaxTimeSinceUpdate: 2}
	dummy := cv.NewMat(1, 1, 1)

	tr.Process(dummy)
	if len(tr.Objects()) != 1 {
		t.Fatalf("after f0 expected 1 track, got %d", len(tr.Objects()))
	}
	tr.Process(dummy) // miss 1
	tr.Process(dummy) // miss 2 (still within budget of 2)
	if len(tr.Objects()) != 1 {
		t.Fatalf("track should survive 2 misses, got %d tracks", len(tr.Objects()))
	}
	if len(tr.Visible()) != 0 {
		t.Fatalf("no track should be visible during misses, got %d", len(tr.Visible()))
	}
	tr.Process(dummy) // miss 3 -> exceeds budget
	if len(tr.Objects()) != 0 {
		t.Fatalf("track should be dropped after 3 misses, got %d", len(tr.Objects()))
	}
}

func TestDetectionBasedTrackerNilDetectorPanics(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("expected panic with nil Detector")
		}
	}()
	tr := &DetectionBasedTracker{}
	tr.Process(cv.NewMat(1, 1, 1))
}

func TestDetectionBasedTrackerReset(t *testing.T) {
	det := &scriptedDetector{frames: [][]cv.Rect{{{X: 0, Y: 0, Width: 5, Height: 5}}}}
	tr := &DetectionBasedTracker{Detector: det}
	tr.Process(cv.NewMat(1, 1, 1))
	if len(tr.Objects()) == 0 {
		t.Fatal("expected a track before reset")
	}
	tr.Reset()
	if len(tr.Objects()) != 0 {
		t.Fatal("Reset should clear all tracks")
	}
}
