package features2d

import (
	"testing"

	cv "github.com/malcolmston/opencv"
)

func sampleKeypoints() []KeyPoint {
	return []KeyPoint{
		{Pt: cv.Point{X: 5, Y: 5}, Size: 10, Angle: 0, Response: 0.1},
		{Pt: cv.Point{X: 50, Y: 50}, Size: 20, Angle: 90, Response: 0.9},
		{Pt: cv.Point{X: 2, Y: 40}, Size: 15, Angle: 45, Response: 0.5},
		{Pt: cv.Point{X: 60, Y: 60}, Size: 30, Angle: 10, Response: 0.7},
	}
}

func TestKeyPointsFilterRunByImageBorder(t *testing.T) {
	var f KeyPointsFilter
	kps := sampleKeypoints()
	out := f.RunByImageBorder(kps, 70, 70, 5)
	for _, kp := range out {
		if kp.Pt.X < 5 || kp.Pt.Y < 5 || kp.Pt.X >= 65 || kp.Pt.Y >= 65 {
			t.Fatalf("kept keypoint inside border: %v", kp.Pt)
		}
	}
	// (2,40) is within 5px of the left edge and must be dropped.
	for _, kp := range out {
		if kp.Pt.X == 2 && kp.Pt.Y == 40 {
			t.Fatal("border keypoint (2,40) should have been removed")
		}
	}
}

func TestKeyPointsFilterRunByKeypointSize(t *testing.T) {
	var f KeyPointsFilter
	out := f.RunByKeypointSize(sampleKeypoints(), 15, 25)
	if len(out) != 2 {
		t.Fatalf("expected 2 keypoints in [15,25), got %d", len(out))
	}
	for _, kp := range out {
		if kp.Size < 15 || kp.Size >= 25 {
			t.Fatalf("size out of range: %v", kp.Size)
		}
	}
}

func TestKeyPointsFilterRetainBest(t *testing.T) {
	var f KeyPointsFilter
	out := f.RetainBest(sampleKeypoints(), 2)
	if len(out) != 2 {
		t.Fatalf("expected 2 kept, got %d", len(out))
	}
	if out[0].Response != 0.9 || out[1].Response != 0.7 {
		t.Fatalf("RetainBest wrong order: %.1f, %.1f", out[0].Response, out[1].Response)
	}
}

func TestKeyPointsFilterRetainBestTies(t *testing.T) {
	var f KeyPointsFilter
	kps := []KeyPoint{
		{Pt: cv.Point{X: 1}, Response: 0.5},
		{Pt: cv.Point{X: 2}, Response: 0.5},
		{Pt: cv.Point{X: 3}, Response: 0.5},
	}
	// Requesting 1 must keep all three because they tie.
	out := f.RetainBest(kps, 1)
	if len(out) != 3 {
		t.Fatalf("expected all tied keypoints kept, got %d", len(out))
	}
}

func TestKeyPointsFilterRemoveDuplicated(t *testing.T) {
	var f KeyPointsFilter
	kps := []KeyPoint{
		{Pt: cv.Point{X: 1, Y: 1}, Size: 5, Angle: 0},
		{Pt: cv.Point{X: 1, Y: 1}, Size: 5, Angle: 0}, // duplicate
		{Pt: cv.Point{X: 1, Y: 1}, Size: 5, Angle: 90},
	}
	out := f.RemoveDuplicated(kps)
	if len(out) != 2 {
		t.Fatalf("expected 2 unique keypoints, got %d", len(out))
	}
}
