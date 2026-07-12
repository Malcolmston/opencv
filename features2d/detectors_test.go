package features2d

import (
	"testing"

	cv "github.com/malcolmston/opencv"
)

func TestFastFeatureDetector(t *testing.T) {
	img := buildScene(120)
	kps := NewFastFeatureDetector(20).Detect(img)
	if len(kps) < 4 {
		t.Fatalf("FAST found too few corners: %d", len(kps))
	}
	for _, kp := range kps {
		if kp.Response <= 0 {
			t.Fatalf("expected positive FAST response, got %v", kp.Response)
		}
	}
	// A raw FAST call should find at least as many points (no non-max) ...
	raw := cv.FASTCorners(img, 20, true)
	if len(raw) != len(kps) {
		t.Fatalf("detector count %d != cv.FASTCorners count %d", len(kps), len(raw))
	}
}

func TestAgastFeatureDetector(t *testing.T) {
	img := buildScene(120)
	kps := NewAgastFeatureDetector(20).Detect(img)
	if len(kps) < 4 {
		t.Fatalf("AGAST found too few corners: %d", len(kps))
	}
	// AGAST score is the maximal qualifying threshold, so it must be >= the base.
	for _, kp := range kps {
		if kp.Response < 20 {
			t.Fatalf("AGAST score %v below base threshold", kp.Response)
		}
	}
}

func TestAgastAndFastAgreeOnCorners(t *testing.T) {
	img := buildScene(120)
	fast := NewFastFeatureDetector(15)
	agast := NewAgastFeatureDetector(15)
	fset := map[cv.Point]bool{}
	for _, kp := range fast.Detect(img) {
		fset[kp.Pt] = true
	}
	// Every AGAST corner (before NMS differences) must be a FAST-9 corner too,
	// since they share the OAST 9_16 mask. Disable NMS to compare the raw sets.
	agast.NonmaxSuppression = false
	fast.NonmaxSuppression = false
	fset = map[cv.Point]bool{}
	for _, kp := range fast.Detect(img) {
		fset[kp.Pt] = true
	}
	for _, kp := range agast.Detect(img) {
		if !fset[kp.Pt] {
			t.Fatalf("AGAST corner %v not detected by FAST", kp.Pt)
		}
	}
}

func TestGFTTDetector(t *testing.T) {
	img := buildScene(120)
	kps := NewGFTTDetector(50).Detect(img)
	if len(kps) < 4 {
		t.Fatalf("GFTT found too few corners: %d", len(kps))
	}
	if len(kps) > 50 {
		t.Fatalf("GFTT exceeded maxCorners: %d", len(kps))
	}
}

func TestSimpleBlobDetector(t *testing.T) {
	img, centers := buildBlobs(140)
	det := NewSimpleBlobDetector()
	det.MinArea = 20
	det.MaxArea = 2000
	kps := det.Detect(img)
	if len(kps) < len(centers) {
		t.Fatalf("expected >= %d blobs, got %d", len(centers), len(kps))
	}
	// Each true centre should have a detection nearby.
	for _, c := range centers {
		found := false
		for _, kp := range kps {
			if abs(kp.Pt.X-c.X) <= 4 && abs(kp.Pt.Y-c.Y) <= 4 {
				found = true
				break
			}
		}
		if !found {
			t.Fatalf("no blob detected near %v", c)
		}
	}
}

func TestDetectorsAreDeterministic(t *testing.T) {
	img := buildScene(120)
	dets := []Detector{
		NewFastFeatureDetector(20),
		NewAgastFeatureDetector(20),
		NewGFTTDetector(40),
	}
	for _, d := range dets {
		a := d.Detect(img)
		b := d.Detect(img)
		if len(a) != len(b) {
			t.Fatalf("%T non-deterministic count", d)
		}
		for i := range a {
			if a[i] != b[i] {
				t.Fatalf("%T non-deterministic keypoint %d", d, i)
			}
		}
	}
}
