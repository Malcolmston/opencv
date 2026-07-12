package features2d

import (
	"testing"

	cv "github.com/malcolmston/opencv"
)

func TestSIFTDetectsAndDescribes(t *testing.T) {
	img := buildScene(120)
	kps, desc := NewSIFT(0).DetectAndCompute(img)
	if len(kps) < 10 {
		t.Fatalf("expected many SIFT keypoints, got %d", len(kps))
	}
	if len(desc) != len(kps) {
		t.Fatalf("descriptor count %d != keypoint count %d", len(desc), len(kps))
	}
	if len(desc[0]) != 128 {
		t.Fatalf("expected 128-d descriptors, got %d", len(desc[0]))
	}
	// Descriptors are unit length (after clamping/renormalisation).
	var norm float64
	for _, v := range desc[0] {
		norm += v * v
	}
	if norm < 0.9 || norm > 1.1 {
		t.Fatalf("descriptor not unit length: |d|^2=%.3f", norm)
	}
	for _, kp := range kps {
		if kp.Angle < 0 || kp.Angle >= 360 {
			t.Fatalf("angle out of range: %v", kp.Angle)
		}
		if kp.Size <= 0 {
			t.Fatalf("non-positive size: %v", kp.Size)
		}
	}
}

func TestSIFTDeterministic(t *testing.T) {
	img := buildScene(120)
	s := NewSIFT(50)
	kA, dA := s.DetectAndCompute(img)
	kB, dB := s.DetectAndCompute(img)
	if len(kA) != len(kB) {
		t.Fatalf("non-deterministic count: %d vs %d", len(kA), len(kB))
	}
	for i := range kA {
		if kA[i] != kB[i] {
			t.Fatalf("keypoint %d differs: %+v vs %+v", i, kA[i], kB[i])
		}
		for j := range dA[i] {
			if dA[i][j] != dB[i][j] {
				t.Fatalf("descriptor %d differs at %d", i, j)
			}
		}
	}
}

func TestSIFTMatchTranslated(t *testing.T) {
	dx, dy := 5, 4
	img1, img2 := translatedPair(dx, dy)
	s := NewSIFT(0)
	kp1, d1 := s.DetectAndCompute(img1)
	kp2, d2 := s.DetectAndCompute(img2)
	if len(kp1) < 8 || len(kp2) < 8 {
		t.Fatalf("too few keypoints: %d and %d", len(kp1), len(kp2))
	}
	matcher := &BFMatcher{Norm: NormL2}
	knn := matcher.KnnMatch(NewFloatDescriptors(d1), NewFloatDescriptors(d2), 2)
	good := RatioTest(knn, 0.8)
	if len(good) < 4 {
		t.Fatalf("too few good SIFT matches: %d", len(good))
	}
	consistent := 0
	for _, m := range good {
		ddx := kp2[m.TrainIdx].Pt.X - kp1[m.QueryIdx].Pt.X
		ddy := kp2[m.TrainIdx].Pt.Y - kp1[m.QueryIdx].Pt.Y
		if abs(ddx-(-dx)) <= 2 && abs(ddy-(-dy)) <= 2 {
			consistent++
		}
	}
	if consistent*2 <= len(good) {
		t.Fatalf("SIFT matches not majority consistent: %d of %d", consistent, len(good))
	}
	t.Logf("sift good=%d consistent=%d kp1=%d kp2=%d", len(good), consistent, len(kp1), len(kp2))
}

func TestSIFTImplementsDetector(t *testing.T) {
	var _ Detector = NewSIFT(0)
	var _ Detector = NewORB(0)
	// Touch cv to keep the import meaningful.
	_ = cv.NewScalar(1)
}
