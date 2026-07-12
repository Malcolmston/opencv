package features2d

import (
	"testing"

	cv "github.com/malcolmston/opencv"
)

func TestComputeRecallPrecisionCurve(t *testing.T) {
	// Two queries, two candidates each. Correct matches have the smaller
	// distance, so sweeping the threshold should reach recall 1 at precision 1.
	matches := [][]DMatch{
		{{Distance: 1}, {Distance: 5}},
		{{Distance: 2}, {Distance: 6}},
	}
	correct := [][]bool{
		{true, false},
		{true, false},
	}
	curve := ComputeRecallPrecisionCurve(matches, correct)
	if len(curve) != 4 {
		t.Fatalf("expected 4 curve points, got %d", len(curve))
	}
	// After the two closest (correct) detections, recall is 1 and precision 1.
	if curve[1].Recall != 1 || curve[1].Precision != 1 {
		t.Fatalf("expected recall=precision=1 at point 1, got %+v", curve[1])
	}
	// The final point includes the two false positives: recall stays 1,
	// precision drops to 0.5.
	last := curve[len(curve)-1]
	if last.Recall != 1 || last.Precision != 0.5 {
		t.Fatalf("expected recall=1 precision=0.5 at end, got %+v", last)
	}
}

func TestComputeRecallPrecisionCurveShapePanic(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("expected panic on shape mismatch")
		}
	}()
	ComputeRecallPrecisionCurve([][]DMatch{{{Distance: 1}}}, nil)
}

func TestEvaluateFeatureDetectorTranslation(t *testing.T) {
	dx, dy := 6, 5
	img1, img2 := translatedPair(dx, dy)
	// img2 = img1 shifted so a point at img1 (x,y) is at img2 (x-dx, y-dy).
	h := cv.PerspectiveMatrix{
		1, 0, float64(-dx),
		0, 1, float64(-dy),
		0, 0, 1,
	}
	det := NewFastFeatureDetector(20)
	rep, corr := EvaluateFeatureDetector(img1, img2, h, det, 2)
	if corr == 0 {
		t.Fatalf("expected correspondences under a pure translation, got 0")
	}
	if rep <= 0.3 {
		t.Fatalf("repeatability unexpectedly low: %.3f (corr=%d)", rep, corr)
	}
	t.Logf("repeatability=%.3f correspondences=%d", rep, corr)
}

func TestEvaluateFeatureDetectorIdentity(t *testing.T) {
	img := buildScene(120)
	identity := cv.PerspectiveMatrix{1, 0, 0, 0, 1, 0, 0, 0, 1}
	det := NewAgastFeatureDetector(20)
	rep, corr := EvaluateFeatureDetector(img, img, identity, det, 1)
	if rep < 0.99 {
		t.Fatalf("identity repeatability should be ~1, got %.3f (corr=%d)", rep, corr)
	}
}
