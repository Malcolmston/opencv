package features2d

import "testing"

// TestScaleSpaceDetectWrappers exercises the Detect-only entry points of the
// scale-space detectors (the DetectAndCompute paths are covered elsewhere).
func TestScaleSpaceDetectWrappers(t *testing.T) {
	img := buildScene(96)
	if got := len(NewSIFT(0).Detect(img)); got == 0 {
		t.Fatal("SIFT.Detect returned no keypoints")
	}
	if got := len(NewKAZE(0).Detect(img)); got == 0 {
		t.Fatal("KAZE.Detect returned no keypoints")
	}
	if got := len(NewAKAZE(0).Detect(img)); got == 0 {
		t.Fatal("AKAZE.Detect returned no keypoints")
	}
}

func TestBOWWordAssignments(t *testing.T) {
	vocab := NewFloatDescriptors([][]float64{{0, 0}, {10, 10}})
	ext := NewBOWImgDescriptorExtractor(NewBFMatcher(NormL2))
	ext.SetVocabulary(vocab)
	desc := NewFloatDescriptors([][]float64{{0.1, 0.1}, {9.9, 9.9}, {0, 0}})
	got := ext.wordAssignments(desc)
	want := []int{0, 1, 0}
	if len(got) != len(want) {
		t.Fatalf("got %d assignments, want %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("assignment %d = %d, want %d", i, got[i], want[i])
		}
	}
}
