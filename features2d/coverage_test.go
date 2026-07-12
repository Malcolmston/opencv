package features2d

import (
	"testing"

	cv "github.com/malcolmston/opencv"
)

func TestDescriptorsHelpers(t *testing.T) {
	b := NewBinaryDescriptors([][]byte{{1}, {2}, {3}})
	if !b.IsBinary() || b.Len() != 3 {
		t.Fatalf("binary descriptors: IsBinary=%v Len=%d", b.IsBinary(), b.Len())
	}
	f := NewFloatDescriptors([][]float64{{1, 2}})
	if f.IsBinary() || f.Len() != 1 {
		t.Fatalf("float descriptors: IsBinary=%v Len=%d", f.IsBinary(), f.Len())
	}
}

func TestNewBRIEFCustomPattern(t *testing.T) {
	b := NewBRIEF(21, 128)
	if b.patchSize() != 21 {
		t.Fatalf("patchSize=%d want 21", b.patchSize())
	}
	if len(b.pat()) != 128 {
		t.Fatalf("pattern length=%d want 128", len(b.pat()))
	}
	img := buildScene(60)
	_, desc := b.Compute(img, []KeyPoint{{Pt: cv.Point{X: 30, Y: 30}, Angle: -1}})
	if len(desc[0]) != 128/8 {
		t.Fatalf("descriptor bytes=%d want %d", len(desc[0]), 128/8)
	}
}

func TestNewBRIEFRejectsBadBits(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("expected panic for non-multiple-of-8 bit count")
		}
	}()
	NewBRIEF(31, 100)
}

func TestORBCustomFieldsAndColorInput(t *testing.T) {
	gray := buildScene(110)
	// Feed a 3-channel image to exercise the grayscale-conversion path, with
	// explicit non-default ORB parameters.
	color := toColor(gray)
	orb := &ORB{NFeatures: 40, FastThreshold: 15, PatchSize: 25}
	kps, desc := orb.DetectAndCompute(color)
	if len(kps) == 0 || len(kps) > 40 {
		t.Fatalf("expected 1..40 keypoints, got %d", len(kps))
	}
	if len(desc) != len(kps) {
		t.Fatalf("desc/kp mismatch: %d vs %d", len(desc), len(kps))
	}
}

func TestKnnMatchRejectsK(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("expected panic for k < 1")
		}
	}()
	NewBFMatcher(NormHamming).KnnMatch(
		NewBinaryDescriptors([][]byte{{1}}),
		NewBinaryDescriptors([][]byte{{1}}), 0)
}
