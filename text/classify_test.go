package text

import (
	"testing"

	cv "github.com/malcolmston/opencv"
)

// glyph builds a small single-channel template from a row-major on/off pattern,
// with "on" pixels set to 255 and "off" pixels to 0.
func glyph(rows, cols int, pattern []int) *cv.Mat {
	m := cv.NewMat(rows, cols, 1)
	for i, v := range pattern {
		if v != 0 {
			m.Data[i] = 255
		}
	}
	return m
}

func TestNearestGlyphClassifier(t *testing.T) {
	// A 3x3 "0" (hollow ring) and "1" (centre column).
	zero := glyph(3, 3, []int{
		1, 1, 1,
		1, 0, 1,
		1, 1, 1,
	})
	one := glyph(3, 3, []int{
		0, 1, 0,
		0, 1, 0,
		0, 1, 0,
	})
	clf := NewNearestGlyphClassifier(3, 3, 127, map[string]*cv.Mat{"0": zero, "1": one})

	if label, dist := clf.Classify(zero); label != "0" || dist != 0 {
		t.Errorf("Classify(zero) = %q dist %d, want \"0\" dist 0", label, dist)
	}
	if label, dist := clf.Classify(one); label != "1" || dist != 0 {
		t.Errorf("Classify(one) = %q dist %d, want \"1\" dist 0", label, dist)
	}

	// A noisy "1" with one flipped pixel still classifies as "1".
	noisyOne := glyph(3, 3, []int{
		0, 1, 0,
		0, 1, 1,
		0, 1, 0,
	})
	if label, _ := clf.Classify(noisyOne); label != "1" {
		t.Errorf("Classify(noisyOne) = %q, want \"1\"", label)
	}
}

func TestNearestGlyphClassifierResizes(t *testing.T) {
	// Template at 4x4, query at 8x8 of the same shape (a solid block): resizing
	// should still match exactly.
	solid4 := cv.NewMat(4, 4, 1)
	solid4.SetTo(255)
	blank4 := cv.NewMat(4, 4, 1)
	clf := NewNearestGlyphClassifier(4, 4, 127, map[string]*cv.Mat{"full": solid4, "empty": blank4})

	solid8 := cv.NewMat(8, 8, 1)
	solid8.SetTo(255)
	if label, dist := clf.Classify(solid8); label != "full" || dist != 0 {
		t.Errorf("Classify(solid8) = %q dist %d, want \"full\" dist 0", label, dist)
	}
}

func TestNearestGlyphClassifierPanics(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Errorf("expected panic on empty template set")
		}
	}()
	NewNearestGlyphClassifier(3, 3, 127, map[string]*cv.Mat{})
}
