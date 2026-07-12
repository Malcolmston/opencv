package cv

import "testing"

func TestGetStructuringElementRect(t *testing.T) {
	k := GetStructuringElement(MorphRect, 3, 3)
	for _, v := range k.Data {
		if v != 1 {
			t.Fatal("rect element must be all ones")
		}
	}
}

func TestGetStructuringElementCross(t *testing.T) {
	k := GetStructuringElement(MorphCross, 3, 3)
	// Centre row and column set; corners clear.
	if k.At(0, 0, 0) != 0 || k.At(2, 2, 0) != 0 {
		t.Error("cross corners should be 0")
	}
	if k.At(1, 1, 0) != 1 || k.At(0, 1, 0) != 1 || k.At(1, 0, 0) != 1 {
		t.Error("cross centre cross should be 1")
	}
}

func TestDilateSinglePixel(t *testing.T) {
	m := NewMat(5, 5, 1)
	m.Set(2, 2, 0, 255)
	k := GetStructuringElement(MorphRect, 3, 3)
	out := Dilate(m, k, 1)
	// The 3x3 block around the centre should be white.
	for y := 1; y <= 3; y++ {
		for x := 1; x <= 3; x++ {
			if out.At(y, x, 0) != 255 {
				t.Fatalf("dilate: (%d,%d) should be 255", y, x)
			}
		}
	}
	// Corners of the image remain 0.
	if out.At(0, 0, 0) != 0 {
		t.Error("dilate leaked to corner")
	}
}

func TestErodeShrinksBlock(t *testing.T) {
	m := NewMat(5, 5, 1)
	for y := 1; y <= 3; y++ {
		for x := 1; x <= 3; x++ {
			m.Set(y, x, 0, 255)
		}
	}
	k := GetStructuringElement(MorphRect, 3, 3)
	out := Erode(m, k, 1)
	// Only the centre survives erosion of a 3x3 block.
	if out.At(2, 2, 0) != 255 {
		t.Error("erode: centre should survive")
	}
	if out.At(1, 1, 0) != 0 {
		t.Error("erode: block edge should vanish")
	}
}

func TestErodeDilateAreDualUnderOpen(t *testing.T) {
	// Opening a single isolated pixel removes it entirely.
	m := NewMat(5, 5, 1)
	m.Set(2, 2, 0, 255)
	k := GetStructuringElement(MorphRect, 3, 3)
	opened := MorphologyEx(m, k, MorphOpen, 1)
	for _, v := range opened.Data {
		if v != 0 {
			t.Fatal("opening should remove an isolated pixel")
		}
	}
}

func TestMorphGradientOutlinesBlock(t *testing.T) {
	m := NewMat(7, 7, 1)
	for y := 2; y <= 4; y++ {
		for x := 2; x <= 4; x++ {
			m.Set(y, x, 0, 255)
		}
	}
	k := GetStructuringElement(MorphRect, 3, 3)
	grad := MorphologyEx(m, k, MorphGradient, 1)
	// Centre of a solid block has zero gradient; the boundary is non-zero.
	if grad.At(3, 3, 0) != 0 {
		t.Errorf("gradient centre = %d, want 0", grad.At(3, 3, 0))
	}
	if grad.At(2, 2, 0) == 0 {
		t.Error("gradient boundary should be non-zero")
	}
}
