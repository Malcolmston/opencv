package cv

import "testing"

// These tests exercise remaining public API surface for coverage and to guard
// against regressions in the less-travelled paths.

func TestEmptyAndSize(t *testing.T) {
	var nilMat *Mat
	if !nilMat.Empty() {
		t.Error("nil Mat should be Empty")
	}
	m := NewMat(3, 5, 1)
	if m.Empty() {
		t.Error("allocated Mat should not be Empty")
	}
	r, c := m.Size()
	if r != 3 || c != 5 {
		t.Errorf("Size = %d,%d", r, c)
	}
}

func TestSobelClamped(t *testing.T) {
	m := synthSquare(10, 3, 3, 4)
	out := Sobel(m, 1, 0, 3, 1, 128)
	// The clamped result must stay in range and vary from the neutral delta.
	varied := false
	for _, v := range out.Data {
		if v != 128 {
			varied = true
		}
	}
	if !varied {
		t.Error("Sobel produced a flat image on an edge")
	}
}

func TestScharrEdge(t *testing.T) {
	m := NewMat(6, 6, 1)
	for y := 0; y < 6; y++ {
		for x := 3; x < 6; x++ {
			m.Set(y, x, 0, 255)
		}
	}
	out := Scharr(m, 1, 0, 1, 0)
	max := uint8(0)
	for _, v := range out.Data {
		if v > max {
			max = v
		}
	}
	if max == 0 {
		t.Error("Scharr found no vertical edge response")
	}
}

func TestLaplacianKsize3(t *testing.T) {
	m := NewMat(5, 5, 1)
	m.SetTo(60)
	out := Laplacian(m, 3, 1, 0)
	for _, v := range out.Data {
		if v != 0 {
			t.Fatalf("laplacian ksize3 of constant = %d", v)
		}
	}
}

func TestRotate90CCW(t *testing.T) {
	m := grayFromValues(2, 3, []uint8{1, 2, 3, 4, 5, 6})
	ccw := Rotate(m, Rotate90CCW)
	if ccw.Rows != 3 || ccw.Cols != 2 {
		t.Fatalf("ccw dims %dx%d", ccw.Rows, ccw.Cols)
	}
	// Top-left after CCW is the old top-right = 3.
	if ccw.At(0, 0, 0) != 3 {
		t.Errorf("rotate90CCW top-left = %d, want 3", ccw.At(0, 0, 0))
	}
}

func TestMorphCloseFillsHole(t *testing.T) {
	// A 5x5 white block with a single dark hole in the centre.
	m := NewMat(7, 7, 1)
	for y := 1; y <= 5; y++ {
		for x := 1; x <= 5; x++ {
			m.Set(y, x, 0, 255)
		}
	}
	m.Set(3, 3, 0, 0)
	k := GetStructuringElement(MorphRect, 3, 3)
	closed := MorphologyEx(m, k, MorphClose, 1)
	if closed.At(3, 3, 0) != 255 {
		t.Errorf("close did not fill the hole, got %d", closed.At(3, 3, 0))
	}
}

func TestMorphTophatAndBlackhat(t *testing.T) {
	m := NewMat(9, 9, 1)
	m.SetTo(50)
	// A small bright speck for tophat.
	m.Set(4, 4, 0, 200)
	k := GetStructuringElement(MorphRect, 3, 3)
	top := MorphologyEx(m, k, MorphTophat, 1)
	if top.At(4, 4, 0) == 0 {
		t.Error("tophat should isolate the bright speck")
	}

	m2 := NewMat(9, 9, 1)
	m2.SetTo(200)
	m2.Set(4, 4, 0, 50) // dark speck
	black := MorphologyEx(m2, k, MorphBlackhat, 1)
	if black.At(4, 4, 0) == 0 {
		t.Error("blackhat should isolate the dark speck")
	}
}

func TestEllipseOutlineAndFill(t *testing.T) {
	m := NewMat(21, 21, 1)
	Ellipse(m, Point{10, 10}, 8, 4, 0, NewScalar(255), Filled)
	if m.At(10, 10, 0) != 255 {
		t.Error("filled ellipse centre not set")
	}
	// Point outside the ellipse extent stays clear.
	if m.At(0, 0, 0) != 0 {
		t.Error("ellipse overreached to corner")
	}

	o := NewMat(21, 21, 1)
	Ellipse(o, Point{10, 10}, 8, 4, 0, NewScalar(255), 1)
	// Centre of an outline ellipse is empty.
	if o.At(10, 10, 0) != 0 {
		t.Error("ellipse outline filled the centre")
	}
	set := 0
	for _, v := range o.Data {
		if v == 255 {
			set++
		}
	}
	if set == 0 {
		t.Error("ellipse outline drew nothing")
	}
}

func TestAdaptiveThresholdGaussian(t *testing.T) {
	m := NewMat(7, 7, 1)
	m.SetTo(100)
	m.Set(3, 3, 0, 220)
	out := AdaptiveThreshold(m, 255, AdaptiveThreshGaussianC, ThreshBinary, 5, 5)
	if out.At(3, 3, 0) != 255 {
		t.Errorf("adaptive gaussian bright spot = %d, want 255", out.At(3, 3, 0))
	}
}

func TestResizeLinearOnRGB(t *testing.T) {
	m := NewMat(2, 2, 3)
	m.SetTo(120)
	out := Resize(m, 5, 5, InterLinear)
	if out.Channels != 3 || out.At(2, 2, 1) != 120 {
		t.Errorf("rgb linear resize wrong: %v", out.AtPixel(2, 2))
	}
}

func TestWarpAffineLinearInterp(t *testing.T) {
	m := grayFromValues(2, 2, []uint8{0, 100, 100, 200})
	id := AffineMatrix{1, 0, 0, 0, 1, 0}
	out := WarpAffine(m, id, 2, 2, InterLinear)
	if out.At(1, 1, 0) != 200 {
		t.Errorf("warp linear identity (1,1) = %d, want 200", out.At(1, 1, 0))
	}
}

func TestJPEGEncodeDecode(t *testing.T) {
	m := NewMat(8, 8, 3)
	m.SetTo(128)
	data, err := IMEncode("jpeg", m)
	if err != nil {
		t.Fatalf("jpeg encode: %v", err)
	}
	back, err := IMDecode(data)
	if err != nil {
		t.Fatalf("jpeg decode: %v", err)
	}
	if back.Rows != 8 || back.Cols != 8 {
		t.Fatalf("jpeg dims %dx%d", back.Rows, back.Cols)
	}
	// JPEG is lossy; allow a small deviation from the constant grey.
	d := int(back.At(4, 4, 0)) - 128
	if d < -12 || d > 12 {
		t.Errorf("jpeg centre = %d, want ~128", back.At(4, 4, 0))
	}
}

func TestFloatMatAt(t *testing.T) {
	f := NewFloatMat(2, 3)
	f.Data[4] = 9.5
	if f.At(1, 1) != 9.5 {
		t.Errorf("FloatMat.At = %v", f.At(1, 1))
	}
}

func TestRGB2HSVGrayValue(t *testing.T) {
	// A mid grey has zero saturation and value equal to the level.
	hsv := CvtColor(mat1x1RGB(128, 128, 128), ColorRGB2HSV)
	if hsv.At(0, 0, 1) != 0 {
		t.Errorf("grey saturation = %d, want 0", hsv.At(0, 0, 1))
	}
	if hsv.At(0, 0, 2) != 128 {
		t.Errorf("grey value = %d, want 128", hsv.At(0, 0, 2))
	}
}
