package cv

import "testing"

func mat1x1RGB(r, g, b uint8) *Mat {
	m := NewMat(1, 1, 3)
	m.Set(0, 0, 0, r)
	m.Set(0, 0, 1, g)
	m.Set(0, 0, 2, b)
	return m
}

func TestRGB2GrayKnownValues(t *testing.T) {
	cases := []struct {
		r, g, b uint8
		want    uint8
	}{
		{255, 255, 255, 255},
		{0, 0, 0, 0},
		{255, 0, 0, 76},  // 0.299*255 = 76.245
		{0, 255, 0, 150}, // 0.587*255 = 149.685
		{0, 0, 255, 29},  // 0.114*255 = 29.07
	}
	for _, c := range cases {
		g := CvtColor(mat1x1RGB(c.r, c.g, c.b), ColorRGB2Gray)
		if g.At(0, 0, 0) != c.want {
			t.Errorf("gray(%d,%d,%d) = %d, want %d", c.r, c.g, c.b, g.At(0, 0, 0), c.want)
		}
	}
}

func TestGray2RGB(t *testing.T) {
	g := NewMat(1, 1, 1)
	g.Set(0, 0, 0, 123)
	rgb := CvtColor(g, ColorGray2RGB)
	if rgb.At(0, 0, 0) != 123 || rgb.At(0, 0, 1) != 123 || rgb.At(0, 0, 2) != 123 {
		t.Errorf("gray2rgb = %v", rgb.AtPixel(0, 0))
	}
}

func TestRGB2BGRSwap(t *testing.T) {
	bgr := CvtColor(mat1x1RGB(1, 2, 3), ColorRGB2BGR)
	if bgr.At(0, 0, 0) != 3 || bgr.At(0, 0, 1) != 2 || bgr.At(0, 0, 2) != 1 {
		t.Errorf("rgb2bgr = %v, want [3 2 1]", bgr.AtPixel(0, 0))
	}
	// Swapping twice is identity.
	back := CvtColor(bgr, ColorBGR2RGB)
	if back.At(0, 0, 0) != 1 || back.At(0, 0, 2) != 3 {
		t.Error("double swap not identity")
	}
}

func TestRGB2HSVKnownValues(t *testing.T) {
	// Pure red: H=0, S=255, V=255.
	hsv := CvtColor(mat1x1RGB(255, 0, 0), ColorRGB2HSV)
	if hsv.At(0, 0, 0) != 0 || hsv.At(0, 0, 1) != 255 || hsv.At(0, 0, 2) != 255 {
		t.Errorf("red HSV = %v, want [0 255 255]", hsv.AtPixel(0, 0))
	}
	// Pure green: H=120deg -> 60 in OpenCV units.
	hsv = CvtColor(mat1x1RGB(0, 255, 0), ColorRGB2HSV)
	if hsv.At(0, 0, 0) != 60 {
		t.Errorf("green H = %d, want 60", hsv.At(0, 0, 0))
	}
	// Pure blue: H=240deg -> 120.
	hsv = CvtColor(mat1x1RGB(0, 0, 255), ColorRGB2HSV)
	if hsv.At(0, 0, 0) != 120 {
		t.Errorf("blue H = %d, want 120", hsv.At(0, 0, 0))
	}
	// White: S=0, V=255.
	hsv = CvtColor(mat1x1RGB(255, 255, 255), ColorRGB2HSV)
	if hsv.At(0, 0, 1) != 0 || hsv.At(0, 0, 2) != 255 {
		t.Errorf("white HSV = %v", hsv.AtPixel(0, 0))
	}
}

func TestHSV2RGBRoundTripPrimaries(t *testing.T) {
	for _, rgb := range [][3]uint8{{255, 0, 0}, {0, 255, 0}, {0, 0, 255}, {255, 255, 255}} {
		src := mat1x1RGB(rgb[0], rgb[1], rgb[2])
		back := CvtColor(CvtColor(src, ColorRGB2HSV), ColorHSV2RGB)
		for c := 0; c < 3; c++ {
			d := int(back.At(0, 0, c)) - int(src.At(0, 0, c))
			if d < -4 || d > 4 {
				t.Errorf("HSV round-trip %v channel %d: got %d want ~%d", rgb, c, back.At(0, 0, c), src.At(0, 0, c))
			}
		}
	}
}
