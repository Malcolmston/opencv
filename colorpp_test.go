package cv

import "testing"

func maxChannelDiff(a, b *Mat) int {
	m := 0
	for i := range a.Data {
		d := int(a.Data[i]) - int(b.Data[i])
		if d < 0 {
			d = -d
		}
		if d > m {
			m = d
		}
	}
	return m
}

func rgbSamples() *Mat {
	m := NewMat(1, 6, 3)
	colors := [][3]uint8{
		{0, 0, 0}, {255, 255, 255}, {200, 30, 40},
		{40, 180, 60}, {50, 60, 220}, {128, 200, 90},
	}
	for i, c := range colors {
		m.SetPixel(0, i, c[:])
	}
	return m
}

func TestLabRoundTrip(t *testing.T) {
	src := rgbSamples()
	lab := CvtColor(src, ColorRGB2Lab)
	back := CvtColor(lab, ColorLab2RGB)
	if d := maxChannelDiff(src, back); d > 6 {
		t.Errorf("Lab round-trip max diff = %d, want <= 6", d)
	}
}

func TestYCrCbRoundTrip(t *testing.T) {
	src := rgbSamples()
	yc := CvtColor(src, ColorRGB2YCrCb)
	back := CvtColor(yc, ColorYCrCb2RGB)
	if d := maxChannelDiff(src, back); d > 2 {
		t.Errorf("YCrCb round-trip max diff = %d, want <= 2", d)
	}
}

func TestHLSRoundTrip(t *testing.T) {
	src := rgbSamples()
	hls := CvtColor(src, ColorRGB2HLS)
	back := CvtColor(hls, ColorHLS2RGB)
	if d := maxChannelDiff(src, back); d > 4 {
		t.Errorf("HLS round-trip max diff = %d, want <= 4", d)
	}
}

func TestYCrCbGrayHasNeutralChroma(t *testing.T) {
	// A grey pixel has Cr = Cb = 128.
	yc := CvtColor(mat1x1RGB(100, 100, 100), ColorRGB2YCrCb)
	if yc.At(0, 0, 0) != 100 {
		t.Errorf("Y of grey = %d, want 100", yc.At(0, 0, 0))
	}
	if yc.At(0, 0, 1) != 128 || yc.At(0, 0, 2) != 128 {
		t.Errorf("chroma of grey = (%d,%d), want (128,128)", yc.At(0, 0, 1), yc.At(0, 0, 2))
	}
}
