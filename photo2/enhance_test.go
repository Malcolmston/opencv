package photo2

import (
	"testing"

	cv "github.com/malcolmston/opencv"
)

func TestGammaCorrection(t *testing.T) {
	img := mkGray(t, 1, 3, []uint8{0, 64, 255})
	// gamma 1 is identity.
	id := GammaCorrection(img, 1)
	for i := range img.Data {
		if id.Data[i] != img.Data[i] {
			t.Fatalf("gamma 1 not identity")
		}
	}
	// gamma 2: 64 -> 255*(64/255)^2 = 16.06 -> 16.
	g := GammaCorrection(img, 2)
	if g.Data[0] != 0 || g.Data[2] != 255 {
		t.Fatalf("gamma endpoints wrong: %v", g.Data)
	}
	if absDiff(g.Data[1], 16) > 1 {
		t.Fatalf("gamma 2 of 64 = %d, want ~16", g.Data[1])
	}
}

func TestLogTransformEndpoints(t *testing.T) {
	img := mkGray(t, 1, 2, []uint8{0, 255})
	out := LogTransform(img)
	if out.Data[0] != 0 || out.Data[1] != 255 {
		t.Fatalf("log endpoints wrong: %v", out.Data)
	}
}

func TestApplyLUT(t *testing.T) {
	var lut [256]uint8
	for i := range lut {
		lut[i] = uint8(255 - i)
	}
	img := mkGray(t, 1, 2, []uint8{0, 255})
	out := ApplyLUT(img, lut)
	if out.Data[0] != 255 || out.Data[1] != 0 {
		t.Fatalf("LUT invert wrong: %v", out.Data)
	}
}

func TestHistogramStretch(t *testing.T) {
	img := mkGray(t, 1, 3, []uint8{100, 150, 200})
	out := HistogramStretch(img)
	if out.Data[0] != 0 || out.Data[2] != 255 {
		t.Fatalf("stretch endpoints wrong: %v", out.Data)
	}
	if absDiff(out.Data[1], 128) > 2 {
		t.Fatalf("stretch mid = %d, want ~128", out.Data[1])
	}
}

func TestHistogramEqualizeMonotonic(t *testing.T) {
	// Increasing gray input must map to non-decreasing output.
	data := make([]uint8, 16)
	for i := range data {
		data[i] = uint8(i * 16)
	}
	img := mkGray(t, 1, 16, data)
	out := HistogramEqualize(img)
	for i := 1; i < len(out.Data); i++ {
		if out.Data[i] < out.Data[i-1] {
			t.Fatalf("equalize not monotonic at %d", i)
		}
	}
	if out.Data[len(out.Data)-1] != 255 {
		t.Fatalf("equalize max not 255: %d", out.Data[len(out.Data)-1])
	}
}

func TestAdjustBrightness(t *testing.T) {
	img := mkGray(t, 1, 3, []uint8{0, 100, 250})
	out := AdjustBrightness(img, 20)
	if out.Data[0] != 20 || out.Data[1] != 120 || out.Data[2] != 255 {
		t.Fatalf("brightness wrong: %v", out.Data)
	}
}

func TestAdjustContrast(t *testing.T) {
	img := mkGray(t, 1, 3, []uint8{128, 138, 118})
	out := AdjustContrast(img, 2)
	// 128 stays, 138 -> 148, 118 -> 108.
	if out.Data[0] != 128 || out.Data[1] != 148 || out.Data[2] != 108 {
		t.Fatalf("contrast wrong: %v", out.Data)
	}
}

func TestAdjustSaturation(t *testing.T) {
	img := constRGB(1, 1, 200, 100, 50)
	gray := AdjustSaturation(img, 0)
	// factor 0 -> all channels equal to luma.
	if gray.Data[0] != gray.Data[1] || gray.Data[1] != gray.Data[2] {
		t.Fatalf("saturation 0 not gray: %v", gray.Data)
	}
	// factor 1 -> identity.
	id := AdjustSaturation(img, 1)
	for i := range img.Data {
		if absDiff(id.Data[i], img.Data[i]) > 1 {
			t.Fatalf("saturation 1 not identity")
		}
	}
}

func TestExposureCompensateIdentity(t *testing.T) {
	img := mkGray(t, 1, 3, []uint8{10, 128, 240})
	out := ExposureCompensate(img, 0)
	for i := range img.Data {
		if absDiff(out.Data[i], img.Data[i]) > 1 {
			t.Fatalf("exposure 0 stops changed pixel: %d vs %d", out.Data[i], img.Data[i])
		}
	}
	// +1 stop brightens.
	up := ExposureCompensate(img, 1)
	if up.Data[1] <= img.Data[1] {
		t.Fatalf("+1 stop did not brighten mid: %d", up.Data[1])
	}
}

func TestBlend(t *testing.T) {
	a := constRGB(1, 1, 100, 100, 100)
	b := constRGB(1, 1, 200, 200, 200)
	if Blend(a, b, 1).Data[0] != 100 {
		t.Fatalf("alpha 1 wrong")
	}
	if Blend(a, b, 0).Data[0] != 200 {
		t.Fatalf("alpha 0 wrong")
	}
	if absDiff(Blend(a, b, 0.5).Data[0], 150) > 1 {
		t.Fatalf("alpha 0.5 wrong")
	}
}

func TestSharpenIdentity(t *testing.T) {
	img := constRGB(4, 4, 90, 90, 90)
	out := Sharpen(img, 1.0)
	for i := range img.Data {
		if absDiff(out.Data[i], img.Data[i]) > 0 {
			t.Fatalf("sharpen of constant changed pixel")
		}
	}
}

func TestUnsharpMaskConstant(t *testing.T) {
	img := constRGB(5, 5, 120, 130, 140)
	out := UnsharpMask(img, 3, 1.0, 1.5)
	for i := range img.Data {
		if absDiff(out.Data[i], img.Data[i]) > 1 {
			t.Fatalf("unsharp of constant changed pixel")
		}
	}
}

func TestVignette(t *testing.T) {
	img := constRGB(9, 9, 200, 200, 200)
	out := Vignette(img, 0.8)
	// Centre pixel (4,4) unchanged.
	ci := (4*9 + 4) * 3
	if absDiff(out.Data[ci], 200) > 1 {
		t.Fatalf("vignette darkened centre: %d", out.Data[ci])
	}
	// Corner darkened.
	if out.Data[0] >= 200 {
		t.Fatalf("vignette did not darken corner: %d", out.Data[0])
	}
}

func TestLocalContrastConstant(t *testing.T) {
	img := constRGB(6, 6, 111, 111, 111)
	out := LocalContrast(img, 1.5, 1.0)
	for i := range img.Data {
		if absDiff(out.Data[i], img.Data[i]) > 1 {
			t.Fatalf("local contrast of constant changed pixel")
		}
	}
}

func TestCLAHEConstant(t *testing.T) {
	img := mkGray(t, 16, 16, func() []uint8 {
		d := make([]uint8, 256)
		for i := range d {
			d[i] = 100
		}
		return d
	}())
	out := CLAHE(img, 2.0, 4)
	if out.Rows != 16 || out.Cols != 16 || out.Channels != 1 {
		t.Fatalf("CLAHE shape wrong")
	}
	// A flat image should stay flat.
	for i := range out.Data {
		if absDiff(out.Data[i], out.Data[0]) > 1 {
			t.Fatalf("CLAHE of flat image not flat")
		}
	}
}

func TestCLAHEColorShape(t *testing.T) {
	img := cv.NewMat(20, 24, 3)
	for i := range img.Data {
		img.Data[i] = uint8((i * 7) % 256)
	}
	out := CLAHE(img, 3.0, 4)
	if out.Rows != 20 || out.Cols != 24 || out.Channels != 3 {
		t.Fatalf("CLAHE color shape wrong")
	}
}
