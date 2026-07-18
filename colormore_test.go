package cv

import "testing"

func TestRGBToYUVKnown(t *testing.T) {
	src := NewMat(1, 2, 3)
	copy(src.Data, []uint8{255, 255, 255, 0, 0, 0})
	yuv := RGBToYUV(src)
	if yuv.At(0, 0, 0) != 255 || yuv.At(0, 0, 1) != 128 || yuv.At(0, 0, 2) != 128 {
		t.Errorf("white YUV = %v", yuv.AtPixel(0, 0))
	}
	if yuv.At(0, 1, 0) != 0 || yuv.At(0, 1, 1) != 128 {
		t.Errorf("black YUV = %v", yuv.AtPixel(0, 1))
	}
}

func TestYUVRoundTrip(t *testing.T) {
	src := NewMat(1, 1, 3)
	copy(src.Data, []uint8{120, 200, 60})
	back := YUVToRGB(RGBToYUV(src))
	for c := 0; c < 3; c++ {
		d := int(back.At(0, 0, c)) - int(src.At(0, 0, c))
		if d < -2 || d > 2 {
			t.Errorf("channel %d round trip off by %d", c, d)
		}
	}
}

func TestRGBToXYZRed(t *testing.T) {
	src := NewMat(1, 1, 3)
	copy(src.Data, []uint8{255, 0, 0})
	xyz := RGBToXYZ(src)
	if xyz.At(0, 0, 0) != 105 || xyz.At(0, 0, 1) != 54 {
		t.Errorf("red XYZ = %v", xyz.AtPixel(0, 0))
	}
}

func TestCMYKRoundTrip(t *testing.T) {
	src := NewMat(1, 1, 3)
	copy(src.Data, []uint8{255, 0, 0})
	cmyk := RGBToCMYK(src)
	if cmyk.Channels != 4 {
		t.Fatalf("channels = %d", cmyk.Channels)
	}
	if cmyk.At(0, 0, 0) != 0 || cmyk.At(0, 0, 1) != 255 || cmyk.At(0, 0, 3) != 0 {
		t.Errorf("red CMYK = %v", cmyk.AtPixel(0, 0))
	}
	back := CMYKToRGB(cmyk)
	if back.At(0, 0, 0) != 255 || back.At(0, 0, 1) != 0 || back.At(0, 0, 2) != 0 {
		t.Errorf("CMYK round trip = %v", back.AtPixel(0, 0))
	}
}

func TestRGBToGray(t *testing.T) {
	src := NewMat(1, 1, 3)
	copy(src.Data, []uint8{255, 255, 255})
	if RGBToGray601(src).At(0, 0, 0) != 255 {
		t.Error("white 601 gray should be 255")
	}
	if RGBToGray709(src).At(0, 0, 0) != 255 {
		t.Error("white 709 gray should be 255")
	}
}
