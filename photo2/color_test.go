package photo2

import (
	"math"
	"testing"
)

func TestRGBToLabWhiteBlack(t *testing.T) {
	white := RGBToLab(constRGB(1, 1, 255, 255, 255))
	if math.Abs(white[0].Data[0]-100) > 0.5 {
		t.Fatalf("white L = %v, want 100", white[0].Data[0])
	}
	if math.Abs(white[1].Data[0]) > 0.5 || math.Abs(white[2].Data[0]) > 0.5 {
		t.Fatalf("white a,b not ~0: %v %v", white[1].Data[0], white[2].Data[0])
	}
	black := RGBToLab(constRGB(1, 1, 0, 0, 0))
	if math.Abs(black[0].Data[0]) > 0.5 {
		t.Fatalf("black L = %v, want 0", black[0].Data[0])
	}
}

func TestLabRoundTrip(t *testing.T) {
	img := mkRGB(t, 1, 4, []uint8{
		30, 60, 90,
		200, 150, 100,
		10, 220, 40,
		128, 128, 128,
	})
	back := LabToRGB(RGBToLab(img))
	for i := range img.Data {
		if absDiff(img.Data[i], back.Data[i]) > 2 {
			t.Fatalf("Lab roundtrip at %d: %d vs %d", i, img.Data[i], back.Data[i])
		}
	}
}

func TestXYZRoundTrip(t *testing.T) {
	img := mkRGB(t, 1, 3, []uint8{40, 80, 160, 255, 0, 0, 12, 200, 240})
	back := XYZToRGB(RGBToXYZ(img))
	for i := range img.Data {
		if absDiff(img.Data[i], back.Data[i]) > 2 {
			t.Fatalf("XYZ roundtrip at %d: %d vs %d", i, img.Data[i], back.Data[i])
		}
	}
}

func TestColorTransferIdentity(t *testing.T) {
	// Transferring an image's own statistics onto itself is (near) identity.
	img := mkRGB(t, 2, 2, []uint8{
		10, 20, 30, 200, 100, 50,
		60, 180, 240, 90, 90, 90,
	})
	out := ColorTransferReinhard(img, img)
	for i := range img.Data {
		if absDiff(img.Data[i], out.Data[i]) > 2 {
			t.Fatalf("self color transfer at %d: %d vs %d", i, img.Data[i], out.Data[i])
		}
	}
}

func TestColorTransferMatchesTargetStats(t *testing.T) {
	src := constRGB(4, 4, 50, 50, 50)
	// Slightly vary src so its std is non-zero.
	src.Data[0] = 60
	target := constRGB(4, 4, 180, 120, 90)
	target.Data[0] = 190
	out := ColorTransferReinhard(src, target)
	// Output mean in lab space should approach target's mean.
	sm, _ := photo2LABStats(out)
	tm, _ := photo2LABStats(target)
	for c := 0; c < 3; c++ {
		if math.Abs(sm[c]-tm[c]) > 0.15 {
			t.Fatalf("channel %d mean %v far from target %v", c, sm[c], tm[c])
		}
	}
}

func TestGrayWorldWhiteBalance(t *testing.T) {
	// Build an image with a red cast: mean R > mean G,B.
	img := mkRGB(t, 1, 2, []uint8{200, 100, 50, 160, 80, 40})
	out := GrayWorldWhiteBalance(img)
	// After balancing, channel means should be equal.
	var mean [3]float64
	total := out.Rows * out.Cols
	for i := 0; i < total; i++ {
		for c := 0; c < 3; c++ {
			mean[c] += float64(out.Data[i*3+c])
		}
	}
	for c := range mean {
		mean[c] /= float64(total)
	}
	if math.Abs(mean[0]-mean[1]) > 1.5 || math.Abs(mean[1]-mean[2]) > 1.5 {
		t.Fatalf("gray world means not equalised: %v", mean)
	}
}

func TestSimpleWhiteBalance(t *testing.T) {
	// A per-pixel gray ramp replicated into all three channels.
	vals := []uint8{60, 80, 100, 120, 140}
	img := mkRGB(t, 1, 5, func() []uint8 {
		d := make([]uint8, len(vals)*3)
		for i, v := range vals {
			d[i*3+0], d[i*3+1], d[i*3+2] = v, v, v
		}
		return d
	}())
	out := SimpleWhiteBalance(img, 0.0, 1.0)
	// With full range, min maps near 0 and max near 255 on each channel.
	if out.Data[0] > 5 {
		t.Fatalf("white balance low end = %d", out.Data[0])
	}
	if out.Data[len(out.Data)-1] < 250 {
		t.Fatalf("white balance high end = %d", out.Data[len(out.Data)-1])
	}
}
