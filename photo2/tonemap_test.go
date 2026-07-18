package photo2

import (
	"math"
	"testing"

	cv "github.com/malcolmston/opencv"
)

// mkHDR builds a linear HDR image (3 float planes) from per-pixel RGB triples.
func mkHDR(rows, cols int, rgb [][3]float64) []*cv.FloatMat {
	R := cv.NewFloatMat(rows, cols)
	G := cv.NewFloatMat(rows, cols)
	B := cv.NewFloatMat(rows, cols)
	for i, t := range rgb {
		R.Data[i] = t[0]
		G.Data[i] = t[1]
		B.Data[i] = t[2]
	}
	return []*cv.FloatMat{R, G, B}
}

func inRange(m *cv.Mat) bool {
	// uint8 is always in range; just assert non-empty and finite construction.
	return m != nil && !m.Empty()
}

func TestGammaToneMap(t *testing.T) {
	hdr := mkHDR(1, 2, [][3]float64{{0, 0, 0}, {1, 1, 1}})
	out := GammaToneMap(hdr, 2.2)
	if out.Data[0] != 0 {
		t.Fatalf("black not 0: %d", out.Data[0])
	}
	if out.Data[3] != 255 {
		t.Fatalf("white not 255: %d", out.Data[3])
	}
}

func TestReinhardToneMapRange(t *testing.T) {
	// A high-dynamic-range gradient.
	rgb := make([][3]float64, 16)
	for i := range rgb {
		v := float64(i) * 4 // up to 60
		rgb[i] = [3]float64{v, v * 0.8, v * 0.5}
	}
	out := ReinhardToneMap(mkHDR(1, 16, rgb), DefaultReinhardParams())
	if !inRange(out) {
		t.Fatalf("reinhard empty")
	}
	// Monotonic in luminance: brighter input -> brighter or equal output gray.
	prev := -1
	for i := 0; i < 16; i++ {
		g := int(out.Data[i*3+0])
		if g < prev-1 {
			t.Fatalf("reinhard not monotonic at %d: %d after %d", i, g, prev)
		}
		prev = g
	}
}

func TestDragoToneMapRange(t *testing.T) {
	rgb := make([][3]float64, 9)
	for i := range rgb {
		v := float64(i+1) * 2
		rgb[i] = [3]float64{v, v, v}
	}
	out := DragoToneMap(mkHDR(3, 3, rgb), DefaultDragoParams())
	if !inRange(out) {
		t.Fatalf("drago empty")
	}
	prev := -1
	for i := 0; i < 9; i++ {
		g := int(out.Data[i*3])
		if g < prev-1 {
			t.Fatalf("drago not monotonic at %d", i)
		}
		prev = g
	}
}

func TestDurandToneMapRange(t *testing.T) {
	rgb := make([][3]float64, 16)
	for i := range rgb {
		v := float64(i+1) * 3
		rgb[i] = [3]float64{v, v * 0.9, v * 0.7}
	}
	out := DurandToneMap(mkHDR(4, 4, rgb), DefaultDurandParams())
	if !inRange(out) {
		t.Fatalf("durand empty")
	}
	if out.Rows != 4 || out.Cols != 4 || out.Channels != 3 {
		t.Fatalf("durand shape wrong")
	}
}

func TestLogToneMap(t *testing.T) {
	hdr := mkHDR(1, 3, [][3]float64{{0, 0, 0}, {1, 1, 1}, {5, 5, 5}})
	out := LogToneMap(hdr, 10)
	// Brightest input maps to the largest output.
	if out.Data[6] < out.Data[3] || out.Data[3] < out.Data[0] {
		t.Fatalf("log tonemap not monotonic: %v", out.Data)
	}
}

func TestMertensFusionIdentical(t *testing.T) {
	// Fusing identical images reproduces the input (pyramids are exact and the
	// normalised weights sum to one at every scale).
	base := cv.NewMat(8, 8, 3)
	for i := range base.Data {
		base.Data[i] = uint8((i*5 + 30) % 256)
	}
	out := MertensFusion([]*cv.Mat{base.Clone(), base.Clone()}, DefaultMertensParams())
	if out.Rows != 8 || out.Cols != 8 {
		t.Fatalf("mertens shape wrong")
	}
	maxErr := 0
	for i := range base.Data {
		if d := absDiff(base.Data[i], out.Data[i]); d > maxErr {
			maxErr = d
		}
	}
	if maxErr > 2 {
		t.Fatalf("mertens of identical images differs by %d", maxErr)
	}
}

func TestMertensFusionPicksExposed(t *testing.T) {
	// A textured, coloured scene photographed at two exposures. The "good"
	// exposure sits near mid-grey; the "dark" one is underexposed. Because the
	// scene has contrast and colour, the well-exposedness measure can act, and
	// fusion should track the well-exposed frame's brightness.
	rows, cols := 12, 12
	good := cv.NewMat(rows, cols, 3)
	dark := cv.NewMat(rows, cols, 3)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			i := y*cols + x
			p := (x + y) & 1 // strong checkerboard texture and colour
			r := uint8(60 + 80*p)
			g := uint8(140 - 80*p)
			b := uint8(100)
			good.Data[i*3+0], good.Data[i*3+1], good.Data[i*3+2] = r, g, b
			// The dark exposure keeps the same structure at 1/5 the level, so it
			// has strictly less contrast, saturation and well-exposedness.
			dark.Data[i*3+0] = r / 5
			dark.Data[i*3+1] = g / 5
			dark.Data[i*3+2] = b / 5
		}
	}
	out := MertensFusion([]*cv.Mat{dark, good}, DefaultMertensParams())
	meanOf := func(m *cv.Mat) float64 {
		var s float64
		for _, v := range m.Data {
			s += float64(v)
		}
		return s / float64(len(m.Data))
	}
	mg, md, mo := meanOf(good), meanOf(dark), meanOf(out)
	if math.Abs(mo-mg) >= math.Abs(mo-md) {
		t.Fatalf("fusion nearer dark (%.1f) than good (%.1f): out=%.1f", md, mg, mo)
	}
}
