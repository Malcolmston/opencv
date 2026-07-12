package plot

import (
	"testing"

	cv "github.com/malcolmston/opencv"
)

// rampMat returns a 1x256 single-channel Mat whose column i holds intensity i.
func rampMat() *cv.Mat {
	m := cv.NewMat(1, 256, 1)
	for i := 0; i < 256; i++ {
		m.Data[i] = uint8(i)
	}
	return m
}

func px3(m *cv.Mat, y, x int) [3]uint8 {
	i := (y*m.Cols + x) * m.Channels
	return [3]uint8{m.Data[i], m.Data[i+1], m.Data[i+2]}
}

func TestApplyColorMapEndpoints(t *testing.T) {
	cases := []struct {
		name   string
		cm     Colormap
		lo, hi [3]uint8
	}{
		{"jet", ColormapJet, [3]uint8{0, 0, 128}, [3]uint8{128, 0, 0}},
		{"hot", ColormapHot, [3]uint8{0, 0, 0}, [3]uint8{255, 255, 255}},
		{"cool", ColormapCool, [3]uint8{0, 255, 255}, [3]uint8{255, 0, 255}},
		{"bone", ColormapBone, [3]uint8{0, 0, 0}, [3]uint8{255, 255, 255}},
		{"hsv", ColormapHSV, [3]uint8{255, 0, 0}, [3]uint8{255, 0, 0}},
		{"viridis", ColormapViridis, [3]uint8{68, 1, 84}, [3]uint8{253, 231, 37}},
		{"plasma", ColormapPlasma, [3]uint8{13, 8, 135}, [3]uint8{240, 249, 33}},
		{"grayscale", ColormapGrayscale, [3]uint8{0, 0, 0}, [3]uint8{255, 255, 255}},
	}
	ramp := rampMat()
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			out := ApplyColorMap(ramp, tc.cm)
			if out.Channels != 3 {
				t.Fatalf("channels = %d, want 3", out.Channels)
			}
			if out.Rows != 1 || out.Cols != 256 {
				t.Fatalf("size = %dx%d, want 1x256", out.Rows, out.Cols)
			}
			if got := px3(out, 0, 0); got != tc.lo {
				t.Errorf("intensity 0 = %v, want %v", got, tc.lo)
			}
			if got := px3(out, 0, 255); got != tc.hi {
				t.Errorf("intensity 255 = %v, want %v", got, tc.hi)
			}
		})
	}
}

func TestColormapTableMonotoneGrayscale(t *testing.T) {
	table := ColormapTable(ColormapGrayscale)
	if len(table) != 256 {
		t.Fatalf("table length = %d, want 256", len(table))
	}
	for i := 0; i < 256; i++ {
		want := [3]uint8{uint8(i), uint8(i), uint8(i)}
		if table[i] != want {
			t.Fatalf("grayscale[%d] = %v, want %v", i, table[i], want)
		}
	}
}

func TestApplyColorMapMapsEveryPixel(t *testing.T) {
	// A 2x2 image of distinct intensities recolours pixel-for-pixel.
	src := cv.NewMat(2, 2, 1)
	src.Data = []uint8{0, 64, 128, 255}
	table := ColormapTable(ColormapViridis)
	out := ApplyColorMap(src, ColormapViridis)
	for p := 0; p < 4; p++ {
		want := table[src.Data[p]]
		got := [3]uint8{out.Data[p*3], out.Data[p*3+1], out.Data[p*3+2]}
		if got != want {
			t.Errorf("pixel %d = %v, want %v", p, got, want)
		}
	}
}

func TestApplyCustomColorMap(t *testing.T) {
	// Custom table: red channel = intensity, others zero.
	table := make([][3]uint8, 256)
	for i := range table {
		table[i] = [3]uint8{uint8(i), 0, 0}
	}
	src := rampMat()
	out := ApplyCustomColorMap(src, table)
	for i := 0; i < 256; i++ {
		got := px3(out, 0, i)
		if got != [3]uint8{uint8(i), 0, 0} {
			t.Fatalf("col %d = %v", i, got)
		}
	}
}

func TestLUTExact(t *testing.T) {
	// Inverting table applied to a 3-channel image maps every sample exactly.
	table := make([]uint8, 256)
	for i := range table {
		table[i] = uint8(255 - i)
	}
	src := cv.NewMat(2, 3, 3)
	for i := range src.Data {
		src.Data[i] = uint8((i * 7) % 256)
	}
	out := LUT(src, table)
	if out.Rows != src.Rows || out.Cols != src.Cols || out.Channels != src.Channels {
		t.Fatalf("shape changed: %dx%dx%d", out.Rows, out.Cols, out.Channels)
	}
	for i := range src.Data {
		if want := uint8(255 - src.Data[i]); out.Data[i] != want {
			t.Fatalf("sample %d = %d, want %d", i, out.Data[i], want)
		}
	}
}

func TestApplyColorMapPanicsOnMultichannel(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("expected panic on 3-channel input")
		}
	}()
	ApplyColorMap(cv.NewMat(2, 2, 3), ColormapJet)
}

func TestLUTPanicsOnShortTable(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("expected panic on short table")
		}
	}()
	LUT(cv.NewMat(1, 1, 1), make([]uint8, 10))
}
