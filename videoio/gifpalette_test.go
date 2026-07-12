package videoio_test

import (
	"path/filepath"
	"testing"

	cv "github.com/malcolmston/opencv"
	"github.com/malcolmston/opencv/videoio"
)

func TestPalettedGIFWriterPerFrameDelay(t *testing.T) {
	orig := []*cv.Mat{makeFrame(4, 6, 0), makeFrame(4, 6, 1), makeFrame(4, 6, 2)}
	want := []int{3, 15, 25}
	path := filepath.Join(t.TempDir(), "pal.gif")

	w, err := videoio.NewPalettedGIFWriter(path, videoio.PalettePlan9, 0)
	if err != nil {
		t.Fatalf("NewPalettedGIFWriter: %v", err)
	}
	for i, f := range orig {
		if err := w.WriteFrame(f, want[i]); err != nil {
			t.Fatalf("WriteFrame %d: %v", i, err)
		}
	}
	if err := w.Release(); err != nil {
		t.Fatalf("Release: %v", err)
	}

	frames, delays, err := videoio.ReadGIF(path)
	if err != nil {
		t.Fatalf("ReadGIF: %v", err)
	}
	if len(frames) != 3 {
		t.Fatalf("frame count = %d, want 3", len(frames))
	}
	for i := range want {
		if delays[i] != want[i] {
			t.Errorf("delay[%d] = %d, want %d", i, delays[i], want[i])
		}
	}
}

func TestWriteGIFDelaysWebSafe(t *testing.T) {
	orig := []*cv.Mat{makeFrame(4, 4, 0), makeFrame(4, 4, 1)}
	path := filepath.Join(t.TempDir(), "d.gif")
	if err := videoio.WriteGIFDelays(path, orig, []int{5, 9}, nil, 0); err != nil {
		t.Fatalf("WriteGIFDelays: %v", err)
	}
	_, delays, err := videoio.ReadGIF(path)
	if err != nil {
		t.Fatalf("ReadGIF: %v", err)
	}
	if len(delays) != 2 || delays[0] != 5 || delays[1] != 9 {
		t.Fatalf("delays = %v, want [5 9]", delays)
	}
}

// TestAdaptivePalette checks the quantizer bounds its palette size and that a
// low-colour clip round-trips through it within the coarse-bin tolerance.
func TestAdaptivePalette(t *testing.T) {
	// Two solid-colour frames use only two colours, well within any palette.
	a := cv.NewMat(4, 4, 3)
	for y := 0; y < 4; y++ {
		for x := 0; x < 4; x++ {
			a.Set(y, x, 0, 200)
			a.Set(y, x, 1, 50)
			a.Set(y, x, 2, 100)
		}
	}
	b := cv.NewMat(4, 4, 3)
	b.SetTo(0)
	frames := []*cv.Mat{a, b}

	pal := videoio.AdaptivePalette(frames, 16)
	if len(pal) == 0 || len(pal) > 16 {
		t.Fatalf("palette size = %d, want 1..16", len(pal))
	}

	path := filepath.Join(t.TempDir(), "adaptive.gif")
	if err := videoio.WriteGIFDelays(path, frames, []int{10, 10}, pal, 0); err != nil {
		t.Fatalf("WriteGIFDelays: %v", err)
	}
	got, _, err := videoio.ReadGIF(path)
	if err != nil {
		t.Fatalf("ReadGIF: %v", err)
	}
	if len(got) != 2 {
		t.Fatalf("frame count = %d, want 2", len(got))
	}
	// The 4-bit bin centres are spaced 17 apart, so worst-case error is ~9.
	const tol = 12
	for c := 0; c < 3; c++ {
		if diff := absDiff(got[0].At(0, 0, c), a.At(0, 0, c)); diff > tol {
			t.Errorf("adaptive frame0 ch %d diff %d > tol %d", c, diff, tol)
		}
	}
}

func TestPalettedGIFWriterErrors(t *testing.T) {
	if _, err := videoio.NewPalettedGIFWriter("", nil, 0); err == nil {
		t.Errorf("empty path: want error")
	}
	if err := videoio.WriteGIFDelays(filepath.Join(t.TempDir(), "x.gif"),
		[]*cv.Mat{makeFrame(2, 2, 0)}, []int{1, 2}, nil, 0); err == nil {
		t.Errorf("mismatched delays: want error")
	}
}
