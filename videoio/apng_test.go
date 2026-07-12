package videoio_test

import (
	"path/filepath"
	"testing"

	cv "github.com/malcolmston/opencv"
	"github.com/malcolmston/opencv/videoio"
)

// TestAPNGRoundTrip checks that APNG is lossless: the frames read back must
// match the originals pixel-for-pixel, unlike the palette-quantized GIF path.
func TestAPNGRoundTrip(t *testing.T) {
	const h, w = 6, 8
	orig := []*cv.Mat{
		makeFrame(h, w, 0),
		makeFrame(h, w, 1),
		makeFrame(h, w, 2),
		makeFrame(h, w, 3),
	}
	path := filepath.Join(t.TempDir(), "clip.png")
	if err := videoio.WriteAPNG(path, orig, 9); err != nil {
		t.Fatalf("WriteAPNG: %v", err)
	}

	frames, delays, err := videoio.ReadAPNG(path)
	if err != nil {
		t.Fatalf("ReadAPNG: %v", err)
	}
	if len(frames) != len(orig) {
		t.Fatalf("frame count = %d, want %d", len(frames), len(orig))
	}
	for i, d := range delays {
		if d != 9 {
			t.Errorf("delay[%d] = %d, want 9", i, d)
		}
	}
	for i, f := range frames {
		if f.Rows != h || f.Cols != w || f.Channels != 3 {
			t.Fatalf("frame %d dims = %dx%dx%d, want %dx%dx3", i, f.Rows, f.Cols, f.Channels, h, w)
		}
		for y := 0; y < h; y++ {
			for x := 0; x < w; x++ {
				for c := 0; c < 3; c++ {
					if got, want := f.At(y, x, c), orig[i].At(y, x, c); got != want {
						t.Fatalf("frame %d pixel (%d,%d) ch %d = %d, want %d (APNG must be lossless)",
							i, y, x, c, got, want)
					}
				}
			}
		}
	}
}

// TestAPNGVariableDelays verifies per-frame delay control round-trips.
func TestAPNGVariableDelays(t *testing.T) {
	orig := []*cv.Mat{makeFrame(4, 4, 0), makeFrame(4, 4, 1), makeFrame(4, 4, 2)}
	want := []int{5, 20, 33}
	path := filepath.Join(t.TempDir(), "vd.png")
	if err := videoio.WriteAPNGDelays(path, orig, want); err != nil {
		t.Fatalf("WriteAPNGDelays: %v", err)
	}
	_, delays, err := videoio.ReadAPNG(path)
	if err != nil {
		t.Fatalf("ReadAPNG: %v", err)
	}
	if len(delays) != len(want) {
		t.Fatalf("delay count = %d, want %d", len(delays), len(want))
	}
	for i := range want {
		if delays[i] != want[i] {
			t.Errorf("delay[%d] = %d, want %d", i, delays[i], want[i])
		}
	}
}

// TestAPNGStreamingWriter exercises the streaming writer and its guards.
func TestAPNGStreamingWriter(t *testing.T) {
	path := filepath.Join(t.TempDir(), "stream.png")
	w, err := videoio.NewAPNGWriter(path, 10)
	if err != nil {
		t.Fatalf("NewAPNGWriter: %v", err)
	}
	w.SetLoopCount(3)
	for i := 0; i < 3; i++ {
		if err := w.Write(makeFrame(5, 5, i)); err != nil {
			t.Fatalf("Write %d: %v", i, err)
		}
	}
	if err := w.Release(); err != nil {
		t.Fatalf("Release: %v", err)
	}
	if err := w.Release(); err == nil {
		t.Errorf("second Release returned nil, want error")
	}
	if err := w.Write(makeFrame(5, 5, 0)); err == nil {
		t.Errorf("Write after Release returned nil, want error")
	}

	cap, err := videoio.OpenAPNG(path)
	if err != nil {
		t.Fatalf("OpenAPNG: %v", err)
	}
	defer cap.Close()
	if cap.FrameCount() != 3 {
		t.Fatalf("FrameCount = %d, want 3", cap.FrameCount())
	}
}

// TestReadPlainPNG confirms a non-animated PNG decodes as a single frame.
func TestReadPlainPNG(t *testing.T) {
	path := filepath.Join(t.TempDir(), "single.png")
	if err := cv.ImWrite(path, makeFrame(4, 4, 1)); err != nil {
		t.Fatalf("ImWrite: %v", err)
	}
	frames, delays, err := videoio.ReadAPNG(path)
	if err != nil {
		t.Fatalf("ReadAPNG plain: %v", err)
	}
	if len(frames) != 1 || len(delays) != 1 {
		t.Fatalf("got %d frames %d delays, want 1 and 1", len(frames), len(delays))
	}
}

func TestAPNGErrors(t *testing.T) {
	if _, err := videoio.NewAPNGWriter("", 1); err == nil {
		t.Errorf("NewAPNGWriter empty path: want error")
	}
	if err := videoio.WriteAPNG(filepath.Join(t.TempDir(), "x.png"), nil, 1); err == nil {
		t.Errorf("WriteAPNG no frames: want error")
	}
	if err := videoio.WriteAPNGDelays(filepath.Join(t.TempDir(), "y.png"),
		[]*cv.Mat{makeFrame(2, 2, 0)}, []int{1, 2}); err == nil {
		t.Errorf("WriteAPNGDelays mismatched delays: want error")
	}
	if _, err := videoio.OpenAPNG(filepath.Join(t.TempDir(), "missing.png")); err == nil {
		t.Errorf("OpenAPNG missing: want error")
	}
}
