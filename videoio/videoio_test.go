package videoio_test

import (
	"path/filepath"
	"testing"

	cv "github.com/malcolmston/opencv"
	"github.com/malcolmston/opencv/videoio"
)

// paletteTol bounds the per-channel round-trip error introduced by mapping to
// the 216-colour web-safe palette (levels 51 apart, so at most ~26 off).
const paletteTol = 26

// makeFrame builds a deterministic h×w RGB frame whose pixel colours depend on
// position and the frame index, so distinct frames look distinct.
func makeFrame(h, w, seed int) *cv.Mat {
	m := cv.NewMat(h, w, 3)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			m.Set(y, x, 0, uint8((x*17+seed*40)&0xff))
			m.Set(y, x, 1, uint8((y*23+seed*11)&0xff))
			m.Set(y, x, 2, uint8((x*x+y*y+seed*7)&0xff))
		}
	}
	return m
}

func absDiff(a, b uint8) int {
	if a > b {
		return int(a) - int(b)
	}
	return int(b) - int(a)
}

func TestWriteReadRoundTrip(t *testing.T) {
	const h, w = 6, 8
	orig := []*cv.Mat{
		makeFrame(h, w, 0),
		makeFrame(h, w, 1),
		makeFrame(h, w, 2),
	}
	path := filepath.Join(t.TempDir(), "clip.gif")

	if err := videoio.WriteGIF(path, orig, 7); err != nil {
		t.Fatalf("WriteGIF: %v", err)
	}

	frames, delays, err := videoio.ReadGIF(path)
	if err != nil {
		t.Fatalf("ReadGIF: %v", err)
	}
	if len(frames) != len(orig) {
		t.Fatalf("frame count = %d, want %d", len(frames), len(orig))
	}
	if len(delays) != len(orig) {
		t.Fatalf("delay count = %d, want %d", len(delays), len(orig))
	}
	for i, d := range delays {
		if d != 7 {
			t.Errorf("delay[%d] = %d, want 7", i, d)
		}
	}
	for i, f := range frames {
		if f.Rows != h || f.Cols != w {
			t.Errorf("frame %d dims = %dx%d, want %dx%d", i, f.Rows, f.Cols, h, w)
		}
		if f.Channels != 3 {
			t.Errorf("frame %d channels = %d, want 3", i, f.Channels)
		}
		for y := 0; y < h; y++ {
			for x := 0; x < w; x++ {
				for c := 0; c < 3; c++ {
					got := f.At(y, x, c)
					want := orig[i].At(y, x, c)
					if diff := absDiff(got, want); diff > paletteTol {
						t.Fatalf("frame %d pixel (%d,%d) ch %d = %d, want %d (diff %d > tol %d)",
							i, y, x, c, got, want, diff, paletteTol)
					}
				}
			}
		}
	}
}

func TestVideoCaptureReadIterates(t *testing.T) {
	const h, w = 4, 5
	orig := []*cv.Mat{makeFrame(h, w, 0), makeFrame(h, w, 1)}
	path := filepath.Join(t.TempDir(), "cap.gif")
	if err := videoio.WriteGIF(path, orig, 5); err != nil {
		t.Fatalf("WriteGIF: %v", err)
	}

	cap, err := videoio.OpenGIF(path)
	if err != nil {
		t.Fatalf("OpenGIF: %v", err)
	}
	defer cap.Close()

	if got := cap.FrameCount(); got != 2 {
		t.Fatalf("FrameCount = %d, want 2", got)
	}
	if got := len(cap.Frames()); got != 2 {
		t.Fatalf("len(Frames) = %d, want 2", got)
	}
	if got := cap.Delays(); len(got) != 2 || got[0] != 5 || got[1] != 5 {
		t.Fatalf("Delays = %v, want [5 5]", got)
	}

	n := 0
	for {
		frame, ok := cap.Read()
		if !ok {
			break
		}
		if frame == nil {
			t.Fatalf("Read returned nil frame with ok=true")
		}
		n++
		if n > 100 {
			t.Fatalf("Read never terminated")
		}
	}
	if n != 2 {
		t.Fatalf("iterated %d frames, want 2", n)
	}
	// Further reads keep returning false.
	if _, ok := cap.Read(); ok {
		t.Fatalf("Read after exhaustion returned ok=true")
	}
}

func TestVideoWriterStreaming(t *testing.T) {
	const h, w = 3, 3
	path := filepath.Join(t.TempDir(), "stream.gif")
	wtr, err := videoio.NewGIFWriter(path, 12)
	if err != nil {
		t.Fatalf("NewGIFWriter: %v", err)
	}
	for i := 0; i < 4; i++ {
		if err := wtr.Write(makeFrame(h, w, i)); err != nil {
			t.Fatalf("Write %d: %v", i, err)
		}
	}
	if err := wtr.Release(); err != nil {
		t.Fatalf("Release: %v", err)
	}
	// Second Release must fail; Write after Release must fail.
	if err := wtr.Release(); err == nil {
		t.Errorf("second Release returned nil, want error")
	}
	if err := wtr.Write(makeFrame(h, w, 0)); err == nil {
		t.Errorf("Write after Release returned nil, want error")
	}

	frames, _, err := videoio.ReadGIF(path)
	if err != nil {
		t.Fatalf("ReadGIF: %v", err)
	}
	if len(frames) != 4 {
		t.Fatalf("frame count = %d, want 4", len(frames))
	}
}

func TestDifferingFrameSizesUsesFirstBounds(t *testing.T) {
	path := filepath.Join(t.TempDir(), "sizes.gif")
	frames := []*cv.Mat{
		makeFrame(6, 8, 0),  // first frame fixes the canvas at 8x6
		makeFrame(4, 5, 1),  // smaller
		makeFrame(9, 12, 2), // larger, must be clipped
	}
	if err := videoio.WriteGIF(path, frames, 4); err != nil {
		t.Fatalf("WriteGIF: %v", err)
	}
	got, _, err := videoio.ReadGIF(path)
	if err != nil {
		t.Fatalf("ReadGIF: %v", err)
	}
	if len(got) != 3 {
		t.Fatalf("frame count = %d, want 3", len(got))
	}
	for i, f := range got {
		if f.Rows != 6 || f.Cols != 8 {
			t.Errorf("frame %d dims = %dx%d, want 8x6", i, f.Cols, f.Rows)
		}
	}
}

func TestErrorPaths(t *testing.T) {
	if _, err := videoio.NewGIFWriter("", 1); err == nil {
		t.Errorf("NewGIFWriter with empty path returned nil error")
	}
	if err := videoio.WriteGIF(filepath.Join(t.TempDir(), "x.gif"), nil, 1); err == nil {
		t.Errorf("WriteGIF with no frames returned nil error")
	}
	if _, err := videoio.OpenGIF(filepath.Join(t.TempDir(), "missing.gif")); err == nil {
		t.Errorf("OpenGIF on missing file returned nil error")
	}

	w, err := videoio.NewGIFWriter(filepath.Join(t.TempDir(), "e.gif"), 1)
	if err != nil {
		t.Fatalf("NewGIFWriter: %v", err)
	}
	if err := w.Write(&cv.Mat{}); err == nil {
		t.Errorf("Write of empty Mat returned nil error")
	}
	if err := w.Release(); err == nil {
		t.Errorf("Release with no frames returned nil error")
	}
}
