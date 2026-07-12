package videoio_test

import (
	"path/filepath"
	"testing"

	cv "github.com/malcolmston/opencv"
	"github.com/malcolmston/opencv/videoio"
)

func TestImageSequencePNGRoundTrip(t *testing.T) {
	const h, w = 5, 7
	orig := []*cv.Mat{makeFrame(h, w, 0), makeFrame(h, w, 1), makeFrame(h, w, 2)}
	dir := t.TempDir()

	paths, err := videoio.WriteImageSequence(dir, "frame%04d.png", orig, 0)
	if err != nil {
		t.Fatalf("WriteImageSequence: %v", err)
	}
	if len(paths) != 3 {
		t.Fatalf("wrote %d paths, want 3", len(paths))
	}

	frames, err := videoio.ReadImageSequence(dir, "frame%04d.png", 0)
	if err != nil {
		t.Fatalf("ReadImageSequence: %v", err)
	}
	if len(frames) != 3 {
		t.Fatalf("read %d frames, want 3", len(frames))
	}
	for i, f := range frames {
		for y := 0; y < h; y++ {
			for x := 0; x < w; x++ {
				for c := 0; c < 3; c++ {
					if got, want := f.At(y, x, c), orig[i].At(y, x, c); got != want {
						t.Fatalf("frame %d (%d,%d) ch %d = %d, want %d (PNG lossless)", i, y, x, c, got, want)
					}
				}
			}
		}
	}
}

func TestImageSequenceStreamingAndCapture(t *testing.T) {
	dir := t.TempDir()
	w, err := videoio.NewImageSequenceWriter(dir, "img%d.png", 1)
	if err != nil {
		t.Fatalf("NewImageSequenceWriter: %v", err)
	}
	for i := 0; i < 4; i++ {
		if _, err := w.Write(makeFrame(3, 3, i)); err != nil {
			t.Fatalf("Write %d: %v", i, err)
		}
	}
	if w.Count() != 4 {
		t.Fatalf("Count = %d, want 4", w.Count())
	}
	if len(w.Files()) != 4 {
		t.Fatalf("Files len = %d, want 4", len(w.Files()))
	}

	cap, err := videoio.OpenImageSequence(dir, "img%d.png", 1, 8)
	if err != nil {
		t.Fatalf("OpenImageSequence: %v", err)
	}
	defer cap.Close()
	if cap.FrameCount() != 4 {
		t.Fatalf("FrameCount = %d, want 4", cap.FrameCount())
	}
	if got := cap.Get(videoio.CAP_PROP_FPS); got < 12.4 || got > 12.6 {
		t.Errorf("FPS = %v, want ~12.5 (delay 8cs)", got)
	}
}

func TestImageSequenceStopsAtGap(t *testing.T) {
	dir := t.TempDir()
	// Write indices 0,1,3 — a gap at 2 must stop the reader after two frames.
	for _, idx := range []int{0, 1, 3} {
		p := filepath.Join(dir, "g"+itoa(idx)+".png")
		if err := cv.ImWrite(p, makeFrame(2, 2, idx)); err != nil {
			t.Fatalf("ImWrite: %v", err)
		}
	}
	frames, err := videoio.ReadImageSequence(dir, "g%d.png", 0)
	if err != nil {
		t.Fatalf("ReadImageSequence: %v", err)
	}
	if len(frames) != 2 {
		t.Fatalf("read %d frames, want 2 (should stop at gap)", len(frames))
	}
}

func TestSequencePatternValidation(t *testing.T) {
	dir := t.TempDir()
	bad := []string{"", "noverb.png", "two%d%d.png", "frame%04d.bmp"}
	for _, p := range bad {
		if _, err := videoio.NewImageSequenceWriter(dir, p, 0); err == nil {
			t.Errorf("pattern %q: want validation error", p)
		}
	}
	if _, err := videoio.NewImageSequenceWriter(dir, "ok%03d.jpg", 0); err != nil {
		t.Errorf("pattern ok%%03d.jpg rejected: %v", err)
	}
}

// itoa is a tiny local integer formatter to keep the gap test free of fmt.
func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	var b []byte
	for n > 0 {
		b = append([]byte{byte('0' + n%10)}, b...)
		n /= 10
	}
	return string(b)
}
