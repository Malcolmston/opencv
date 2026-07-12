package videoio_test

import (
	"path/filepath"
	"testing"

	cv "github.com/malcolmston/opencv"
	"github.com/malcolmston/opencv/videoio"
)

func TestFourCCRoundTrip(t *testing.T) {
	code := videoio.FourCC('M', 'J', 'P', 'G')
	if got := videoio.FourCCString(code); got != "MJPG" {
		t.Fatalf("FourCCString = %q, want MJPG", got)
	}
}

func TestCaptureProperties(t *testing.T) {
	const h, w = 6, 8
	orig := []*cv.Mat{makeFrame(h, w, 0), makeFrame(h, w, 1), makeFrame(h, w, 2), makeFrame(h, w, 3)}
	path := filepath.Join(t.TempDir(), "props.gif")
	if err := videoio.WriteGIF(path, orig, 10); err != nil { // delay 10cs -> 10 fps
		t.Fatalf("WriteGIF: %v", err)
	}
	cap, err := videoio.OpenGIF(path)
	if err != nil {
		t.Fatalf("OpenGIF: %v", err)
	}
	defer cap.Close()

	if got := cap.Get(videoio.CAP_PROP_FRAME_COUNT); got != 4 {
		t.Errorf("FRAME_COUNT = %v, want 4", got)
	}
	if got := cap.Get(videoio.CAP_PROP_FRAME_WIDTH); got != float64(w) {
		t.Errorf("FRAME_WIDTH = %v, want %d", got, w)
	}
	if got := cap.Get(videoio.CAP_PROP_FRAME_HEIGHT); got != float64(h) {
		t.Errorf("FRAME_HEIGHT = %v, want %d", got, h)
	}
	if got := cap.Get(videoio.CAP_PROP_FPS); got != 10 {
		t.Errorf("FPS = %v, want 10", got)
	}
	if got := cap.Get(videoio.CAP_PROP_POS_FRAMES); got != 0 {
		t.Errorf("initial POS_FRAMES = %v, want 0", got)
	}
}

func TestGrabRetrieveAndSeek(t *testing.T) {
	orig := []*cv.Mat{makeFrame(4, 4, 0), makeFrame(4, 4, 1), makeFrame(4, 4, 2)}
	path := filepath.Join(t.TempDir(), "seek.gif")
	if err := videoio.WriteGIF(path, orig, 5); err != nil {
		t.Fatalf("WriteGIF: %v", err)
	}
	cap, err := videoio.OpenGIF(path)
	if err != nil {
		t.Fatalf("OpenGIF: %v", err)
	}
	defer cap.Close()

	// Grab then Retrieve yields the first frame; Retrieve again is idempotent.
	if !cap.Grab() {
		t.Fatal("Grab returned false on first frame")
	}
	f0a, ok := cap.Retrieve()
	if !ok || f0a == nil {
		t.Fatal("Retrieve after Grab failed")
	}
	f0b, _ := cap.Retrieve()
	if f0a != f0b {
		t.Fatal("Retrieve is not idempotent")
	}
	if cap.PosFrames() != 1 {
		t.Errorf("PosFrames after one Grab = %d, want 1", cap.PosFrames())
	}

	// Seek back to 0 and Read the first frame again.
	if got := cap.SetPosFrames(0); got != 0 {
		t.Fatalf("SetPosFrames(0) = %d, want 0", got)
	}
	if got := cap.Get(videoio.CAP_PROP_POS_FRAMES); got != 0 {
		t.Errorf("POS_FRAMES after seek = %v, want 0", got)
	}
	rf, ok := cap.Read()
	if !ok || rf == nil {
		t.Fatal("Read after seek failed")
	}

	// Seek past the end clamps to FrameCount.
	if got := cap.SetPosFrames(999); got != 3 {
		t.Errorf("SetPosFrames(999) = %d, want 3 (clamped)", got)
	}
	if _, ok := cap.Read(); ok {
		t.Error("Read at end returned ok=true")
	}

	// Set via property API: fractional seek and FPS rewrite.
	if !cap.Set(videoio.CAP_PROP_POS_AVI_RATIO, 0) {
		t.Error("Set POS_AVI_RATIO returned false")
	}
	if cap.PosFrames() != 0 {
		t.Errorf("PosFrames after ratio seek = %d, want 0", cap.PosFrames())
	}
	if !cap.Set(videoio.CAP_PROP_FPS, 20) { // 20 fps -> delay 5cs
		t.Error("Set FPS returned false")
	}
	if got := cap.Get(videoio.CAP_PROP_FPS); got != 20 {
		t.Errorf("FPS after Set = %v, want 20", got)
	}
	if cap.Set(videoio.CAP_PROP_FRAME_WIDTH, 100) {
		t.Error("Set FRAME_WIDTH returned true, want false (read-only)")
	}
}

func TestFrameGrabberInterface(t *testing.T) {
	orig := []*cv.Mat{makeFrame(3, 3, 0), makeFrame(3, 3, 1)}
	path := filepath.Join(t.TempDir(), "grab.gif")
	if err := videoio.WriteGIF(path, orig, 5); err != nil {
		t.Fatalf("WriteGIF: %v", err)
	}
	cap, err := videoio.OpenGIF(path)
	if err != nil {
		t.Fatalf("OpenGIF: %v", err)
	}
	var g videoio.FrameGrabber = cap // must satisfy the interface
	defer g.Close()
	n := 0
	for g.Grab() {
		if _, ok := g.Retrieve(); !ok {
			t.Fatal("Retrieve failed after Grab")
		}
		n++
	}
	if n != 2 {
		t.Errorf("grabbed %d frames, want 2", n)
	}
}

func TestWriterProperties(t *testing.T) {
	w, err := videoio.NewGIFWriter(filepath.Join(t.TempDir(), "wp.gif"), 10)
	if err != nil {
		t.Fatalf("NewGIFWriter: %v", err)
	}
	if got := w.Get(videoio.CAP_PROP_FPS); got != 10 {
		t.Errorf("writer FPS = %v, want 10", got)
	}
	// FPS is settable before any frame is written.
	if !w.Set(videoio.CAP_PROP_FPS, 25) {
		t.Error("Set FPS before frames returned false")
	}
	if got := w.Get(videoio.CAP_PROP_FPS); got != 25 {
		t.Errorf("writer FPS after Set = %v, want 25", got)
	}
	if err := w.Write(makeFrame(4, 4, 0)); err != nil {
		t.Fatalf("Write: %v", err)
	}
	if got := w.Get(videoio.CAP_PROP_FRAME_COUNT); got != 1 {
		t.Errorf("FRAME_COUNT = %v, want 1", got)
	}
	if got := w.Get(videoio.CAP_PROP_FRAME_WIDTH); got != 4 {
		t.Errorf("FRAME_WIDTH = %v, want 4", got)
	}
	// FPS is no longer settable once frames exist.
	if w.Set(videoio.CAP_PROP_FPS, 30) {
		t.Error("Set FPS after first frame returned true, want false")
	}
	_ = w.Release()
}
