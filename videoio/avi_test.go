package videoio_test

import (
	"math"
	"path/filepath"
	"testing"

	cv "github.com/malcolmston/opencv"
	"github.com/malcolmston/opencv/videoio"
)

// jpegTol bounds the per-channel error introduced by the JPEG codec used in the
// MJPEG AVI path. Quality is 95, but chroma subsampling on synthetic
// high-frequency content still shifts samples noticeably.
const jpegTol = 40

// smoothFrame builds a low-frequency gradient frame that JPEG reproduces
// tightly, so the round-trip assertion is meaningful rather than slack.
func smoothFrame(h, w, seed int) *cv.Mat {
	m := cv.NewMat(h, w, 3)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			m.Set(y, x, 0, uint8((x*8+seed*20)&0xff))
			m.Set(y, x, 1, uint8((y*8+40)&0xff))
			m.Set(y, x, 2, uint8((seed*30+64)&0xff))
		}
	}
	return m
}

func TestMJPEGAVIRoundTrip(t *testing.T) {
	const h, w = 16, 24
	orig := []*cv.Mat{smoothFrame(h, w, 0), smoothFrame(h, w, 1), smoothFrame(h, w, 2)}
	path := filepath.Join(t.TempDir(), "clip.avi")
	if err := videoio.WriteMJPEGAVI(path, orig, 12); err != nil {
		t.Fatalf("WriteMJPEGAVI: %v", err)
	}

	frames, fps, err := videoio.ReadMJPEGAVI(path)
	if err != nil {
		t.Fatalf("ReadMJPEGAVI: %v", err)
	}
	if len(frames) != len(orig) {
		t.Fatalf("frame count = %d, want %d", len(frames), len(orig))
	}
	if math.Abs(fps-12) > 0.5 {
		t.Errorf("fps = %v, want ~12", fps)
	}
	for i, f := range frames {
		if f.Rows != h || f.Cols != w || f.Channels != 3 {
			t.Fatalf("frame %d dims = %dx%dx%d, want %dx%dx3", i, f.Rows, f.Cols, f.Channels, h, w)
		}
		for y := 0; y < h; y++ {
			for x := 0; x < w; x++ {
				for c := 0; c < 3; c++ {
					if diff := absDiff(f.At(y, x, c), orig[i].At(y, x, c)); diff > jpegTol {
						t.Fatalf("frame %d pixel (%d,%d) ch %d diff %d > tol %d", i, y, x, c, diff, jpegTol)
					}
				}
			}
		}
	}
}

func TestOpenAVICapture(t *testing.T) {
	orig := []*cv.Mat{smoothFrame(8, 8, 0), smoothFrame(8, 8, 1)}
	path := filepath.Join(t.TempDir(), "cap.avi")
	if err := videoio.WriteMJPEGAVI(path, orig, 25); err != nil {
		t.Fatalf("WriteMJPEGAVI: %v", err)
	}
	cap, err := videoio.OpenAVI(path)
	if err != nil {
		t.Fatalf("OpenAVI: %v", err)
	}
	defer cap.Close()
	if got := cap.Get(videoio.CAP_PROP_FRAME_COUNT); got != 2 {
		t.Errorf("FRAME_COUNT = %v, want 2", got)
	}
	if got := cap.Get(videoio.CAP_PROP_FPS); math.Abs(got-25) > 1 {
		t.Errorf("FPS = %v, want ~25", got)
	}
	n := 0
	for {
		if _, ok := cap.Read(); !ok {
			break
		}
		n++
	}
	if n != 2 {
		t.Errorf("iterated %d frames, want 2", n)
	}
}

func TestAVIErrors(t *testing.T) {
	if _, err := videoio.NewAVIWriter("", 1); err == nil {
		t.Errorf("NewAVIWriter empty path: want error")
	}
	if err := videoio.WriteMJPEGAVI(filepath.Join(t.TempDir(), "x.avi"), nil, 1); err == nil {
		t.Errorf("WriteMJPEGAVI no frames: want error")
	}
	if _, err := videoio.OpenAVI(filepath.Join(t.TempDir(), "missing.avi")); err == nil {
		t.Errorf("OpenAVI missing: want error")
	}
	w, err := videoio.NewAVIWriter(filepath.Join(t.TempDir(), "e.avi"), 10)
	if err != nil {
		t.Fatalf("NewAVIWriter: %v", err)
	}
	if err := w.Release(); err == nil {
		t.Errorf("Release with no frames: want error")
	}
}
