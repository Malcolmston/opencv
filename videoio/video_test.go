package videoio_test

import (
	"math"
	"path/filepath"
	"testing"

	cv "github.com/malcolmston/opencv"
	"github.com/malcolmston/opencv/videoio"
)

func TestWriteReadVideoDispatch(t *testing.T) {
	orig := []*cv.Mat{smoothFrame(12, 12, 0), smoothFrame(12, 12, 1), smoothFrame(12, 12, 2)}
	dir := t.TempDir()
	cases := []struct {
		name    string
		wantFPS float64
	}{
		{"clip.gif", 10},
		{"clip.png", 10},
		{"clip.avi", 10},
	}
	for _, tc := range cases {
		path := filepath.Join(dir, tc.name)
		if err := videoio.WriteVideoFromMats(path, orig, 10); err != nil {
			t.Fatalf("%s WriteVideoFromMats: %v", tc.name, err)
		}
		frames, fps, err := videoio.ReadVideoToMats(path)
		if err != nil {
			t.Fatalf("%s ReadVideoToMats: %v", tc.name, err)
		}
		if len(frames) != 3 {
			t.Errorf("%s frame count = %d, want 3", tc.name, len(frames))
		}
		if math.Abs(fps-tc.wantFPS) > 0.6 {
			t.Errorf("%s fps = %v, want ~%v", tc.name, fps, tc.wantFPS)
		}
	}
}

func TestWriteVideoUnsupported(t *testing.T) {
	err := videoio.WriteVideoFromMats(filepath.Join(t.TempDir(), "x.mp4"),
		[]*cv.Mat{makeFrame(2, 2, 0)}, 10)
	if err == nil {
		t.Error("WriteVideoFromMats .mp4: want error")
	}
	if _, _, err := videoio.ReadVideoToMats(filepath.Join(t.TempDir(), "x.mkv")); err == nil {
		t.Error("ReadVideoToMats .mkv: want error")
	}
}

func TestResampleFramesUpsample(t *testing.T) {
	// Two frames, each shown for 50 cs (total 100 cs = 1 s). Resampling to 10
	// fps should produce ~10 frames spanning the same second.
	frames := []*cv.Mat{makeFrame(3, 3, 0), makeFrame(3, 3, 1)}
	delays := []int{50, 50}
	out, outDelays := videoio.ResampleFrames(frames, delays, 10)
	if len(out) != 10 {
		t.Fatalf("upsample produced %d frames, want 10", len(out))
	}
	for i, d := range outDelays {
		if d != 10 {
			t.Errorf("outDelay[%d] = %d, want 10", i, d)
		}
	}
	// First half should sample frame 0, second half frame 1.
	if out[0] != frames[0] {
		t.Error("first output frame is not source frame 0")
	}
	if out[len(out)-1] != frames[1] {
		t.Error("last output frame is not source frame 1")
	}
}

func TestResampleFramesDownsample(t *testing.T) {
	// Four frames at 20 cs each (total 80 cs, i.e. 5 fps). Resampling down to
	// 2.5 fps (40 cs period) should halve the frame count to two.
	frames := []*cv.Mat{makeFrame(2, 2, 0), makeFrame(2, 2, 1), makeFrame(2, 2, 2), makeFrame(2, 2, 3)}
	delays := []int{20, 20, 20, 20}
	out, _ := videoio.ResampleFrames(frames, delays, 2.5)
	if len(out) != 2 {
		t.Fatalf("downsample produced %d frames, want 2", len(out))
	}
}

func TestResampleCapture(t *testing.T) {
	path := filepath.Join(t.TempDir(), "rs.gif")
	frames := []*cv.Mat{makeFrame(3, 3, 0), makeFrame(3, 3, 1)}
	if err := videoio.WriteGIF(path, frames, 50); err != nil { // 50 cs/frame
		t.Fatalf("WriteGIF: %v", err)
	}
	cap, err := videoio.OpenGIF(path)
	if err != nil {
		t.Fatalf("OpenGIF: %v", err)
	}
	defer cap.Close()
	rs := videoio.ResampleCapture(cap, 10)
	if rs.FrameCount() != 10 {
		t.Errorf("resampled capture has %d frames, want 10", rs.FrameCount())
	}
	if got := rs.Get(videoio.CAP_PROP_FPS); got != 10 {
		t.Errorf("resampled FPS = %v, want 10", got)
	}
}

func TestResampleFramesPanics(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Error("ResampleFrames with mismatched lengths did not panic")
		}
	}()
	videoio.ResampleFrames([]*cv.Mat{makeFrame(2, 2, 0)}, []int{1, 2}, 10)
}
