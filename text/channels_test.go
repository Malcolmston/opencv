package text

import (
	"testing"

	cv "github.com/malcolmston/opencv"
)

func TestComputeNMChannelsRGBLGrad(t *testing.T) {
	img := cv.NewMat(6, 6, 3)
	// Left half red, right half green, to create a vertical edge.
	for y := 0; y < 6; y++ {
		for x := 0; x < 6; x++ {
			if x < 3 {
				img.SetPixel(y, x, []uint8{200, 0, 0})
			} else {
				img.SetPixel(y, x, []uint8{0, 200, 0})
			}
		}
	}

	ch := ComputeNMChannels(img, NMRGBLGrad)
	if len(ch) != 5 {
		t.Fatalf("got %d channels, want 5", len(ch))
	}
	for i, c := range ch {
		if c.Channels != 1 || c.Rows != 6 || c.Cols != 6 {
			t.Fatalf("channel %d is %dx%dx%d, want 6x6x1", i, c.Rows, c.Cols, c.Channels)
		}
	}
	// Channel 0 is the red plane: 200 on the left, 0 on the right.
	if ch[0].At(0, 0, 0) != 200 || ch[0].At(0, 5, 0) != 0 {
		t.Errorf("red channel wrong: left=%d right=%d", ch[0].At(0, 0, 0), ch[0].At(0, 5, 0))
	}
	// The gradient channel (index 4) must respond at the vertical colour edge.
	grad := ch[4]
	if grad.At(2, 2, 0) == 0 && grad.At(2, 3, 0) == 0 {
		t.Errorf("gradient channel is flat at the edge, expected a response")
	}
}

func TestComputeNMChannelsIHSGrad(t *testing.T) {
	img := cv.NewMat(4, 4, 3)
	img.SetTo(128)
	ch := ComputeNMChannels(img, NMIHSGrad)
	if len(ch) != 4 {
		t.Fatalf("got %d channels, want 4", len(ch))
	}
}

func TestComputeNMChannelsPanicsOnGray(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Errorf("expected panic on single-channel input")
		}
	}()
	ComputeNMChannels(cv.NewMat(4, 4, 1), NMRGBLGrad)
}
