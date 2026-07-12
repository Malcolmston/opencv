package photo

import (
	"testing"

	cv "github.com/malcolmston/opencv"
)

func TestDecolorSingleChannelWithContrast(t *testing.T) {
	// Build an image with several distinct, fairly iso-luminant colours so plain
	// luma would lose contrast that a variance-maximising mixture keeps.
	img := cv.NewMat(8, 8, 3)
	colors := [][]uint8{
		{200, 40, 40},
		{40, 200, 40},
		{40, 40, 200},
		{200, 200, 40},
	}
	for y := 0; y < 8; y++ {
		for x := 0; x < 8; x++ {
			img.SetPixel(y, x, colors[(x/4)+(y/4)*2])
		}
	}

	gray, boost := Decolor(img)
	if gray.Channels != 1 {
		t.Fatalf("gray must be single-channel, got %d", gray.Channels)
	}
	if boost.Channels != 3 {
		t.Fatalf("boost must be three-channel, got %d", boost.Channels)
	}
	_, variance := meanVar(gray)
	if variance < 50 {
		t.Errorf("decolor produced low contrast: variance=%.1f", variance)
	}
	t.Logf("gray variance=%.1f", variance)
}

func TestDecolorConstantImage(t *testing.T) {
	img := cv.NewMat(4, 4, 3)
	for i := range img.Data {
		img.Data[i] = 90
	}
	gray, _ := Decolor(img)
	for _, v := range gray.Data {
		if v != 90 {
			t.Fatalf("constant image should decolor to constant, got %d", v)
		}
	}
}
