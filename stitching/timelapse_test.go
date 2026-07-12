package stitching

import (
	"image"
	"testing"

	cv "github.com/malcolmston/opencv"
)

func solidImage(rows, cols int, val uint8) *cv.Mat {
	m := cv.NewMat(rows, cols, 1)
	for i := range m.Data {
		m.Data[i] = val
	}
	return m
}

func TestTimelapserForCornersSizeAndPlacement(t *testing.T) {
	imgA := solidImage(30, 40, 100)
	imgB := solidImage(30, 40, 200)
	corners := []image.Point{{X: 0, Y: 0}, {X: 25, Y: 5}}
	sizes := []image.Point{{X: 40, Y: 30}, {X: 40, Y: 30}}

	tl := NewTimelapserForCorners(corners, sizes, 1)
	rows, cols := tl.Size()
	// Union: x in [0,65), y in [0,35).
	if cols != 65 || rows != 35 {
		t.Fatalf("canvas = %dx%d, want 65x35", cols, rows)
	}
	tl.Process(imgA, nil, corners[0])
	tl.Process(imgB, nil, corners[1])
	canvas := tl.Canvas()
	// A pixel unique to A.
	if got := canvas.Data[2*cols+2]; got != 100 {
		t.Errorf("canvas at A-only = %d, want 100", got)
	}
	// A pixel unique to B (global (60,30) -> canvas (60,30)).
	if got := canvas.Data[30*cols+60]; got != 200 {
		t.Errorf("canvas at B-only = %d, want 200", got)
	}
}

func TestTimelapserFrameIsolatesImage(t *testing.T) {
	imgB := solidImage(20, 20, 200)
	corners := []image.Point{{X: 0, Y: 0}, {X: 30, Y: 0}}
	sizes := []image.Point{{X: 20, Y: 20}, {X: 20, Y: 20}}
	tl := NewTimelapserForCorners(corners, sizes, 1)

	frame := tl.Frame(imgB, nil, corners[1])
	_, cols := tl.Size()
	// B's region is filled.
	if frame.Data[5*cols+35] != 200 {
		t.Errorf("frame B region = %d, want 200", frame.Data[5*cols+35])
	}
	// A's region is blank in B's frame.
	if frame.Data[5*cols+5] != 0 {
		t.Errorf("frame A region = %d, want 0 (blank)", frame.Data[5*cols+5])
	}
}

func TestTimelapserMaskSkipsBlack(t *testing.T) {
	canvasImg := solidImage(10, 10, 50)
	tl := NewTimelapser(10, 10, 1)
	tl.Process(canvasImg, nil, image.Point{})
	// Overlay an image but with a zero mask everywhere: nothing should change.
	overlay := solidImage(10, 10, 255)
	mask := cv.NewFloatMat(10, 10) // all zero
	tl.Process(overlay, mask, image.Point{})
	for i, v := range tl.Canvas().Data {
		if v != 50 {
			t.Fatalf("masked process changed pixel %d to %d", i, v)
		}
	}
}
