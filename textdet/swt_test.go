package textdet

import (
	"testing"

	cv "github.com/malcolmston/opencv"
)

func TestStrokeWidthTransformBar(t *testing.T) {
	// A black vertical bar (width 3) on a white background. Canny marks the
	// two boundary edges one pixel outside each side, so every ray crossing the
	// stroke measures a constant width of 4 pixels.
	m := newGray(20, 20, 255)
	paintRect(m, 8, 2, 3, 16, 0)
	res, err := StrokeWidthTransform(m, DefaultSWTOptions())
	if err != nil {
		t.Fatal(err)
	}
	if res.Rows != 20 || res.Cols != 20 {
		t.Fatalf("result dims = %dx%d, want 20x20", res.Rows, res.Cols)
	}
	nonZero := 0
	for _, v := range res.Width {
		if v > 0 {
			nonZero++
			if v != 4 {
				t.Fatalf("stroke width = %v, want 4", v)
			}
		}
	}
	if nonZero == 0 {
		t.Fatalf("no stroke pixels found")
	}
	if res.Max() != 4 {
		t.Fatalf("Max = %v, want 4", res.Max())
	}
}

func TestSWTLetters(t *testing.T) {
	m := newGray(20, 20, 255)
	paintRect(m, 8, 2, 3, 16, 0)
	res, _ := StrokeWidthTransform(m, DefaultSWTOptions())
	letters, err := SWTLetters(res, 3.0, 5)
	if err != nil {
		t.Fatal(err)
	}
	if len(letters) != 1 {
		t.Fatalf("letters = %d, want 1", len(letters))
	}
	l := letters[0]
	if l.MeanStrokeWidth != 4 {
		t.Fatalf("mean stroke width = %v, want 4", l.MeanStrokeWidth)
	}
	if l.StrokeWidthStd != 0 {
		t.Fatalf("stroke width std = %v, want 0", l.StrokeWidthStd)
	}
	// The candidate box must cover the bar's vertical extent.
	if l.Bounds.Height < 16 {
		t.Fatalf("letter height = %d, want >= 16", l.Bounds.Height)
	}
}

func TestSWTToMat(t *testing.T) {
	m := newGray(20, 20, 255)
	paintRect(m, 8, 2, 3, 16, 0)
	res, _ := StrokeWidthTransform(m, DefaultSWTOptions())
	vis := res.ToMat()
	if vis.Rows != 20 || vis.Cols != 20 || vis.Channels != 1 {
		t.Fatalf("ToMat shape = %dx%dx%d", vis.Rows, vis.Cols, vis.Channels)
	}
	// Uniform stroke width maps to the maximum grey level 255.
	max := uint8(0)
	for _, v := range vis.Data {
		if v > max {
			max = v
		}
	}
	if max != 255 {
		t.Fatalf("ToMat max grey = %d, want 255", max)
	}
}

func TestSWTEmptyAndInvalid(t *testing.T) {
	var empty cv.Mat
	if _, err := StrokeWidthTransform(&empty, DefaultSWTOptions()); err != ErrEmpty {
		t.Fatalf("empty err = %v, want ErrEmpty", err)
	}
	res := &SWTResult{Rows: 1, Cols: 1, Width: []float64{0}}
	if _, err := SWTLetters(res, 0.5, 1); err != ErrInvalidArgument {
		t.Fatalf("bad ratio err = %v, want ErrInvalidArgument", err)
	}
}
