package text

import (
	"testing"

	cv "github.com/malcolmston/opencv"
)

// drawBar fills a bright vertical bar of the given width and height at column x0.
func drawBar(img *cv.Mat, x0, y0, w, h int) {
	for y := y0; y < y0+h; y++ {
		for x := x0; x < x0+w; x++ {
			img.Set(y, x, 0, 255)
		}
	}
}

func TestStrokeWidthTransformConstantOnBar(t *testing.T) {
	img := cv.NewMat(30, 20, 1)
	drawBar(img, 8, 3, 3, 24)

	swt := StrokeWidthTransform(img, DefaultSWTParams())
	// Interior bar pixels must carry a positive, uniform stroke width.
	var vals []float64
	for y := 6; y < 24; y++ {
		v := swt[y*img.Cols+9] // centre column of the bar
		if v <= 0 {
			t.Fatalf("bar centre (%d,9) has no stroke width", y)
		}
		vals = append(vals, v)
	}
	for _, v := range vals {
		if v != vals[0] {
			t.Errorf("stroke width not constant along bar: %v", vals)
			break
		}
	}
}

func TestTextDetectorSWTFindsBars(t *testing.T) {
	img := cv.NewMat(30, 40, 1)
	drawBar(img, 6, 3, 3, 24)
	drawBar(img, 24, 3, 3, 24)

	d := NewTextDetectorSWT(DefaultSWTParams())
	comps := d.DetectComponents(img)
	if len(comps) != 2 {
		t.Fatalf("got %d components, want 2: %+v", len(comps), comps)
	}
	for _, c := range comps {
		if c.StrokeWidthStd > 0.5 {
			t.Errorf("component %+v stroke width not uniform (std %.2f)", c.Rect, c.StrokeWidthStd)
		}
		if c.StrokeWidthMean <= 0 || c.StrokeWidthMean > 5 {
			t.Errorf("component %+v implausible stroke width %.2f", c.Rect, c.StrokeWidthMean)
		}
	}

	boxes := d.Detect(img)
	if len(boxes) != 2 {
		t.Fatalf("Detect got %d boxes, want 2", len(boxes))
	}
	// Boxes must land on the planted bars.
	if boxes[0].X > 6 || boxes[0].X+boxes[0].Width < 9 {
		t.Errorf("first box %+v does not cover the first bar", boxes[0])
	}
}

func TestStrokeWidthTransformEmptyOnFlat(t *testing.T) {
	img := cv.NewMat(20, 20, 1)
	img.SetTo(128) // no edges at all
	swt := StrokeWidthTransform(img, DefaultSWTParams())
	for _, v := range swt {
		if v != 0 {
			t.Fatalf("flat image produced a nonzero stroke width %v", v)
		}
	}
}
