package photo

import (
	"testing"

	cv "github.com/malcolmston/opencv"
)

// stepImage builds a single-channel image with a vertical edge: left half dark,
// right half bright.
func stepImage(rows, cols int, lo, hi uint8) *cv.Mat {
	m := cv.NewMat(rows, cols, 1)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			v := lo
			if x >= cols/2 {
				v = hi
			}
			m.Set(y, x, 0, v)
		}
	}
	return m
}

func TestEdgePreservingFilterKeepsEdge(t *testing.T) {
	img := stepImage(12, 12, 30, 220)
	out := EdgePreservingFilter(img, RecursFilter, 30, 0.2)
	if out.Rows != 12 || out.Cols != 12 || out.Channels != 1 {
		t.Fatalf("unexpected shape %dx%dx%d", out.Rows, out.Cols, out.Channels)
	}
	// The strong central edge must survive: far-left stays dark, far-right bright.
	if out.At(6, 0, 0) > 80 {
		t.Errorf("left side brightened too much: %d", out.At(6, 0, 0))
	}
	if out.At(6, 11, 0) < 170 {
		t.Errorf("right side darkened too much: %d", out.At(6, 11, 0))
	}
}

func TestDetailEnhanceIncreasesContrast(t *testing.T) {
	// A low-amplitude checkerboard texture: edge-preserving smoothing averages it
	// toward the local mean, and enhancement amplifies the recovered detail.
	img := cv.NewMat(12, 12, 1)
	for y := 0; y < 12; y++ {
		for x := 0; x < 12; x++ {
			v := uint8(105)
			if (x+y)%2 == 0 {
				v = 115
			}
			img.Set(y, x, 0, v)
		}
	}
	_, varIn := meanVar(img)
	out := DetailEnhance(img, 30, 0.4)
	_, varOut := meanVar(out)
	if varOut <= varIn {
		t.Errorf("detail enhance did not increase contrast: var %.1f -> %.1f", varIn, varOut)
	}
	t.Logf("variance %.1f -> %.1f", varIn, varOut)
}

func TestStylizationShape(t *testing.T) {
	img := cv.NewMat(10, 10, 3)
	for i := range img.Data {
		img.Data[i] = uint8((i * 7) % 256)
	}
	out := Stylization(img, 40, 0.3)
	if out.Rows != 10 || out.Cols != 10 || out.Channels != 3 {
		t.Fatalf("unexpected shape %dx%dx%d", out.Rows, out.Cols, out.Channels)
	}
}
