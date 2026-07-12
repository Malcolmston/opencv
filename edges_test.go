package cv

import "testing"

// synthSquare returns a black image containing a filled white square.
func synthSquare(size, x0, y0, side int) *Mat {
	m := NewMat(size, size, 1)
	for y := y0; y < y0+side; y++ {
		for x := x0; x < x0+side; x++ {
			m.Set(y, x, 0, 255)
		}
	}
	return m
}

func TestCannyFindsSquareEdges(t *testing.T) {
	m := synthSquare(40, 10, 10, 20)
	edges := Canny(m, 50, 150)
	// There must be some edges.
	count := 0
	for _, v := range edges.Data {
		if v == 255 {
			count++
		}
	}
	if count == 0 {
		t.Fatal("Canny found no edges on a square")
	}
	// The interior of the square is flat -> no edges there.
	if edges.At(20, 20, 0) != 0 {
		t.Error("Canny marked a flat interior pixel as an edge")
	}
	// There should be an edge somewhere along the top boundary row (y≈10).
	found := false
	for x := 8; x < 32; x++ {
		if edges.At(10, x, 0) == 255 || edges.At(9, x, 0) == 255 || edges.At(11, x, 0) == 255 {
			found = true
			break
		}
	}
	if !found {
		t.Error("Canny did not find the top edge of the square")
	}
}

func TestMatchTemplateLocatesPatch(t *testing.T) {
	// Build a 20x20 image with a distinctive 4x4 bright patch at (12, 8).
	src := NewMat(20, 20, 1)
	for i := range src.Data {
		src.Data[i] = uint8((i * 7) % 100) // structured background
	}
	const px, py = 8, 12
	// Give the patch internal structure so it has non-zero variance (required
	// for a well-defined normalised correlation coefficient).
	for y := 0; y < 4; y++ {
		for x := 0; x < 4; x++ {
			src.Set(py+y, px+x, 0, uint8(180+(y*4+x)*4))
		}
	}
	templ := src.Region(py, px, 4, 4)

	// TmSqdiff: best match is the global minimum, ~0 at the true location.
	res := MatchTemplate(src, templ, TmSqdiff)
	minVal, _, minX, minY, _, _ := MinMaxLoc(res)
	if minX != px || minY != py {
		t.Errorf("sqdiff min at (%d,%d), want (%d,%d)", minX, minY, px, py)
	}
	if minVal > 1e-6 {
		t.Errorf("sqdiff min value = %v, want ~0", minVal)
	}

	// TmCcoeffNormed: best match is the global maximum near 1.
	res2 := MatchTemplate(src, templ, TmCcoeffNormed)
	_, maxVal, _, _, maxX, maxY := MinMaxLoc(res2)
	if maxX != px || maxY != py {
		t.Errorf("ccoeff max at (%d,%d), want (%d,%d)", maxX, maxY, px, py)
	}
	if maxVal < 0.99 {
		t.Errorf("ccoeff max value = %v, want ~1", maxVal)
	}
}

func TestMinMaxLoc(t *testing.T) {
	f := NewFloatMat(2, 2)
	f.Data = []float64{1, 5, -3, 2}
	minVal, maxVal, minX, minY, maxX, maxY := MinMaxLoc(f)
	if minVal != -3 || minX != 0 || minY != 1 {
		t.Errorf("min = %v at (%d,%d)", minVal, minX, minY)
	}
	if maxVal != 5 || maxX != 1 || maxY != 0 {
		t.Errorf("max = %v at (%d,%d)", maxVal, maxX, maxY)
	}
}
