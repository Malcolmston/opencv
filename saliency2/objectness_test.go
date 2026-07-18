package saliency2_test

import (
	"image"
	"testing"

	saliency2pkg "github.com/malcolmston/opencv/saliency2"
)

// TestBoxGeometry checks the Box helper arithmetic against hand values.
func TestBoxGeometry(t *testing.T) {
	a := saliency2pkg.Box{Rect: image.Rect(0, 0, 10, 10)}
	b := saliency2pkg.Box{Rect: image.Rect(5, 5, 15, 15)}
	if a.Area() != 100 {
		t.Fatalf("Area = %d, want 100", a.Area())
	}
	x, y := a.Center()
	if x != 5 || y != 5 {
		t.Fatalf("Center = (%v,%v), want (5,5)", x, y)
	}
	// Intersection 5x5=25, union 100+100-25=175 -> IoU 25/175.
	if got := a.IoU(b); absf(got-25.0/175.0) > 1e-9 {
		t.Fatalf("IoU = %v, want %v", got, 25.0/175.0)
	}
	disjoint := saliency2pkg.Box{Rect: image.Rect(50, 50, 60, 60)}
	if a.IoU(disjoint) != 0 {
		t.Fatalf("disjoint IoU = %v, want 0", a.IoU(disjoint))
	}
}

func absf(v float64) float64 {
	if v < 0 {
		return -v
	}
	return v
}

// TestObjectnessLocalisesSquare checks the top BING-lite proposal snaps onto a
// bright square whose edges align with a candidate window.
func TestObjectnessLocalisesSquare(t *testing.T) {
	const size = 64
	const x0, y0, side = 16, 16, 32 // side = size/2 is a candidate width
	img := squareImage(size, x0, y0, side, 30, 220)

	boxes := saliency2pkg.NewObjectnessBING().ObjectnessBoundingBoxes(img)
	if len(boxes) == 0 {
		t.Fatal("no proposals returned")
	}
	// Proposals must be ordered by non-increasing score.
	for i := 1; i < len(boxes); i++ {
		if boxes[i].Score > boxes[i-1].Score+1e-12 {
			t.Fatalf("proposals not sorted at %d", i)
		}
	}
	truth := saliency2pkg.Box{Rect: image.Rect(x0, y0, x0+side, y0+side)}
	if iou := boxes[0].IoU(truth); iou < 0.6 {
		t.Fatalf("top proposal IoU with square = %.3f (rect %v), want >= 0.6", iou, boxes[0].Rect)
	}
}

// TestObjectnessMapShape checks the objectness field is single channel and
// image-sized.
func TestObjectnessMapShape(t *testing.T) {
	img := squareImage(32, 8, 8, 16, 20, 200)
	m := saliency2pkg.NewObjectnessBING().ObjectnessMap(img)
	if m.Rows != 32 || m.Cols != 32 {
		t.Fatalf("objectness map %dx%d, want 32x32", m.Rows, m.Cols)
	}
	out := saliency2pkg.NewObjectnessBING().ComputeSaliency(img)
	if out.Channels != 1 {
		t.Fatalf("ComputeSaliency channels = %d, want 1", out.Channels)
	}
}

// TestNormedGradientEdges checks the normed-gradient map peaks on the object
// boundary, not its flat interior or the flat background.
func TestNormedGradientEdges(t *testing.T) {
	const size = 40
	img := squareImage(size, 12, 12, 16, 30, 220)
	ng := saliency2pkg.NormedGradient(img)
	edge := ng.At(12, 20)     // top edge of the square
	interior := ng.At(20, 20) // deep inside the square
	back := ng.At(2, 2)       // far background
	if !(edge > interior && edge > back) {
		t.Fatalf("edge %.2f should exceed interior %.2f and background %.2f", edge, interior, back)
	}
}
