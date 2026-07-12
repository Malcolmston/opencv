package objdetect

import (
	"testing"

	cv "github.com/malcolmston/opencv"
)

// drawFinderPattern paints a QR finder pattern (7x7 modules of the given module
// size) with its top-left module corner at (ox, oy): a 7x7 dark square, a 5x5
// white square inset by one module, and a 3x3 dark centre.
func drawFinderPattern(img *cv.Mat, ox, oy, module int) {
	fill := func(x0, y0, w, h int, v uint8) {
		for y := y0; y < y0+h; y++ {
			for x := x0; x < x0+w; x++ {
				img.Set(y, x, 0, v)
			}
		}
	}
	fill(ox, oy, 7*module, 7*module, 0)                   // outer dark
	fill(ox+module, oy+module, 5*module, 5*module, 255)   // inner white
	fill(ox+2*module, oy+2*module, 3*module, 3*module, 0) // centre dark
}

// TestQRFinderDetection builds a synthetic QR-like image with three finder
// patterns and checks the detector locates all three near their true centres.
func TestQRFinderDetection(t *testing.T) {
	const (
		size   = 140
		module = 4
		patPx  = 7 * module // 28
	)
	img := cv.NewMat(size, size, 1)
	img.SetTo(255) // white background

	// Three finder patterns: top-left, top-right, bottom-left.
	origins := [][2]int{
		{10, 10},
		{size - patPx - 10, 10},
		{10, size - patPx - 10},
	}
	for _, o := range origins {
		drawFinderPattern(img, o[0], o[1], module)
	}

	d := NewQRCodeDetector()
	corners, found := d.Detect(img)
	if !found {
		t.Fatalf("expected to find QR finder patterns, got found=false (%d corners)", len(corners))
	}
	if len(corners) != 3 {
		t.Fatalf("expected 3 corners, got %d: %v", len(corners), corners)
	}

	// Each true centre is origin + 3.5*module.
	half := patPx / 2
	trueCenters := make([][2]int, 3)
	for i, o := range origins {
		trueCenters[i] = [2]int{o[0] + half, o[1] + half}
	}

	tol := module + 1
	for _, tc := range trueCenters {
		matched := false
		for _, c := range corners {
			if abs(c.X-tc[0]) <= tol && abs(c.Y-tc[1]) <= tol {
				matched = true
				break
			}
		}
		if !matched {
			t.Fatalf("no detected corner near true centre %v; corners=%v", tc, corners)
		}
	}
}

// TestQRNoPatterns verifies a blank image yields no detection.
func TestQRNoPatterns(t *testing.T) {
	img := cv.NewMat(80, 80, 1)
	img.SetTo(255)
	d := NewQRCodeDetector()
	corners, found := d.Detect(img)
	if found {
		t.Fatalf("expected no patterns on blank image, got %v", corners)
	}
}

// TestCheckFinderRatio unit-tests the ratio predicate directly.
func TestCheckFinderRatio(t *testing.T) {
	if !checkFinderRatio([]int{4, 4, 12, 4, 4}) {
		t.Fatal("perfect 1:1:3:1:1 should pass")
	}
	if !checkFinderRatio([]int{3, 4, 13, 5, 4}) {
		t.Fatal("slightly noisy pattern should pass")
	}
	if checkFinderRatio([]int{4, 4, 4, 4, 4}) {
		t.Fatal("uniform runs (no 3x centre) should fail")
	}
	if checkFinderRatio([]int{1, 1, 1, 1, 1}) {
		t.Fatal("too-small total should fail")
	}
}

func abs(v int) int {
	if v < 0 {
		return -v
	}
	return v
}
