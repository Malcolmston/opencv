package objdetect

import (
	"testing"

	cv "github.com/malcolmston/opencv"
)

// drawSquareSymbol paints three QR finder patterns arranged as the corners of a
// square: top-left at (ox,oy), top-right at (ox+span,oy) and bottom-left at
// (ox,oy+span), each pattern being 7*module pixels.
func drawSquareSymbol(img *cv.Mat, ox, oy, span, module int) {
	drawFinderPattern(img, ox, oy, module)      // top-left
	drawFinderPattern(img, ox+span, oy, module) // top-right
	drawFinderPattern(img, ox, oy+span, module) // bottom-left
}

func TestDetectFinderPatternsSingle(t *testing.T) {
	img := cv.NewMat(112, 112, 1)
	img.SetTo(255)
	drawSquareSymbol(img, 10, 10, 60, 4)

	d := NewQRCodeDetector()
	centres := d.DetectFinderPatterns(img)
	if len(centres) != 3 {
		t.Fatalf("expected 3 finder patterns, got %d: %v", len(centres), centres)
	}
}

func TestDetectMultiSingle(t *testing.T) {
	img := cv.NewMat(112, 112, 1)
	img.SetTo(255)
	drawSquareSymbol(img, 10, 10, 60, 4)

	d := NewQRCodeDetector()
	quads, found := d.DetectMulti(img)
	if !found {
		t.Fatal("expected to find one QR symbol")
	}
	if len(quads) != 1 {
		t.Fatalf("expected 1 quad, got %d: %v", len(quads), quads)
	}
	if len(quads[0]) != 4 {
		t.Fatalf("quad should have 4 corners, got %d", len(quads[0]))
	}
}

func TestDetectMultiTwoSymbols(t *testing.T) {
	img := cv.NewMat(112, 224, 1)
	img.SetTo(255)
	drawSquareSymbol(img, 10, 10, 60, 4)  // symbol A, centres near (24,24)
	drawSquareSymbol(img, 130, 10, 60, 4) // symbol B, centres near (144,24)

	d := NewQRCodeDetector()
	centres := d.DetectFinderPatterns(img)
	if len(centres) != 6 {
		t.Fatalf("expected 6 finder patterns, got %d: %v", len(centres), centres)
	}

	quads, found := d.DetectMulti(img)
	if !found {
		t.Fatal("expected to find QR symbols")
	}
	if len(quads) < 2 {
		t.Fatalf("expected at least 2 quads, got %d: %v", len(quads), quads)
	}
	for _, q := range quads {
		if len(q) != 4 {
			t.Fatalf("quad should have 4 corners, got %d", len(q))
		}
	}
}

func TestDetectMultiNone(t *testing.T) {
	img := cv.NewMat(80, 80, 1)
	img.SetTo(255)
	d := NewQRCodeDetector()
	if quads, found := d.DetectMulti(img); found || quads != nil {
		t.Fatalf("blank image should yield no symbols, got %v", quads)
	}
}
