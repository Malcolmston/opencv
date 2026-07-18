package textdet

import (
	"testing"

	cv "github.com/malcolmston/opencv"
)

// twoDarkSquares builds a 20x24 white image with two 6x6 black squares.
func twoDarkSquares() *cv.Mat {
	m := newGray(20, 24, 255)
	paintRect(m, 2, 2, 6, 6, 0)
	paintRect(m, 14, 2, 6, 6, 0)
	return m
}

func TestDetectMSERDarkSquares(t *testing.T) {
	regions, err := DetectMSER(twoDarkSquares(), DefaultMSEROptions())
	if err != nil {
		t.Fatal(err)
	}
	if len(regions) != 2 {
		t.Fatalf("regions = %d, want 2", len(regions))
	}
	for _, r := range regions {
		if r.Area != 36 {
			t.Fatalf("region area = %d, want 36", r.Area)
		}
		if r.Bounds.Width != 6 || r.Bounds.Height != 6 {
			t.Fatalf("region bounds = %+v, want 6x6", r.Bounds)
		}
		if r.Variation != 0 {
			t.Fatalf("region variation = %v, want 0 (perfectly stable)", r.Variation)
		}
	}
	// Regions are ordered left-to-right, so the first is the left square.
	if regions[0].Bounds.X != 2 || regions[1].Bounds.X != 14 {
		t.Fatalf("region order X = %d,%d want 2,14", regions[0].Bounds.X, regions[1].Bounds.X)
	}
}

func TestDetectMSERBrightPolarity(t *testing.T) {
	// Invert: two bright squares on a dark page. MSERBright should recover them.
	m := newGray(20, 24, 0)
	paintRect(m, 2, 2, 6, 6, 255)
	paintRect(m, 14, 2, 6, 6, 255)
	opts := DefaultMSEROptions()
	opts.Polarity = MSERBright
	regions, err := DetectMSER(m, opts)
	if err != nil {
		t.Fatal(err)
	}
	if len(regions) != 2 {
		t.Fatalf("bright regions = %d, want 2", len(regions))
	}
}

func TestFilterTextRegions(t *testing.T) {
	regions := []MSERRegion{
		{Bounds: cv.Rect{Width: 6, Height: 6}, Area: 36},   // square, full
		{Bounds: cv.Rect{Width: 40, Height: 3}, Area: 120}, // very wide (line)
	}
	got := FilterTextRegions(regions, 0.3, 3.0, 4, 0, 0.5)
	if len(got) != 1 {
		t.Fatalf("filtered = %d, want 1", len(got))
	}
	if got[0].Bounds.Width != 6 {
		t.Fatalf("kept wrong region: %+v", got[0].Bounds)
	}
}

func TestRegionsToMask(t *testing.T) {
	regions, _ := DetectMSER(twoDarkSquares(), DefaultMSEROptions())
	mask, err := RegionsToMask(regions, 20, 24)
	if err != nil {
		t.Fatal(err)
	}
	// Exactly 72 pixels (two 6x6 squares) should be foreground.
	fg := 0
	for _, v := range mask.Data {
		if v != 0 {
			fg++
		}
	}
	if fg != 72 {
		t.Fatalf("mask foreground = %d, want 72", fg)
	}
}

func TestMSERErrors(t *testing.T) {
	var empty cv.Mat
	if _, err := DetectMSER(&empty, DefaultMSEROptions()); err != ErrEmpty {
		t.Fatalf("empty err = %v, want ErrEmpty", err)
	}
	opts := DefaultMSEROptions()
	opts.Delta = 0
	if _, err := DetectMSER(newGray(4, 4, 0), opts); err != ErrInvalidArgument {
		t.Fatalf("delta 0 err = %v, want ErrInvalidArgument", err)
	}
}

// BenchmarkDetectMSER exercises the heaviest routine: the 256-level threshold
// sweep with connected-component labelling at every level.
func BenchmarkDetectMSER(b *testing.B) {
	m := newGray(48, 48, 255)
	paintRect(m, 4, 4, 8, 8, 0)
	paintRect(m, 20, 4, 8, 8, 0)
	paintRect(m, 36, 4, 8, 8, 0)
	paintRect(m, 4, 20, 8, 8, 0)
	opts := DefaultMSEROptions()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := DetectMSER(m, opts); err != nil {
			b.Fatal(err)
		}
	}
}
