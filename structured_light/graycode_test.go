package structured_light

import (
	"testing"

	cv "github.com/malcolmston/opencv"
)

func TestNumberOfPatternImages(t *testing.T) {
	g := NewGrayCodePattern(GrayCodeParams{Width: 64, Height: 32})
	// 64 -> 6 bits, 32 -> 5 bits => 2*(6+5) = 22.
	if got, want := g.NumColBits(), 6; got != want {
		t.Fatalf("NumColBits = %d, want %d", got, want)
	}
	if got, want := g.NumRowBits(), 5; got != want {
		t.Fatalf("NumRowBits = %d, want %d", got, want)
	}
	if got, want := g.NumberOfPatternImages(), 22; got != want {
		t.Fatalf("NumberOfPatternImages = %d, want %d", got, want)
	}
	imgs := g.Generate()
	if len(imgs) != want22 {
		t.Fatalf("Generate produced %d images, want 22", len(imgs))
	}
	for i, m := range imgs {
		if m.Rows != 32 || m.Cols != 64 || m.Channels != 1 {
			t.Fatalf("image %d is %dx%dx%d, want 32x64x1", i, m.Rows, m.Cols, m.Channels)
		}
	}
}

const want22 = 22

func TestGrayBinaryRoundTrip(t *testing.T) {
	for n := uint(0); n < 1024; n++ {
		if got := grayToBinary(binaryToGray(n)); got != n {
			t.Fatalf("round trip failed for %d: got %d", n, got)
		}
	}
	// Adjacent Gray codes differ in exactly one bit.
	for n := uint(1); n < 1024; n++ {
		diff := binaryToGray(n) ^ binaryToGray(n-1)
		if diff&(diff-1) != 0 {
			t.Fatalf("Gray codes for %d and %d differ in more than one bit", n-1, n)
		}
	}
}

// captureIdentity simulates capturing the projected stack from a camera that
// coincides with the projector (identity correspondence) over a lit rectangle;
// pixels outside the rectangle stay dark, exercising the shadow mask.
func captureIdentity(patterns []*cv.Mat, litX0, litY0, litX1, litY1 int) (captured []*cv.Mat, white, black *cv.Mat) {
	rows, cols := patterns[0].Rows, patterns[0].Cols
	lit := func(y, x int) bool { return x >= litX0 && x < litX1 && y >= litY0 && y < litY1 }
	captured = make([]*cv.Mat, len(patterns))
	for i, p := range patterns {
		m := cv.NewMat(rows, cols, 1)
		for y := 0; y < rows; y++ {
			for x := 0; x < cols; x++ {
				if lit(y, x) {
					m.Set(y, x, 0, p.At(y, x, 0))
				}
			}
		}
		captured[i] = m
	}
	white = cv.NewMat(rows, cols, 1)
	black = cv.NewMat(rows, cols, 1)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			if lit(y, x) {
				white.Set(y, x, 0, 255)
			}
		}
	}
	return captured, white, black
}

func TestGrayCodeDecodeIdentity(t *testing.T) {
	g := NewGrayCodePattern(GrayCodeParams{Width: 64, Height: 32})
	patterns := g.Generate()
	litX0, litY0, litX1, litY1 := 5, 3, 60, 30
	captured, white, black := captureIdentity(patterns, litX0, litY0, litX1, litY1)

	dec, err := g.Decode(captured, white, black)
	if err != nil {
		t.Fatalf("Decode error: %v", err)
	}
	if dec.Rows != 32 || dec.Cols != 64 {
		t.Fatalf("decoded size %dx%d, want 32x64", dec.Rows, dec.Cols)
	}

	litCount := 0
	for y := 0; y < 32; y++ {
		for x := 0; x < 64; x++ {
			col, row, ok := dec.At(y, x)
			inLit := x >= litX0 && x < litX1 && y >= litY0 && y < litY1
			if inLit {
				if !ok {
					t.Fatalf("lit pixel (%d,%d) decoded invalid", x, y)
				}
				if col != x || row != y {
					t.Fatalf("pixel (%d,%d) decoded to (col=%d,row=%d), want (%d,%d)", x, y, col, row, x, y)
				}
				litCount++
			} else if ok {
				t.Fatalf("shadow pixel (%d,%d) decoded valid to (col=%d,row=%d)", x, y, col, row)
			}
		}
	}
	if want := (litX1 - litX0) * (litY1 - litY0); litCount != want {
		t.Fatalf("decoded %d lit pixels, want %d", litCount, want)
	}
}

// TestGrayCodeDecodeShifted uses a non-identity but known camera→projector
// mapping (a horizontal shift) to confirm the decoded coordinate follows the
// mapping rather than the pixel position.
func TestGrayCodeDecodeShifted(t *testing.T) {
	g := NewGrayCodePattern(GrayCodeParams{Width: 128, Height: 8})
	patterns := g.Generate()
	rows, cols := 8, 128
	const shift = 10 // projector col = camera col + shift

	captured := make([]*cv.Mat, len(patterns))
	for i, p := range patterns {
		m := cv.NewMat(rows, cols, 1)
		for y := 0; y < rows; y++ {
			for x := 0; x < cols; x++ {
				px := x + shift
				if px < cols {
					m.Set(y, x, 0, p.At(y, px, 0))
				}
			}
		}
		captured[i] = m
	}
	white := cv.NewMat(rows, cols, 1)
	black := cv.NewMat(rows, cols, 1)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			if x+shift < cols {
				white.Set(y, x, 0, 255)
			}
		}
	}

	dec, err := g.Decode(captured, white, black)
	if err != nil {
		t.Fatalf("Decode error: %v", err)
	}
	for y := 0; y < rows; y++ {
		for x := 0; x < cols-shift; x++ {
			col, row, ok := dec.At(y, x)
			if !ok {
				t.Fatalf("pixel (%d,%d) invalid", x, y)
			}
			if col != x+shift || row != y {
				t.Fatalf("pixel (%d,%d) decoded (col=%d,row=%d), want (%d,%d)", x, y, col, row, x+shift, y)
			}
		}
	}
}

func TestReferenceImages(t *testing.T) {
	g := NewGrayCodePattern(GrayCodeParams{Width: 8, Height: 4})
	white, black := g.ReferenceImages()
	for i := range white.Data {
		if white.Data[i] != 255 {
			t.Fatalf("white[%d] = %d, want 255", i, white.Data[i])
		}
		if black.Data[i] != 0 {
			t.Fatalf("black[%d] = %d, want 0", i, black.Data[i])
		}
	}
}

func TestDecodeErrors(t *testing.T) {
	g := NewGrayCodePattern(GrayCodeParams{Width: 8, Height: 4})
	white, black := g.ReferenceImages()

	if _, err := g.Decode(nil, white, black); err == nil {
		t.Fatal("expected error for wrong stack size")
	}
	patterns := g.Generate()
	if _, err := g.Decode(patterns, nil, black); err == nil {
		t.Fatal("expected error for nil white")
	}
	// mismatched reference sizes
	badBlack := cv.NewMat(2, 2, 1)
	if _, err := g.Decode(patterns, white, badBlack); err == nil {
		t.Fatal("expected error for size mismatch")
	}
}

func TestNewGrayCodePatternPanics(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("expected panic for tiny dimensions")
		}
	}()
	NewGrayCodePattern(GrayCodeParams{Width: 1, Height: 1})
}

func TestVisualizeHelpers(t *testing.T) {
	coord := []int{-1, 0, 5, 10}
	m := CoordMapToMat(coord, 2, 2, 10)
	if m.Data[0] != 0 {
		t.Fatalf("invalid coord should map to 0, got %d", m.Data[0])
	}
	if m.Data[3] != 255 {
		t.Fatalf("max coord should map to 255, got %d", m.Data[3])
	}
	if m.Data[2] <= m.Data[1] {
		t.Fatal("expected monotonic scaling")
	}

	mask := []bool{true, false, false, true}
	mm := MaskToMat(mask, 2, 2)
	if mm.Data[0] != 255 || mm.Data[1] != 0 {
		t.Fatal("MaskToMat wrong values")
	}
}
