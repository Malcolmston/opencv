package structured_light

import "testing"

func TestCodePatternDecodeIdentity(t *testing.T) {
	for _, enc := range []Encoding{EncodingGray, EncodingBinary} {
		c := NewCodePattern(GrayCodeParams{Width: 64, Height: 32}, enc)
		if c.NumberOfPatternImages() != 22 {
			t.Fatalf("%s: NumberOfPatternImages = %d, want 22", enc, c.NumberOfPatternImages())
		}
		patterns := c.Generate()
		captured, white, black := captureIdentity(patterns, 0, 0, 64, 32)
		dec, err := c.Decode(captured, white, black)
		if err != nil {
			t.Fatalf("%s: Decode error: %v", enc, err)
		}
		for y := 0; y < 32; y++ {
			for x := 0; x < 64; x++ {
				col, row, ok := dec.At(y, x)
				if !ok || col != x || row != y {
					t.Fatalf("%s: pixel (%d,%d) decoded (col=%d,row=%d,ok=%v)", enc, x, y, col, row, ok)
				}
			}
		}
	}
}

func TestCodePatternGrayMatchesGrayCodePattern(t *testing.T) {
	c := NewCodePattern(GrayCodeParams{Width: 16, Height: 8}, EncodingGray)
	g := NewGrayCodePattern(GrayCodeParams{Width: 16, Height: 8})
	cp := c.Generate()
	gp := g.Generate()
	if len(cp) != len(gp) {
		t.Fatalf("stack sizes differ: %d vs %d", len(cp), len(gp))
	}
	for i := range cp {
		for j := range cp[i].Data {
			if cp[i].Data[j] != gp[i].Data[j] {
				t.Fatalf("gray CodePattern image %d differs from GrayCodePattern at %d", i, j)
			}
		}
	}
}

func TestBinaryDiffersFromGray(t *testing.T) {
	// Binary and Gray encodings must actually project different bit patterns.
	bin := NewCodePattern(GrayCodeParams{Width: 16, Height: 4}, EncodingBinary).Generate()
	gray := NewCodePattern(GrayCodeParams{Width: 16, Height: 4}, EncodingGray).Generate()
	differ := false
	for i := range bin {
		for j := range bin[i].Data {
			if bin[i].Data[j] != gray[i].Data[j] {
				differ = true
			}
		}
	}
	if !differ {
		t.Fatal("binary and gray encodings produced identical stacks")
	}
	if EncodingGray.String() != "gray" || EncodingBinary.String() != "binary" {
		t.Fatal("Encoding.String() wrong")
	}
}

func TestNewCodePatternPanics(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("expected panic for tiny dimensions")
		}
	}()
	NewCodePattern(GrayCodeParams{Width: 1, Height: 1}, EncodingGray)
}
