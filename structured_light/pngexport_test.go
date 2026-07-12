package structured_light

import (
	"bytes"
	"image/png"
	"os"
	"path/filepath"
	"testing"

	cv "github.com/malcolmston/opencv"
)

func TestEncodePatternPNGRoundTrip(t *testing.T) {
	g := NewGrayCodePattern(GrayCodeParams{Width: 16, Height: 8})
	m := g.Generate()[0]

	data, err := EncodePatternPNG(m)
	if err != nil {
		t.Fatalf("EncodePatternPNG error: %v", err)
	}
	img, err := png.Decode(bytes.NewReader(data))
	if err != nil {
		t.Fatalf("png.Decode error: %v", err)
	}
	back := cv.FromImage(img)
	if back.Rows != m.Rows || back.Cols != m.Cols {
		t.Fatalf("decoded size %dx%d, want %dx%d", back.Rows, back.Cols, m.Rows, m.Cols)
	}
	for i := range m.Data {
		if back.Data[i] != m.Data[i] {
			t.Fatalf("round-trip mismatch at %d: %d vs %d", i, back.Data[i], m.Data[i])
		}
	}
}

func TestSavePatternStack(t *testing.T) {
	s := NewSinusoidalPattern(Params{Width: 8, Height: 4, NumOfPatternImages: 3, Frequency: 1})
	stack := s.Generate()
	dir := t.TempDir()

	paths, err := SavePatternStack(dir, "fringe", stack)
	if err != nil {
		t.Fatalf("SavePatternStack error: %v", err)
	}
	if len(paths) != len(stack) {
		t.Fatalf("wrote %d paths, want %d", len(paths), len(stack))
	}
	for _, p := range paths {
		if _, err := os.Stat(p); err != nil {
			t.Fatalf("expected file %s: %v", p, err)
		}
	}
	if filepath.Base(paths[0]) != "fringe00.png" {
		t.Fatalf("first file name = %s, want fringe00.png", filepath.Base(paths[0]))
	}

	// WritePatternPNG to a buffer also works.
	var buf bytes.Buffer
	if err := WritePatternPNG(&buf, stack[0]); err != nil {
		t.Fatalf("WritePatternPNG error: %v", err)
	}
	if buf.Len() == 0 {
		t.Fatal("WritePatternPNG produced no bytes")
	}
}
