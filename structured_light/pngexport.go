package structured_light

import (
	"bytes"
	"fmt"
	"image/png"
	"io"
	"os"
	"path/filepath"

	cv "github.com/malcolmston/opencv"
)

// WritePatternPNG encodes a single pattern image as a PNG and writes it to w. The
// Mat is rendered through its standard-library image form (grayscale for a
// single channel, RGB otherwise). It returns any encoding or write error.
func WritePatternPNG(w io.Writer, m *cv.Mat) error {
	if m == nil {
		return fmt.Errorf("structured_light: WritePatternPNG got nil Mat")
	}
	return png.Encode(w, m.ToImage())
}

// EncodePatternPNG returns the PNG encoding of a pattern image as a byte slice,
// convenient for embedding a generated pattern in memory or a test. It returns
// any encoding error.
func EncodePatternPNG(m *cv.Mat) ([]byte, error) {
	var buf bytes.Buffer
	if err := WritePatternPNG(&buf, m); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

// SavePatternPNG writes a single pattern image to path as a PNG file, creating
// or truncating it. It returns any file or encoding error.
func SavePatternPNG(path string, m *cv.Mat) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	if err := WritePatternPNG(f, m); err != nil {
		f.Close()
		return err
	}
	return f.Close()
}

// SavePatternStack writes an entire pattern stack to dir as zero-padded PNG
// files named "<prefix>NN.png" (for example projector patterns from
// [GrayCodePattern.Generate] or [SinusoidalPattern.Generate]). The directory is
// created if necessary. It returns the paths written, in stack order, and any
// error.
func SavePatternStack(dir, prefix string, stack []*cv.Mat) ([]string, error) {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, err
	}
	paths := make([]string, 0, len(stack))
	width := len(fmt.Sprintf("%d", len(stack)))
	if width < 2 {
		width = 2
	}
	for i, m := range stack {
		name := fmt.Sprintf("%s%0*d.png", prefix, width, i)
		path := filepath.Join(dir, name)
		if err := SavePatternPNG(path, m); err != nil {
			return paths, err
		}
		paths = append(paths, path)
	}
	return paths, nil
}
