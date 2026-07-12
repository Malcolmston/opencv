package cv

import (
	"bytes"
	"fmt"
	"image"
	"image/jpeg"
	"image/png"
	"os"
	"strings"
)

// ImRead loads an image file from path and returns it as a Mat. PNG and JPEG
// are supported through the standard library decoders. The channel count of
// the result follows [FromImage]: grayscale files yield a single-channel Mat,
// everything else yields three-channel RGB.
func ImRead(path string) (*Mat, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("cv: ImRead: %w", err)
	}
	defer f.Close()
	img, _, err := image.Decode(f)
	if err != nil {
		return nil, fmt.Errorf("cv: ImRead decode %q: %w", path, err)
	}
	return FromImage(img), nil
}

// ImWrite encodes the Mat and writes it to path. The format is chosen from the
// file extension: ".png" uses PNG, ".jpg"/".jpeg" uses JPEG at quality 95.
func ImWrite(path string, m *Mat) error {
	data, err := IMEncode(extFormat(path), m)
	if err != nil {
		return err
	}
	if err := os.WriteFile(path, data, 0o644); err != nil {
		return fmt.Errorf("cv: ImWrite: %w", err)
	}
	return nil
}

// IMDecode decodes an in-memory image (PNG or JPEG) into a Mat.
func IMDecode(data []byte) (*Mat, error) {
	img, _, err := image.Decode(bytes.NewReader(data))
	if err != nil {
		return nil, fmt.Errorf("cv: IMDecode: %w", err)
	}
	return FromImage(img), nil
}

// IMEncode encodes the Mat into a byte slice using the named format. Valid
// formats are "png", "jpg" and "jpeg" (case-insensitive, a leading dot is
// ignored).
func IMEncode(format string, m *Mat) ([]byte, error) {
	img := m.ToImage()
	var buf bytes.Buffer
	switch normFormat(format) {
	case "png":
		if err := png.Encode(&buf, img); err != nil {
			return nil, fmt.Errorf("cv: IMEncode png: %w", err)
		}
	case "jpg", "jpeg":
		if err := jpeg.Encode(&buf, img, &jpeg.Options{Quality: 95}); err != nil {
			return nil, fmt.Errorf("cv: IMEncode jpeg: %w", err)
		}
	default:
		return nil, fmt.Errorf("cv: IMEncode unsupported format %q", format)
	}
	return buf.Bytes(), nil
}

func normFormat(format string) string {
	return strings.ToLower(strings.TrimPrefix(format, "."))
}

func extFormat(path string) string {
	i := strings.LastIndex(path, ".")
	if i < 0 {
		return ""
	}
	return path[i:]
}
