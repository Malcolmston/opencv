package cv

import (
	"path/filepath"
	"testing"
)

func TestIMEncodeDecodePNGRoundTrip(t *testing.T) {
	m := NewMat(3, 3, 3)
	for i := range m.Data {
		m.Data[i] = uint8(i * 5)
	}
	data, err := IMEncode("png", m)
	if err != nil {
		t.Fatalf("IMEncode: %v", err)
	}
	back, err := IMDecode(data)
	if err != nil {
		t.Fatalf("IMDecode: %v", err)
	}
	if back.Rows != 3 || back.Cols != 3 || back.Channels != 3 {
		t.Fatalf("decoded dims %dx%dx%d", back.Rows, back.Cols, back.Channels)
	}
	// PNG is lossless, so the data must match exactly.
	for i := range m.Data {
		if back.Data[i] != m.Data[i] {
			t.Fatalf("png round-trip mismatch at %d: %d vs %d", i, back.Data[i], m.Data[i])
		}
	}
}

func TestImWriteImReadPNG(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "out.png")
	m := NewMat(2, 2, 1)
	m.SetTo(123)
	if err := ImWrite(path, m); err != nil {
		t.Fatalf("ImWrite: %v", err)
	}
	back, err := ImRead(path)
	if err != nil {
		t.Fatalf("ImRead: %v", err)
	}
	if back.Channels != 1 || back.At(0, 0, 0) != 123 {
		t.Errorf("read back %d channels, value %d", back.Channels, back.At(0, 0, 0))
	}
}

func TestIMEncodeUnsupportedFormat(t *testing.T) {
	m := NewMat(1, 1, 1)
	if _, err := IMEncode("gif", m); err == nil {
		t.Error("expected error for unsupported format")
	}
}

func TestImReadMissingFile(t *testing.T) {
	if _, err := ImRead("/nonexistent/does-not-exist.png"); err == nil {
		t.Error("expected error reading missing file")
	}
}
