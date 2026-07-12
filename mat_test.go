package cv

import (
	"image"
	"image/color"
	"testing"
)

func TestNewMatAndAccessors(t *testing.T) {
	m := NewMat(3, 4, 3)
	if m.Rows != 3 || m.Cols != 4 || m.Channels != 3 {
		t.Fatalf("unexpected dims %dx%dx%d", m.Rows, m.Cols, m.Channels)
	}
	if len(m.Data) != 3*4*3 {
		t.Fatalf("unexpected data length %d", len(m.Data))
	}
	if m.Total() != 12 {
		t.Fatalf("Total = %d, want 12", m.Total())
	}
	m.Set(1, 2, 0, 111)
	m.Set(1, 2, 1, 222)
	m.Set(1, 2, 2, 33)
	if got := m.At(1, 2, 0); got != 111 {
		t.Errorf("At(1,2,0) = %d, want 111", got)
	}
	if got := m.At(1, 2, 2); got != 33 {
		t.Errorf("At(1,2,2) = %d, want 33", got)
	}
	px := m.AtPixel(1, 2)
	if px[0] != 111 || px[1] != 222 || px[2] != 33 {
		t.Errorf("AtPixel = %v, want [111 222 33]", px)
	}
}

func TestAtOutOfRangePanics(t *testing.T) {
	m := NewMat(2, 2, 1)
	defer func() {
		if recover() == nil {
			t.Error("expected panic on out-of-range At")
		}
	}()
	_ = m.At(5, 5, 0)
}

func TestCloneIndependent(t *testing.T) {
	m := NewMat(2, 2, 1)
	m.Set(0, 0, 0, 42)
	c := m.Clone()
	if c.At(0, 0, 0) != 42 {
		t.Fatal("clone did not copy data")
	}
	c.Set(0, 0, 0, 7)
	if m.At(0, 0, 0) != 42 {
		t.Error("mutating clone affected the original")
	}
}

func TestRegionROI(t *testing.T) {
	m := NewMat(4, 4, 1)
	for y := 0; y < 4; y++ {
		for x := 0; x < 4; x++ {
			m.Set(y, x, 0, uint8(y*4+x))
		}
	}
	r := m.Region(1, 1, 2, 2) // rows 1-2, cols 1-2
	if r.Rows != 2 || r.Cols != 2 {
		t.Fatalf("region dims %dx%d", r.Rows, r.Cols)
	}
	// Expected values: (1,1)=5 (1,2)=6 (2,1)=9 (2,2)=10
	if r.At(0, 0, 0) != 5 || r.At(0, 1, 0) != 6 || r.At(1, 0, 0) != 9 || r.At(1, 1, 0) != 10 {
		t.Errorf("region content wrong: %v", r.Data)
	}
	// Region is a copy: mutating it must not touch parent.
	r.Set(0, 0, 0, 200)
	if m.At(1, 1, 0) != 5 {
		t.Error("region mutation leaked into parent")
	}
}

func TestSplitMergeRoundTrip(t *testing.T) {
	m := NewMat(2, 2, 3)
	for i := range m.Data {
		m.Data[i] = uint8(i * 3)
	}
	planes := m.Split()
	if len(planes) != 3 {
		t.Fatalf("Split returned %d planes", len(planes))
	}
	if planes[0].At(0, 0, 0) != m.At(0, 0, 0) {
		t.Error("split plane 0 mismatch")
	}
	if planes[2].At(1, 1, 0) != m.At(1, 1, 2) {
		t.Error("split plane 2 mismatch")
	}
	merged := Merge(planes)
	if merged.Channels != 3 {
		t.Fatalf("merged channels %d", merged.Channels)
	}
	for i := range m.Data {
		if merged.Data[i] != m.Data[i] {
			t.Fatalf("round-trip mismatch at %d: %d vs %d", i, merged.Data[i], m.Data[i])
		}
	}
}

func TestFromImageAndToImageGray(t *testing.T) {
	// SetGray takes (x, y); Mat.At takes (y=row, x=col).
	g := image.NewGray(image.Rect(0, 0, 2, 2))
	g.SetGray(0, 0, color.Gray{Y: 10}) // x=0,y=0 -> At(0,0)
	g.SetGray(1, 0, color.Gray{Y: 20}) // x=1,y=0 -> At(0,1)
	g.SetGray(0, 1, color.Gray{Y: 30}) // x=0,y=1 -> At(1,0)
	g.SetGray(1, 1, color.Gray{Y: 40}) // x=1,y=1 -> At(1,1)
	m := FromImage(g)
	if m.Channels != 1 {
		t.Fatalf("gray image should be 1 channel, got %d", m.Channels)
	}
	if m.At(0, 1, 0) != 20 {
		t.Errorf("At(0,1) = %d, want 20", m.At(0, 1, 0))
	}
	if m.At(1, 0, 0) != 30 {
		t.Errorf("At(1,0) = %d, want 30", m.At(1, 0, 0))
	}
	back := m.ToImage()
	gg, ok := back.(*image.Gray)
	if !ok {
		t.Fatal("ToImage did not return *image.Gray")
	}
	if gg.GrayAt(1, 0).Y != 20 {
		t.Errorf("round-trip gray mismatch, got %d want 20", gg.GrayAt(1, 0).Y)
	}
}

func TestFromImageRGBA(t *testing.T) {
	rgba := image.NewRGBA(image.Rect(0, 0, 1, 1))
	rgba.SetRGBA(0, 0, color.RGBA{R: 10, G: 20, B: 30, A: 255})
	m := FromImage(rgba)
	if m.Channels != 3 {
		t.Fatalf("want 3 channels, got %d", m.Channels)
	}
	if m.At(0, 0, 0) != 10 || m.At(0, 0, 1) != 20 || m.At(0, 0, 2) != 30 {
		t.Errorf("RGBA import mismatch: %v", m.AtPixel(0, 0))
	}
}

func TestCopyTo(t *testing.T) {
	dst := NewMat(4, 4, 1)
	patch := NewMat(2, 2, 1)
	patch.SetTo(99)
	patch.CopyTo(dst, 1, 1)
	if dst.At(1, 1, 0) != 99 || dst.At(2, 2, 0) != 99 {
		t.Error("CopyTo did not place patch")
	}
	if dst.At(0, 0, 0) != 0 {
		t.Error("CopyTo touched outside region")
	}
}
