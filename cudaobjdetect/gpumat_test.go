package cudaobjdetect

import (
	"testing"

	cv "github.com/malcolmston/opencv"
)

// TestGpuMatUploadDownload checks the host round-trip and copy semantics.
func TestGpuMatUploadDownload(t *testing.T) {
	src := cv.NewMat(4, 5, 1)
	src.Set(1, 2, 0, 77)

	var g GpuMat
	g.Upload(src)
	if r, c := g.Size(); r != 4 || c != 5 {
		t.Fatalf("Size() = %dx%d, want 4x5", r, c)
	}
	if g.Channels() != 1 {
		t.Fatalf("Channels() = %d, want 1", g.Channels())
	}
	if g.Empty() {
		t.Fatal("Empty() = true after Upload")
	}

	// Upload must take an independent copy.
	src.Set(1, 2, 0, 200)
	down := g.Download()
	if down.At(1, 2, 0) != 77 {
		t.Fatalf("Download sample = %d, want 77 (independent copy expected)", down.At(1, 2, 0))
	}
}

// TestGpuMatConstructors checks the allocating and wrapping constructors.
func TestGpuMatConstructors(t *testing.T) {
	g := NewGpuMat(3, 3, 3)
	if g.Channels() != 3 {
		t.Fatalf("Channels() = %d, want 3", g.Channels())
	}
	m := cv.NewMat(2, 2, 1)
	w := NewGpuMatFromMat(m)
	if w.Mat() != m {
		t.Fatal("NewGpuMatFromMat should share the Mat without copying")
	}
	clone := w.Clone()
	if clone.Mat() == m {
		t.Fatal("Clone should have independent storage")
	}
	if r, c := clone.Size(); r != 2 || c != 2 {
		t.Fatalf("clone Size() = %dx%d", r, c)
	}
}

// TestGpuMatEmptyAndNoImage checks zero-value and detection-only mats.
func TestGpuMatEmptyAndNoImage(t *testing.T) {
	var g GpuMat
	if !g.Empty() {
		t.Fatal("zero-value GpuMat should be Empty")
	}
	if r, c := g.Size(); r != 0 || c != 0 {
		t.Fatalf("empty Size() = %dx%d, want 0x0", r, c)
	}
	if g.Channels() != 0 {
		t.Fatalf("empty Channels() = %d, want 0", g.Channels())
	}
	if g.Mat() != nil {
		t.Fatal("empty Mat() should be nil")
	}
}

// TestGpuMatPanics checks the nil-argument and no-image panics.
func TestGpuMatPanics(t *testing.T) {
	mustPanic := func(name string, fn func()) {
		defer func() {
			if recover() == nil {
				t.Fatalf("%s: expected panic", name)
			}
		}()
		fn()
	}
	mustPanic("NewGpuMatFromMat nil", func() { NewGpuMatFromMat(nil) })
	mustPanic("Upload nil", func() { (&GpuMat{}).Upload(nil) })
	mustPanic("Download no image", func() { (&GpuMat{}).Download() })
}

// TestStreamNoOp checks the no-op stream methods.
func TestStreamNoOp(t *testing.T) {
	s := NewStream()
	s.WaitForCompletion() // must not panic
	if !s.QueryIfComplete() {
		t.Fatal("QueryIfComplete() should always be true")
	}
}
