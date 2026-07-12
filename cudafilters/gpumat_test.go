package cudafilters

import (
	"image"
	"testing"

	cv "github.com/malcolmston/opencv"
)

func TestGpuMatUploadDownloadDeepCopy(t *testing.T) {
	src := cv.NewMat(4, 4, 1)
	src.Set(0, 0, 0, 100)
	g := NewGpuMat()
	if !g.Empty() {
		t.Fatal("fresh GpuMat should be empty")
	}
	g.Upload(src)
	if g.Empty() {
		t.Fatal("GpuMat should be non-empty after Upload")
	}
	// Mutating the source must not change the uploaded copy.
	src.Set(0, 0, 0, 7)
	out := g.Download()
	if out.At(0, 0, 0) != 100 {
		t.Fatalf("Upload was not a deep copy: got %d want 100", out.At(0, 0, 0))
	}
	// Mutating the downloaded copy must not change the GpuMat.
	out.Set(0, 0, 0, 42)
	out2 := g.Download()
	if out2.At(0, 0, 0) != 100 {
		t.Fatalf("Download was not a deep copy: got %d want 100", out2.At(0, 0, 0))
	}
}

func TestGpuMatFromMat(t *testing.T) {
	src := cv.NewMat(3, 5, 3)
	g := GpuMatFromMat(src)
	r, c := g.Size()
	if r != 3 || c != 5 {
		t.Fatalf("Size = (%d,%d) want (3,5)", r, c)
	}
	if g.Channels() != 3 {
		t.Fatalf("Channels = %d want 3", g.Channels())
	}
}

func TestGpuMatEmptyBehaviour(t *testing.T) {
	var g *GpuMat
	if !g.Empty() {
		t.Fatal("nil GpuMat should report empty")
	}
	empty := NewGpuMat()
	if empty.Download() != nil {
		t.Fatal("empty Download should be nil")
	}
	if r, c := empty.Size(); r != 0 || c != 0 {
		t.Fatalf("empty Size = (%d,%d) want (0,0)", r, c)
	}
	if empty.Channels() != 0 {
		t.Fatal("empty Channels should be 0")
	}
	// Uploading nil/empty keeps it empty.
	empty.Upload(nil)
	if !empty.Empty() {
		t.Fatal("Upload(nil) should keep GpuMat empty")
	}
}

func TestGpuMatCloneAndRelease(t *testing.T) {
	src := cv.NewMat(3, 3, 1)
	src.Set(1, 1, 0, 200)
	g := GpuMatFromMat(src)
	clone := g.Clone()
	g.Release()
	if !g.Empty() {
		t.Fatal("Released GpuMat should be empty")
	}
	if clone.Empty() {
		t.Fatal("clone should survive release of the original")
	}
	if clone.Download().At(1, 1, 0) != 200 {
		t.Fatal("clone lost its data")
	}
	// Clone of an empty GpuMat is empty, not nil.
	ec := NewGpuMat().Clone()
	if !ec.Empty() {
		t.Fatal("clone of empty should be empty")
	}
}

func TestApplyPanicsOnEmpty(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("Apply on empty GpuMat should panic")
		}
	}()
	CreateBoxFilter(image.Pt(3, 3), AnchorCenter, BorderDefault).Apply(NewGpuMat())
}

func TestFactoryValidation(t *testing.T) {
	mustPanic := func(name string, fn func()) {
		defer func() {
			if recover() == nil {
				t.Fatalf("%s should panic", name)
			}
		}()
		fn()
	}
	mustPanic("even ksize box", func() { CreateBoxFilter(image.Pt(4, 4), AnchorCenter, BorderDefault) })
	mustPanic("non-square box", func() { CreateBoxFilter(image.Pt(3, 5), AnchorCenter, BorderDefault) })
	mustPanic("off-center anchor", func() { CreateBoxFilter(image.Pt(3, 3), image.Pt(0, 0), BorderDefault) })
	mustPanic("even gaussian", func() { CreateGaussianFilter(image.Pt(4, 3), 1, 1, BorderDefault) })
	mustPanic("empty sep kernel", func() { CreateSeparableLinearFilter(nil, []float64{1}, 0, AnchorCenter, BorderDefault) })
	mustPanic("even median", func() { CreateMedianFilter(4) })
	mustPanic("even row sum", func() { CreateRowSumFilter(2) })
	mustPanic("even col sum", func() { CreateColumnSumFilter(2) })
	mustPanic("nil morph kernel", func() { CreateMorphologyFilter(MorphErode, nil, AnchorCenter, 1) })
	mustPanic("bad morph op", func() { MorphOp(99).toCVMorphType() })
}

func TestStreamNoop(t *testing.T) {
	s := NewStream()
	s.WaitForCompletion()
	s.Release()
}
