package cudaarithm

import (
	"testing"

	cv "github.com/malcolmston/opencv"
)

func TestGpuMatUploadDownloadCopies(t *testing.T) {
	src := constMat(3, 4, 7)
	g := NewGpuMat(src)

	// Mutating src after upload must not change the GpuMat (copy semantics).
	src.Data[0] = 99
	got := g.Download()
	if got.Data[0] != 7 {
		t.Fatalf("Upload did not copy: got %d, want 7", got.Data[0])
	}
	// Mutating the download must not change the GpuMat either.
	got.Data[1] = 42
	again := g.Download()
	if again.Data[1] != 7 {
		t.Fatalf("Download did not copy: got %d, want 7", again.Data[1])
	}
}

func TestGpuMatMetadata(t *testing.T) {
	g := NewGpuMat(cv.NewMat(5, 8, 3))
	if r, c := g.Size(); r != 5 || c != 8 {
		t.Errorf("Size = (%d,%d), want (5,8)", r, c)
	}
	if g.Channels() != 3 {
		t.Errorf("Channels = %d, want 3", g.Channels())
	}
	if g.Type() != CV_8UC3 {
		t.Errorf("Type = %v, want CV_8UC3", g.Type())
	}
	if g.Type().String() != "CV_8UC3" {
		t.Errorf("Type.String = %q", g.Type().String())
	}
	if g.Empty() {
		t.Error("non-empty GpuMat reported empty")
	}
}

func TestGpuMatEmptyStates(t *testing.T) {
	var nilMat *GpuMat
	if !nilMat.Empty() {
		t.Error("nil GpuMat should be empty")
	}
	empty := NewGpuMat(nil)
	if !empty.Empty() {
		t.Error("NewGpuMat(nil) should be empty")
	}
	if empty.Download() != nil {
		t.Error("empty Download should be nil")
	}
	if r, c := empty.Size(); r != 0 || c != 0 {
		t.Errorf("empty Size = (%d,%d), want (0,0)", r, c)
	}
	if empty.Channels() != 0 {
		t.Error("empty Channels should be 0")
	}
	if empty.Type() != CV_8UC1 {
		t.Error("empty Type should be CV_8UC1")
	}
	if !empty.Clone().Empty() {
		t.Error("clone of empty should be empty")
	}
}

func TestGpuMatCloneAndRelease(t *testing.T) {
	g := NewGpuMat(constMat(2, 2, 5))
	clone := g.Clone()
	clone.Mat().Data[0] = 100
	if g.Mat().Data[0] != 5 {
		t.Error("Clone shares storage with original")
	}
	g.Release()
	if !g.Empty() {
		t.Error("Release should leave GpuMat empty")
	}
}

func TestUploadEmptyClears(t *testing.T) {
	g := NewGpuMat(constMat(2, 2, 1))
	g.Upload(nil)
	if !g.Empty() {
		t.Error("Upload(nil) should clear the GpuMat")
	}
}

func TestMakeType(t *testing.T) {
	if MakeType(1) != CV_8UC1 || MakeType(2) != CV_8UC2 || MakeType(3) != CV_8UC3 || MakeType(4) != CV_8UC4 {
		t.Error("MakeType does not match CV_8UC constants")
	}
	if CV_8UC4.Channels() != 4 {
		t.Errorf("CV_8UC4.Channels = %d, want 4", CV_8UC4.Channels())
	}
}

func TestMakeTypePanics(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("expected panic on non-positive channels")
		}
	}()
	MakeType(0)
}
