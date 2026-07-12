package cudawarping_test

import (
	"testing"

	cv "github.com/malcolmston/opencv"
	"github.com/malcolmston/opencv/cudawarping"
)

// gradient builds a deterministic rows×cols×channels image whose samples vary
// with position and channel, so resampling errors are easy to detect.
func gradient(rows, cols, channels int) *cv.Mat {
	m := cv.NewMat(rows, cols, channels)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			for c := 0; c < channels; c++ {
				m.Set(y, x, c, uint8((x*7+y*13+c*29)%256))
			}
		}
	}
	return m
}

// equalMat reports whether two Mats have identical shape and samples.
func equalMat(a, b *cv.Mat) bool {
	if a.Rows != b.Rows || a.Cols != b.Cols || a.Channels != b.Channels {
		return false
	}
	for i := range a.Data {
		if a.Data[i] != b.Data[i] {
			return false
		}
	}
	return true
}

func TestUploadDownloadRoundTrip(t *testing.T) {
	src := gradient(6, 8, 3)
	g := cudawarping.Upload(src)
	if g.Empty() {
		t.Fatal("uploaded GpuMat should not be empty")
	}
	if r, c := g.Size(); r != 6 || c != 8 {
		t.Fatalf("Size = (%d,%d), want (6,8)", r, c)
	}
	if ch := g.Channels(); ch != 3 {
		t.Fatalf("Channels = %d, want 3", ch)
	}
	out := g.Download()
	if !equalMat(src, out) {
		t.Fatal("download did not reproduce the uploaded image")
	}
	// Download must be an independent copy.
	out.Data[0] ^= 0xFF
	if equalMat(src, g.Download()) == false {
		t.Fatal("mutating the download must not affect the GpuMat")
	}
}

func TestUploadIsDeepCopy(t *testing.T) {
	src := gradient(4, 4, 1)
	g := cudawarping.Upload(src)
	src.Set(0, 0, 0, src.At(0, 0, 0)^0xFF)
	if g.Download().At(0, 0, 0) == src.At(0, 0, 0) {
		t.Fatal("Upload should deep-copy the source")
	}
}

func TestNewGpuMatAndRelease(t *testing.T) {
	g := cudawarping.NewGpuMat(3, 5, 2)
	if g.Empty() {
		t.Fatal("new GpuMat should not be empty")
	}
	g.Release()
	if !g.Empty() {
		t.Fatal("released GpuMat should be empty")
	}
	if r, c := g.Size(); r != 0 || c != 0 {
		t.Fatalf("released Size = (%d,%d), want (0,0)", r, c)
	}
	if g.Channels() != 0 {
		t.Fatal("released Channels should be 0")
	}
}

func TestCloneIsIndependent(t *testing.T) {
	g := cudawarping.Upload(gradient(4, 4, 1))
	clone := g.Clone()
	orig := g.Download()
	m := clone.Download()
	m.Set(0, 0, 0, m.At(0, 0, 0)^0xFF)
	if !equalMat(orig, g.Download()) {
		t.Fatal("clone must not share storage with the original")
	}
}

func TestStreamNoOps(t *testing.T) {
	s := cudawarping.NewStream()
	s.WaitForCompletion()
	if !s.QueryIfComplete() {
		t.Fatal("CPU stream should always report complete")
	}
	var nilStream *cudawarping.Stream
	// A nil stream must be usable everywhere.
	g := cudawarping.Upload(gradient(4, 4, 1))
	if g.Transpose(nilStream).Empty() {
		t.Fatal("operation with a nil stream should succeed")
	}
}

func TestUploadEmptyPanics(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("Upload(nil) should panic")
		}
	}()
	cudawarping.Upload(nil)
}

func TestDownloadEmptyPanics(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("Download on empty GpuMat should panic")
		}
	}()
	var g cudawarping.GpuMat
	g.Download()
}

func TestEmptyGpuMatOpPanics(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("Transpose on empty GpuMat should panic")
		}
	}()
	var g cudawarping.GpuMat
	g.Transpose(nil)
}
