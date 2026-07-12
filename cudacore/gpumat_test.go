package cudacore_test

import (
	"reflect"
	"testing"

	cv "github.com/malcolmston/opencv"
	"github.com/malcolmston/opencv/cudacore"
)

// makeMat builds a cv.Mat from explicit sample data.
func makeMat(rows, cols, ch int, data []uint8) *cv.Mat {
	m := cv.NewMat(rows, cols, ch)
	copy(m.Data, data)
	return m
}

func TestUploadDownloadRoundTrip(t *testing.T) {
	src := makeMat(2, 3, 1, []uint8{1, 2, 3, 4, 5, 6})
	g := cudacore.NewGpuMat(src)

	got := g.Download()
	if !reflect.DeepEqual(got.Data, src.Data) {
		t.Fatalf("round-trip data = %v, want %v", got.Data, src.Data)
	}
	// Download must not alias, and mutating src must not change the GpuMat.
	src.Data[0] = 99
	if g.Download().Data[0] != 1 {
		t.Fatalf("GpuMat aliased its source after upload")
	}
	got.Data[1] = 88
	if g.Download().Data[1] != 2 {
		t.Fatalf("Download aliased the device buffer")
	}
}

func TestUploadEmptyClears(t *testing.T) {
	g := cudacore.NewGpuMat(makeMat(1, 1, 1, []uint8{7}))
	g.Upload(nil)
	if !g.Empty() {
		t.Fatalf("Upload(nil) should empty the GpuMat")
	}
	if g.Download() != nil {
		t.Fatalf("Download of empty GpuMat should be nil")
	}
}

func TestCreateAndAccessors(t *testing.T) {
	g := &cudacore.GpuMat{}
	if !g.Empty() {
		t.Fatalf("zero GpuMat should be empty")
	}
	g.Create(4, 5, cudacore.CV_8UC3)
	r, c := g.Size()
	if r != 4 || c != 5 {
		t.Fatalf("Size = (%d,%d), want (4,5)", r, c)
	}
	if g.Channels() != 3 {
		t.Fatalf("Channels = %d, want 3", g.Channels())
	}
	if g.Type() != cudacore.CV_8UC3 {
		t.Fatalf("Type = %v, want CV_8UC3", g.Type())
	}
	if !g.IsContinuous() {
		t.Fatalf("fresh GpuMat should be continuous")
	}
}

func TestReleaseAndClone(t *testing.T) {
	g := cudacore.NewGpuMat(makeMat(1, 2, 1, []uint8{3, 4}))
	clone := g.Clone()
	g.Release()
	if !g.Empty() {
		t.Fatalf("Release should empty the GpuMat")
	}
	if clone.Empty() || clone.Download().Data[0] != 3 {
		t.Fatalf("Clone should be independent of the released original")
	}
}

func TestCopyTo(t *testing.T) {
	src := makeMat(2, 2, 1, []uint8{10, 20, 30, 40})
	g := cudacore.NewGpuMat(src)
	var dst cudacore.GpuMat
	g.CopyTo(&dst)
	if !reflect.DeepEqual(dst.Download().Data, src.Data) {
		t.Fatalf("CopyTo data = %v, want %v", dst.Download().Data, src.Data)
	}
	// Independence from the receiver.
	g.SetTo(cv.NewScalar(0))
	if dst.Download().Data[0] != 10 {
		t.Fatalf("CopyTo destination aliased the source")
	}
}

func TestSetToMatchesRoot(t *testing.T) {
	g := cudacore.NewGpuMat(makeMat(2, 2, 1, []uint8{0, 0, 0, 0}))
	g.SetTo(cv.NewScalar(7))

	want := cv.NewMat(2, 2, 1)
	want.SetTo(7) // root cv.Mat.SetTo fills a single-channel matrix uniformly
	if !reflect.DeepEqual(g.Download().Data, want.Data) {
		t.Fatalf("SetTo = %v, want %v", g.Download().Data, want.Data)
	}
}

func TestSetToPerChannelAndRounding(t *testing.T) {
	g := cudacore.NewGpuMat(cv.NewMat(1, 2, 3))
	g.SetTo(cv.NewScalar(1.4, 2.6, 300)) // round 1.4->1, 2.6->3, saturate 300->255
	want := []uint8{1, 3, 255, 1, 3, 255}
	if !reflect.DeepEqual(g.Download().Data, want) {
		t.Fatalf("SetTo per-channel = %v, want %v", g.Download().Data, want)
	}
}

func TestConvertToMatchesRootScaleAbs(t *testing.T) {
	src := makeMat(1, 4, 1, []uint8{0, 10, 100, 200})
	g := cudacore.NewGpuMat(src)
	// Non-negative alpha/beta keep results non-negative, so convertTo agrees
	// with the root package's convertScaleAbs.
	got := g.ConvertTo(1.5, 2).Download()
	want := cv.ConvertScaleAbs(src, 1.5, 2)
	if !reflect.DeepEqual(got.Data, want.Data) {
		t.Fatalf("ConvertTo = %v, want %v", got.Data, want.Data)
	}
}

func TestConvertToNegativeClampsToZero(t *testing.T) {
	src := makeMat(1, 2, 1, []uint8{10, 200})
	g := cudacore.NewGpuMat(src)
	got := g.ConvertTo(-1, 0).Download() // negative results clamp to 0 (not abs)
	if got.Data[0] != 0 || got.Data[1] != 0 {
		t.Fatalf("ConvertTo negative = %v, want [0 0]", got.Data)
	}
}

func TestReshape(t *testing.T) {
	g := cudacore.NewGpuMat(makeMat(2, 3, 1, []uint8{1, 2, 3, 4, 5, 6}))
	// 6 samples, 1 channel -> reshape to 3 channels, 1 row => 1x2x3.
	r := g.Reshape(3, 1)
	rows, cols := r.Size()
	if rows != 1 || cols != 2 || r.Channels() != 3 {
		t.Fatalf("Reshape shape = %dx%dx%d, want 1x2x3", rows, cols, r.Channels())
	}
	if !reflect.DeepEqual(r.Download().Data, []uint8{1, 2, 3, 4, 5, 6}) {
		t.Fatalf("Reshape altered data: %v", r.Download().Data)
	}
	// keep channels (0), change rows.
	r2 := g.Reshape(0, 3)
	rows2, cols2 := r2.Size()
	if rows2 != 3 || cols2 != 2 {
		t.Fatalf("Reshape(0,3) = %dx%d, want 3x2", rows2, cols2)
	}
}

func TestReshapeInvalid(t *testing.T) {
	g := cudacore.NewGpuMat(makeMat(1, 3, 1, []uint8{1, 2, 3}))
	assertPanics(t, "Reshape non-divisible channels", func() { g.Reshape(2, 0) })
	assertPanics(t, "Reshape bad rows", func() { g.Reshape(1, 2) })
}

func TestRowColRanges(t *testing.T) {
	g := cudacore.NewGpuMat(makeMat(3, 3, 1, []uint8{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
	}))
	if !reflect.DeepEqual(g.Row(1).Download().Data, []uint8{4, 5, 6}) {
		t.Fatalf("Row(1) = %v", g.Row(1).Download().Data)
	}
	if !reflect.DeepEqual(g.Col(2).Download().Data, []uint8{3, 6, 9}) {
		t.Fatalf("Col(2) = %v", g.Col(2).Download().Data)
	}
	if !reflect.DeepEqual(g.RowRange(0, 2).Download().Data, []uint8{1, 2, 3, 4, 5, 6}) {
		t.Fatalf("RowRange = %v", g.RowRange(0, 2).Download().Data)
	}
	if !reflect.DeepEqual(g.ColRange(1, 3).Download().Data, []uint8{2, 3, 5, 6, 8, 9}) {
		t.Fatalf("ColRange = %v", g.ColRange(1, 3).Download().Data)
	}
	assertPanics(t, "RowRange oob", func() { g.RowRange(0, 5) })
	assertPanics(t, "ColRange oob", func() { g.ColRange(-1, 2) })
}

func TestSwapChannels(t *testing.T) {
	// two RGB pixels
	g := cudacore.NewGpuMat(makeMat(1, 2, 3, []uint8{10, 20, 30, 40, 50, 60}))
	g.SwapChannels([]int{2, 1, 0}) // RGB -> BGR
	want := []uint8{30, 20, 10, 60, 50, 40}
	if !reflect.DeepEqual(g.Download().Data, want) {
		t.Fatalf("SwapChannels = %v, want %v", g.Download().Data, want)
	}
	assertPanics(t, "SwapChannels wrong len", func() { g.SwapChannels([]int{0, 1}) })
	assertPanics(t, "SwapChannels bad index", func() { g.SwapChannels([]int{0, 1, 9}) })
}

func TestCopyMakeBorderConstant(t *testing.T) {
	g := cudacore.NewGpuMat(makeMat(1, 1, 1, []uint8{5}))
	out := g.CopyMakeBorder(1, 1, 2, 2, cudacore.BorderConstant, cv.NewScalar(9))
	rows, cols := out.Size()
	if rows != 3 || cols != 5 {
		t.Fatalf("border size = %dx%d, want 3x5", rows, cols)
	}
	want := []uint8{
		9, 9, 9, 9, 9,
		9, 9, 5, 9, 9,
		9, 9, 9, 9, 9,
	}
	if !reflect.DeepEqual(out.Download().Data, want) {
		t.Fatalf("CopyMakeBorder constant = %v, want %v", out.Download().Data, want)
	}
}

func TestCopyMakeBorderReplicateReflect(t *testing.T) {
	g := cudacore.NewGpuMat(makeMat(1, 3, 1, []uint8{1, 2, 3}))

	rep := g.CopyMakeBorder(0, 0, 2, 2, cudacore.BorderReplicate, cv.Scalar{})
	if !reflect.DeepEqual(rep.Download().Data, []uint8{1, 1, 1, 2, 3, 3, 3}) {
		t.Fatalf("replicate = %v", rep.Download().Data)
	}
	ref101 := g.CopyMakeBorder(0, 0, 2, 2, cudacore.BorderReflect101, cv.Scalar{})
	if !reflect.DeepEqual(ref101.Download().Data, []uint8{3, 2, 1, 2, 3, 2, 1}) {
		t.Fatalf("reflect101 = %v", ref101.Download().Data)
	}
	ref := g.CopyMakeBorder(0, 0, 2, 2, cudacore.BorderReflect, cv.Scalar{})
	if !reflect.DeepEqual(ref.Download().Data, []uint8{2, 1, 1, 2, 3, 3, 2}) {
		t.Fatalf("reflect = %v", ref.Download().Data)
	}
	wrap := g.CopyMakeBorder(0, 0, 2, 2, cudacore.BorderWrap, cv.Scalar{})
	if !reflect.DeepEqual(wrap.Download().Data, []uint8{2, 3, 1, 2, 3, 1, 2}) {
		t.Fatalf("wrap = %v", wrap.Download().Data)
	}
}

func TestCopyMakeBorderInteriorMatchesSource(t *testing.T) {
	src := makeMat(2, 2, 1, []uint8{1, 2, 3, 4})
	g := cudacore.NewGpuMat(src)
	out := g.CopyMakeBorder(1, 1, 1, 1, cudacore.BorderConstant, cv.NewScalar(0)).Download()
	// interior 2x2 must equal the source
	interior := out.Region(1, 1, 2, 2)
	if !reflect.DeepEqual(interior.Data, src.Data) {
		t.Fatalf("interior = %v, want %v", interior.Data, src.Data)
	}
}

func TestEnsureSizeIsEnough(t *testing.T) {
	var g cudacore.GpuMat
	cudacore.EnsureSizeIsEnough(4, 4, cudacore.CV_8UC1, &g)
	first := g.Mat()
	// Already big enough: same layout must be reused (no reallocation).
	cudacore.EnsureSizeIsEnough(4, 4, cudacore.CV_8UC1, &g)
	if g.Mat() != first {
		t.Fatalf("EnsureSizeIsEnough reallocated an already-sufficient buffer")
	}
	// Different channels: must reallocate.
	cudacore.EnsureSizeIsEnough(4, 4, cudacore.CV_8UC3, &g)
	if g.Channels() != 3 {
		t.Fatalf("EnsureSizeIsEnough did not grow channels")
	}
}

func TestPtrStepSz(t *testing.T) {
	g := cudacore.NewGpuMat(makeMat(2, 3, 2, make([]uint8, 12)))
	p := g.PtrStepSz()
	if p.Rows != 2 || p.Cols != 3 || p.Channels != 2 || p.Step != 6 {
		t.Fatalf("PtrStepSz = %+v", p)
	}
	if len(p.Data) != 12 {
		t.Fatalf("PtrStepSz data len = %d, want 12", len(p.Data))
	}
}

func TestMatType(t *testing.T) {
	if cudacore.MakeType(4) != cudacore.CV_8UC4 {
		t.Fatalf("MakeType(4) != CV_8UC4")
	}
	if cudacore.CV_8UC3.Channels() != 3 {
		t.Fatalf("CV_8UC3.Channels != 3")
	}
	if cudacore.CV_8UC2.String() != "CV_8UC2" {
		t.Fatalf("String = %q", cudacore.CV_8UC2.String())
	}
	assertPanics(t, "MakeType(0)", func() { cudacore.MakeType(0) })
}

func TestEmptyGpuMatPanics(t *testing.T) {
	var g cudacore.GpuMat
	assertPanics(t, "SetTo empty", func() { g.SetTo(cv.NewScalar(1)) })
	assertPanics(t, "ConvertTo empty", func() { g.ConvertTo(1, 0) })
	assertPanics(t, "Reshape empty", func() { g.Reshape(1, 1) })
	assertPanics(t, "PtrStepSz empty", func() { g.PtrStepSz() })
	assertPanics(t, "CopyMakeBorder empty", func() {
		g.CopyMakeBorder(1, 1, 1, 1, cudacore.BorderConstant, cv.Scalar{})
	})
}

// assertPanics fails the test if fn does not panic.
func assertPanics(t *testing.T, name string, fn func()) {
	t.Helper()
	defer func() {
		if recover() == nil {
			t.Fatalf("%s: expected panic, got none", name)
		}
	}()
	fn()
}
