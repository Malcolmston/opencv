package cudawarping_test

import (
	"image"
	"testing"

	cv "github.com/malcolmston/opencv"
	"github.com/malcolmston/opencv/cudawarping"
)

func TestInstanceUploadReplacesContents(t *testing.T) {
	g := cudawarping.NewGpuMat(2, 2, 1)
	src := gradient(3, 4, 2)
	g.Upload(src)
	if r, c := g.Size(); r != 3 || c != 4 {
		t.Fatalf("after Upload size = (%d,%d), want (3,4)", r, c)
	}
	if !equalMat(src, g.Download()) {
		t.Fatal("instance Upload did not replace contents")
	}
}

func TestInstanceUploadEmptyPanics(t *testing.T) {
	g := cudawarping.NewGpuMat(2, 2, 1)
	defer func() {
		if recover() == nil {
			t.Fatal("instance Upload(nil) should panic")
		}
	}()
	g.Upload(nil)
}

func TestStreamWaitForCompletion(t *testing.T) {
	s := cudawarping.NewStream()
	s.WaitForCompletion() // must not block or panic
	if !s.QueryIfComplete() {
		t.Fatal("stream should be complete")
	}
}

func TestCloneEmpty(t *testing.T) {
	var g cudawarping.GpuMat
	if !g.Clone().Empty() {
		t.Fatal("clone of empty GpuMat should be empty")
	}
}

// TestCopyMakeBorderReflectModes checks the exact reflect / reflect-101 / wrap
// index mappings, exercising every branch of the border interpolation.
func TestCopyMakeBorderReflectModes(t *testing.T) {
	src := cv.NewMat(1, 4, 1) // one row: 10 20 30 40
	src.Set(0, 0, 0, 10)
	src.Set(0, 1, 0, 20)
	src.Set(0, 2, 0, 30)
	src.Set(0, 3, 0, 40)
	g := cudawarping.Upload(src)

	cases := []struct {
		mode cudawarping.BorderMode
		want [3]uint8 // the three left-border columns
	}{
		{cudawarping.BorderReflect101, [3]uint8{40, 30, 20}}, // gfedcb|abcd
		{cudawarping.BorderReflect, [3]uint8{30, 20, 10}},    // cba|abcd
		{cudawarping.BorderWrap, [3]uint8{20, 30, 40}},       // bcd|abcd
	}
	for _, tc := range cases {
		out := g.CopyMakeBorder(0, 0, 3, 0, tc.mode, 0, nil).Download()
		got := [3]uint8{out.At(0, 0, 0), out.At(0, 1, 0), out.At(0, 2, 0)}
		if got != tc.want {
			t.Fatalf("border mode %d left cols = %v, want %v", tc.mode, got, tc.want)
		}
	}
}

// TestBorderReflectSingleRow exercises the length==1 short-circuit in the
// reflect branches.
func TestBorderReflectSingleRow(t *testing.T) {
	src := cv.NewMat(1, 1, 1)
	src.SetTo(88)
	g := cudawarping.Upload(src)
	out := g.CopyMakeBorder(2, 2, 2, 2, cudawarping.BorderReflect101, 0, nil).Download()
	// Every pixel must be the single source sample.
	for y := 0; y < out.Rows; y++ {
		for x := 0; x < out.Cols; x++ {
			if out.At(y, x, 0) != 88 {
				t.Fatalf("reflect of 1x1 at (%d,%d) = %d, want 88", y, x, out.At(y, x, 0))
			}
		}
	}
}

func TestWarpPerspectiveIdentityNonDefaultBorder(t *testing.T) {
	src := gradient(6, 6, 3)
	g := cudawarping.Upload(src)
	id := cv.PerspectiveMatrix{1, 0, 0, 0, 1, 0, 0, 0, 1}
	// Cubic + replicate forces the local map-based path (invertPerspective +
	// BuildWarpPerspectiveMaps with inverse=false). Identity leaves src unchanged.
	out := g.WarpPerspective(id, image.Point{X: 6, Y: 6}, int(cudawarping.InterCubic), cudawarping.BorderReplicate, 0, nil).Download()
	if !equalMat(src, out) {
		t.Fatal("identity perspective warp (non-default border) changed the image")
	}
}

func TestWarpPerspectiveInverseFlagNonDefault(t *testing.T) {
	src := gradient(6, 6, 3)
	g := cudawarping.Upload(src)
	id := cv.PerspectiveMatrix{1, 0, 0, 0, 1, 0, 0, 0, 1}
	flags := int(cudawarping.InterCubic) | cudawarping.WarpInverseMap
	out := g.WarpPerspective(id, image.Point{X: 6, Y: 6}, flags, cudawarping.BorderReplicate, 0, nil).Download()
	if !equalMat(src, out) {
		t.Fatal("inverse identity perspective warp changed the image")
	}
}

func TestResizeAreaEnlarge(t *testing.T) {
	src := gradient(2, 2, 1)
	g := cudawarping.Upload(src)
	// Enlarging with area collapses to nearest-neighbour; just verify size and
	// that corner samples come from the source.
	out := g.Resize(image.Point{X: 4, Y: 4}, 0, 0, cudawarping.InterArea, nil).Download()
	if out.Rows != 4 || out.Cols != 4 {
		t.Fatalf("area enlarge size = (%d,%d), want (4,4)", out.Rows, out.Cols)
	}
	if out.At(0, 0, 0) != src.At(0, 0, 0) {
		t.Fatalf("area enlarge corner = %d, want %d", out.At(0, 0, 0), src.At(0, 0, 0))
	}
}

func TestResizeScalePanics(t *testing.T) {
	g := cudawarping.Upload(gradient(3, 3, 1))
	defer func() {
		if recover() == nil {
			t.Fatal("ResizeScale with non-positive factor should panic")
		}
	}()
	g.ResizeScale(0, 2, cudawarping.InterLinear, nil)
}

func TestWarpAffineZeroDsizePanics(t *testing.T) {
	g := cudawarping.Upload(gradient(3, 3, 1))
	defer func() {
		if recover() == nil {
			t.Fatal("WarpAffine with zero dsize should panic")
		}
	}()
	g.WarpAffine(cv.AffineMatrix{1, 0, 0, 0, 1, 0}, image.Point{}, int(cudawarping.InterLinear), cudawarping.BorderConstant, 0, nil)
}

func TestBuildWarpPerspectiveMapsInverse(t *testing.T) {
	// inverse=true uses the matrix directly; identity maps must be the grid.
	id := cv.PerspectiveMatrix{1, 0, 0, 0, 1, 0, 0, 0, 1}
	xmap, ymap := cudawarping.BuildWarpPerspectiveMaps(id, true, image.Point{X: 3, Y: 2}, nil)
	if xmap.At(1, 2) != 2 || ymap.At(1, 2) != 1 {
		t.Fatalf("identity perspective maps at (1,2) = (%v,%v), want (2,1)", xmap.At(1, 2), ymap.At(1, 2))
	}
}

func TestNonInvertibleAffinePanics(t *testing.T) {
	g := cudawarping.Upload(gradient(4, 4, 1))
	defer func() {
		if recover() == nil {
			t.Fatal("non-invertible affine should panic on the local path")
		}
	}()
	// Degenerate (zero linear part) matrix with a non-default border to force
	// the map-building path.
	g.WarpAffine(cv.AffineMatrix{0, 0, 0, 0, 0, 0}, image.Point{X: 4, Y: 4}, int(cudawarping.InterCubic), cudawarping.BorderReplicate, 0, nil)
}
