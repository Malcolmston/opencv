package cudawarping_test

import (
	"image"
	"testing"

	cv "github.com/malcolmston/opencv"
	"github.com/malcolmston/opencv/cudawarping"
)

func TestResizeIdentity(t *testing.T) {
	src := gradient(6, 8, 3)
	g := cudawarping.Upload(src)
	for _, interp := range []cudawarping.Interpolation{
		cudawarping.InterNearest,
		cudawarping.InterLinear,
		cudawarping.InterCubic,
		cudawarping.InterArea,
	} {
		out := g.Resize(image.Point{X: 8, Y: 6}, 0, 0, interp, nil).Download()
		if !equalMat(src, out) {
			t.Fatalf("identity resize with interp %d changed the image", interp)
		}
	}
}

func TestResizeMatchesCV(t *testing.T) {
	src := gradient(5, 7, 3)
	g := cudawarping.Upload(src)
	cases := []struct {
		interp cudawarping.Interpolation
		cv     cv.InterpolationFlag
	}{
		{cudawarping.InterNearest, cv.InterNearest},
		{cudawarping.InterLinear, cv.InterLinear},
	}
	for _, tc := range cases {
		got := g.Resize(image.Point{X: 14, Y: 10}, 0, 0, tc.interp, nil).Download()
		want := cv.Resize(src, 14, 10, tc.cv)
		if !equalMat(got, want) {
			t.Fatalf("Resize interp %d does not match cv.Resize", tc.interp)
		}
	}
}

func TestResizeScale(t *testing.T) {
	src := gradient(4, 6, 1)
	g := cudawarping.Upload(src)
	out := g.ResizeScale(2, 3, cudawarping.InterNearest, nil)
	if r, c := out.Size(); r != 12 || c != 12 {
		t.Fatalf("ResizeScale size = (%d,%d), want (12,12)", r, c)
	}
}

func TestResizeAreaShrinkAverages(t *testing.T) {
	// A 2x2 image downscaled to 1x1 with area interpolation must be the mean.
	src := cv.NewMat(2, 2, 1)
	src.Set(0, 0, 0, 0)
	src.Set(0, 1, 0, 100)
	src.Set(1, 0, 0, 100)
	src.Set(1, 1, 0, 200)
	g := cudawarping.Upload(src)
	out := g.Resize(image.Point{X: 1, Y: 1}, 0, 0, cudawarping.InterArea, nil).Download()
	if got := out.At(0, 0, 0); got != 100 { // (0+100+100+200)/4 = 100
		t.Fatalf("area shrink = %d, want 100", got)
	}
}

func TestResizeCubicIsBounded(t *testing.T) {
	src := gradient(5, 5, 1)
	g := cudawarping.Upload(src)
	out := g.Resize(image.Point{X: 11, Y: 9}, 0, 0, cudawarping.InterCubic, nil).Download()
	if r, c := out.Rows, out.Cols; r != 9 || c != 11 {
		t.Fatalf("cubic resize size = (%d,%d), want (9,11)", r, c)
	}
}

func TestResizeZeroDsizeNeedsPositiveScale(t *testing.T) {
	g := cudawarping.Upload(gradient(4, 4, 1))
	defer func() {
		if recover() == nil {
			t.Fatal("Resize with zero dsize and zero scale should panic")
		}
	}()
	g.Resize(image.Point{}, 0, 0, cudawarping.InterLinear, nil)
}
