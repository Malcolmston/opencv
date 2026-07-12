package cudawarping_test

import (
	"image"
	"testing"

	cv "github.com/malcolmston/opencv"
	"github.com/malcolmston/opencv/cudawarping"
)

func TestLinearPolarKnownPosition(t *testing.T) {
	// A bright pixel 5 to the right of the centre (angle 0, radius 5) must land
	// on row 0 (angle 0), column 5 of the linear-polar image when
	// maxRadius == width == 10.
	src := cv.NewMat(21, 21, 1)
	src.Set(10, 15, 0, 255) // centre (10,10), so this is dx=5, dy=0
	g := cudawarping.Upload(src)
	polar := g.LinearPolar(image.Point{X: 10, Y: 8}, cudawarping.Point2f{X: 10, Y: 10}, 10, int(cudawarping.InterNearest), nil).Download()
	if got := polar.At(0, 5, 0); got != 255 {
		t.Fatalf("polar[0,5] = %d, want 255", got)
	}
}

func TestLinearPolarConstantRoundTrip(t *testing.T) {
	// A constant image maps to a constant polar image and back to a constant
	// image at pixels within the radius.
	src := cv.NewMat(21, 21, 1)
	src.SetTo(128)
	g := cudawarping.Upload(src)
	center := cudawarping.Point2f{X: 10, Y: 10}
	dsize := image.Point{X: 32, Y: 32}
	polar := g.LinearPolar(dsize, center, 10, int(cudawarping.InterLinear), nil)
	if polar.Download().At(5, 5, 0) != 128 {
		t.Fatalf("polar interior = %d, want 128", polar.Download().At(5, 5, 0))
	}
	back := polar.LinearPolar(image.Point{X: 21, Y: 21}, center, 10, int(cudawarping.InterLinear)|cudawarping.WarpInverseMap, nil).Download()
	if got := back.At(10, 10, 0); got != 128 {
		t.Fatalf("round-trip centre = %d, want 128", got)
	}
}

func TestLogPolarShapeAndInverse(t *testing.T) {
	src := gradient(31, 31, 3)
	g := cudawarping.Upload(src)
	center := cudawarping.Point2f{X: 15, Y: 15}
	dsize := image.Point{X: 40, Y: 40}
	polar := g.LogPolar(dsize, center, 15, int(cudawarping.InterLinear), nil)
	if r, c := polar.Size(); r != 40 || c != 40 {
		t.Fatalf("log-polar size = (%d,%d), want (40,40)", r, c)
	}
	back := polar.LogPolar(image.Point{X: 31, Y: 31}, center, 15, int(cudawarping.InterLinear)|cudawarping.WarpInverseMap, nil)
	if r, c := back.Size(); r != 31 || c != 31 {
		t.Fatalf("inverse log-polar size = (%d,%d), want (31,31)", r, c)
	}
}

func TestWarpPolarLogFlagEqualsLogPolar(t *testing.T) {
	src := gradient(25, 25, 1)
	g := cudawarping.Upload(src)
	center := cudawarping.Point2f{X: 12, Y: 12}
	dsize := image.Point{X: 20, Y: 20}
	viaWarp := g.WarpPolar(dsize, center, 12, int(cudawarping.InterLinear)|cudawarping.WarpPolarLog, nil).Download()
	viaLog := g.LogPolar(dsize, center, 12, int(cudawarping.InterLinear), nil).Download()
	if !equalMat(viaWarp, viaLog) {
		t.Fatal("WarpPolar with WarpPolarLog should equal LogPolar")
	}
}

func TestWarpPolarInvalidRadiusPanics(t *testing.T) {
	g := cudawarping.Upload(gradient(10, 10, 1))
	defer func() {
		if recover() == nil {
			t.Fatal("non-positive maxRadius should panic")
		}
	}()
	g.WarpPolar(image.Point{X: 8, Y: 8}, cudawarping.Point2f{X: 5, Y: 5}, 0, int(cudawarping.InterLinear), nil)
}
