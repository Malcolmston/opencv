package cudawarping_test

import (
	"testing"

	cv "github.com/malcolmston/opencv"
	"github.com/malcolmston/opencv/cudawarping"
)

// identityMaps builds coordinate maps that reproduce the source unchanged.
func identityMaps(rows, cols int) (xmap, ymap *cv.FloatMat) {
	xmap = cv.NewFloatMat(rows, cols)
	ymap = cv.NewFloatMat(rows, cols)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			xmap.Data[y*cols+x] = float64(x)
			ymap.Data[y*cols+x] = float64(y)
		}
	}
	return xmap, ymap
}

func TestRemapIdentity(t *testing.T) {
	src := gradient(5, 7, 3)
	g := cudawarping.Upload(src)
	xmap, ymap := identityMaps(5, 7)
	for _, interp := range []cudawarping.Interpolation{
		cudawarping.InterNearest, cudawarping.InterLinear, cudawarping.InterCubic,
	} {
		out := g.Remap(xmap, ymap, interp, cudawarping.BorderReplicate, 0, nil).Download()
		if !equalMat(src, out) {
			t.Fatalf("identity remap with interp %d changed the image", interp)
		}
	}
}

func TestRemapDefaultMatchesCV(t *testing.T) {
	src := gradient(6, 6, 3)
	g := cudawarping.Upload(src)
	// A shift-by-one map.
	xmap := cv.NewFloatMat(6, 6)
	ymap := cv.NewFloatMat(6, 6)
	for y := 0; y < 6; y++ {
		for x := 0; x < 6; x++ {
			xmap.Data[y*6+x] = float64(x) - 1
			ymap.Data[y*6+x] = float64(y) + 0.5
		}
	}
	got := g.Remap(xmap, ymap, cudawarping.InterLinear, cudawarping.BorderConstant, 0, nil).Download()
	want := cv.Remap(src, xmap, ymap, cv.InterLinear)
	if !equalMat(got, want) {
		t.Fatal("default Remap did not delegate to cv.Remap exactly")
	}
}

func TestRemapBorderModes(t *testing.T) {
	src := cv.NewMat(4, 4, 1)
	src.SetTo(150)
	g := cudawarping.Upload(src)
	// Sample well outside the source at every pixel.
	xmap := cv.NewFloatMat(4, 4)
	ymap := cv.NewFloatMat(4, 4)
	for i := range xmap.Data {
		xmap.Data[i] = -5
		ymap.Data[i] = -5
	}
	rep := g.Remap(xmap, ymap, cudawarping.InterNearest, cudawarping.BorderReplicate, 0, nil).Download()
	if rep.At(0, 0, 0) != 150 {
		t.Fatalf("replicate remap = %d, want 150", rep.At(0, 0, 0))
	}
	con := g.Remap(xmap, ymap, cudawarping.InterNearest, cudawarping.BorderConstant, 42, nil).Download()
	if con.At(0, 0, 0) != 42 {
		t.Fatalf("constant remap = %d, want 42", con.At(0, 0, 0))
	}
}

func TestRemapMismatchedMapsPanic(t *testing.T) {
	g := cudawarping.Upload(gradient(4, 4, 1))
	defer func() {
		if recover() == nil {
			t.Fatal("mismatched map sizes should panic")
		}
	}()
	g.Remap(cv.NewFloatMat(4, 4), cv.NewFloatMat(3, 4), cudawarping.InterLinear, cudawarping.BorderConstant, 0, nil)
}
