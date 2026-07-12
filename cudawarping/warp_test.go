package cudawarping_test

import (
	"image"
	"testing"

	cv "github.com/malcolmston/opencv"
	"github.com/malcolmston/opencv/cudawarping"
)

// brightAt returns a black image with a single 255 pixel at (y, x).
func brightAt(rows, cols, y, x int) *cv.Mat {
	m := cv.NewMat(rows, cols, 1)
	m.Set(y, x, 0, 255)
	return m
}

// findBright returns the location of the maximum sample in a single-channel Mat.
func findBright(m *cv.Mat) (x, y int, val uint8) {
	for yy := 0; yy < m.Rows; yy++ {
		for xx := 0; xx < m.Cols; xx++ {
			if v := m.At(yy, xx, 0); v > val {
				val, x, y = v, xx, yy
			}
		}
	}
	return x, y, val
}

func TestWarpAffineIdentity(t *testing.T) {
	src := gradient(6, 8, 3)
	g := cudawarping.Upload(src)
	id := cv.AffineMatrix{1, 0, 0, 0, 1, 0}
	out := g.WarpAffine(id, image.Point{X: 8, Y: 6}, int(cudawarping.InterLinear), cudawarping.BorderConstant, 0, nil).Download()
	if !equalMat(src, out) {
		t.Fatal("identity affine warp changed the image")
	}
}

func TestWarpAffineTranslationKnownPosition(t *testing.T) {
	// Forward translation by (dx, dy): a feature at src (x0,y0) lands at
	// dst (x0+dx, y0+dy).
	const dx, dy = 2, 1
	src := brightAt(9, 9, 3, 4) // bright at x=4, y=3
	g := cudawarping.Upload(src)
	m := cv.AffineMatrix{1, 0, dx, 0, 1, dy}
	out := g.WarpAffine(m, image.Point{X: 9, Y: 9}, int(cudawarping.InterNearest), cudawarping.BorderConstant, 0, nil).Download()
	x, y, val := findBright(out)
	if val != 255 || x != 4+dx || y != 3+dy {
		t.Fatalf("translated bright at (%d,%d) val %d, want (%d,%d) 255", x, y, val, 4+dx, 3+dy)
	}
}

func TestWarpAffineInverseFlagKnownPosition(t *testing.T) {
	// With WARP_INVERSE_MAP the matrix maps destination to source directly:
	// dst(x,y) = src(x+dx, y+dy) for the translation matrix below. So a feature
	// at src (x0,y0) appears at dst (x0-dx, y0-dy).
	const dx, dy = 2, 1
	src := brightAt(9, 9, 5, 5) // bright at x=5, y=5
	g := cudawarping.Upload(src)
	inv := cv.AffineMatrix{1, 0, dx, 0, 1, dy}
	flags := int(cudawarping.InterNearest) | cudawarping.WarpInverseMap
	out := g.WarpAffine(inv, image.Point{X: 9, Y: 9}, flags, cudawarping.BorderConstant, 0, nil).Download()
	x, y, val := findBright(out)
	if val != 255 || x != 5-dx || y != 5-dy {
		t.Fatalf("inverse-map bright at (%d,%d) val %d, want (%d,%d) 255", x, y, val, 5-dx, 5-dy)
	}
}

// TestWarpAffineInverseFlagMapConsistency checks that the inverse-map flag with a
// non-default border resamples through BuildWarpAffineMaps(inverse=true).
func TestWarpAffineInverseFlagMapConsistency(t *testing.T) {
	src := gradient(7, 7, 1)
	g := cudawarping.Upload(src)
	m := cv.GetRotationMatrix2D(3, 3, 30, 1.2)
	dsize := image.Point{X: 7, Y: 7}
	flags := int(cudawarping.InterCubic) | cudawarping.WarpInverseMap
	got := g.WarpAffine(m, dsize, flags, cudawarping.BorderReplicate, 0, nil).Download()

	xmap, ymap := cudawarping.BuildWarpAffineMaps(m, true, dsize, nil)
	want := g.Remap(xmap, ymap, cudawarping.InterCubic, cudawarping.BorderReplicate, 0, nil).Download()
	if !equalMat(got, want) {
		t.Fatal("inverse-map cubic warp did not match the map-based remap")
	}
}

func TestWarpAffineDefaultMatchesCV(t *testing.T) {
	src := gradient(6, 6, 3)
	g := cudawarping.Upload(src)
	m := cv.GetRotationMatrix2D(2.5, 2.5, 20, 1)
	got := g.WarpAffine(m, image.Point{X: 6, Y: 6}, int(cudawarping.InterLinear), cudawarping.BorderConstant, 0, nil).Download()
	want := cv.WarpAffine(src, m, 6, 6, cv.InterLinear)
	if !equalMat(got, want) {
		t.Fatal("default WarpAffine did not delegate to cv.WarpAffine exactly")
	}
}

func TestWarpAffineBorderReplicate(t *testing.T) {
	// Fill the image with a constant so replicate keeps that value even where
	// the sample falls outside, while a constant-0 border produces 0.
	src := cv.NewMat(5, 5, 1)
	src.SetTo(200)
	g := cudawarping.Upload(src)
	// Translate content up-left so the bottom-right maps outside the source.
	m := cv.AffineMatrix{1, 0, 3, 0, 1, 3}
	rep := g.WarpAffine(m, image.Point{X: 5, Y: 5}, int(cudawarping.InterNearest), cudawarping.BorderReplicate, 0, nil).Download()
	con := g.WarpAffine(m, image.Point{X: 5, Y: 5}, int(cudawarping.InterNearest), cudawarping.BorderConstant, 0, nil).Download()
	if rep.At(0, 0, 0) != 200 {
		t.Fatalf("replicate border = %d, want 200", rep.At(0, 0, 0))
	}
	if con.At(0, 0, 0) != 0 {
		t.Fatalf("constant border = %d, want 0", con.At(0, 0, 0))
	}
}

func TestWarpAffineBorderValue(t *testing.T) {
	src := cv.NewMat(4, 4, 1)
	src.SetTo(10)
	g := cudawarping.Upload(src)
	// Map everything outside the source.
	m := cv.AffineMatrix{1, 0, 100, 0, 1, 100}
	out := g.WarpAffine(m, image.Point{X: 4, Y: 4}, int(cudawarping.InterNearest), cudawarping.BorderConstant, 77, nil).Download()
	if out.At(0, 0, 0) != 77 {
		t.Fatalf("border value = %d, want 77", out.At(0, 0, 0))
	}
}

func TestWarpPerspectiveIdentity(t *testing.T) {
	src := gradient(6, 6, 3)
	g := cudawarping.Upload(src)
	id := cv.PerspectiveMatrix{1, 0, 0, 0, 1, 0, 0, 0, 1}
	out := g.WarpPerspective(id, image.Point{X: 6, Y: 6}, int(cudawarping.InterLinear), cudawarping.BorderConstant, 0, nil).Download()
	if !equalMat(src, out) {
		t.Fatal("identity perspective warp changed the image")
	}
}

func TestWarpPerspectiveRecoversCorners(t *testing.T) {
	// A homography mapping the unit square corners to a known quad, applied to a
	// bright corner, should land the corner at its mapped position.
	srcPts := [4]cv.Point{{X: 0, Y: 0}, {X: 8, Y: 0}, {X: 8, Y: 8}, {X: 0, Y: 8}}
	dstPts := [4]cv.Point{{X: 2, Y: 1}, {X: 8, Y: 0}, {X: 7, Y: 8}, {X: 1, Y: 7}}
	h := cv.GetPerspectiveTransform(srcPts, dstPts)
	src := brightAt(9, 9, 0, 8) // bright at src corner (x=8,y=0)
	g := cudawarping.Upload(src)
	out := g.WarpPerspective(h, image.Point{X: 9, Y: 9}, int(cudawarping.InterNearest), cudawarping.BorderConstant, 0, nil).Download()
	x, y, val := findBright(out)
	if val != 255 || x != 8 || y != 0 {
		t.Fatalf("mapped corner at (%d,%d) val %d, want (8,0) 255", x, y, val)
	}
}

func TestWarpPerspectiveDefaultMatchesCV(t *testing.T) {
	src := gradient(6, 6, 3)
	g := cudawarping.Upload(src)
	srcPts := [4]cv.Point{{X: 0, Y: 0}, {X: 5, Y: 0}, {X: 5, Y: 5}, {X: 0, Y: 5}}
	dstPts := [4]cv.Point{{X: 1, Y: 0}, {X: 5, Y: 1}, {X: 4, Y: 5}, {X: 0, Y: 4}}
	h := cv.GetPerspectiveTransform(srcPts, dstPts)
	got := g.WarpPerspective(h, image.Point{X: 6, Y: 6}, int(cudawarping.InterLinear), cudawarping.BorderConstant, 0, nil).Download()
	want := cv.WarpPerspective(src, h, 6, 6, cv.InterLinear)
	if !equalMat(got, want) {
		t.Fatal("default WarpPerspective did not delegate to cv.WarpPerspective exactly")
	}
}

func TestRotate90MatchesCV(t *testing.T) {
	src := gradient(4, 6, 3)
	g := cudawarping.Upload(src)
	for _, code := range []cudawarping.RotateCode{
		cudawarping.Rotate90CW, cudawarping.Rotate180, cudawarping.Rotate90CCW,
	} {
		got := g.Rotate90(code, nil).Download()
		want := cv.Rotate(src, code)
		if !equalMat(got, want) {
			t.Fatalf("Rotate90 code %d did not match cv.Rotate", code)
		}
	}
}

func TestRotateArbitraryAngle(t *testing.T) {
	// A 90° rotation about the origin with a shift that maps the image back into
	// frame. Point (x,y) -> (cos·x+sin·y+xs, -sin·x+cos·y+ys); at 90°,
	// (x,y) -> (y+xs, -x+ys). Choose shifts so a known bright pixel lands in view.
	src := brightAt(9, 9, 1, 0) // bright at x=0, y=1
	g := cudawarping.Upload(src)
	// angle 90: cos=0, sin=1 -> dst = (y+xs, -x+ys). With xs=0, ys=8:
	// src(0,1) -> dst(1, 8).
	out := g.Rotate(image.Point{X: 9, Y: 9}, 90, 0, 8, cudawarping.InterNearest, nil).Download()
	x, y, val := findBright(out)
	if val != 255 || x != 1 || y != 8 {
		t.Fatalf("rotated bright at (%d,%d) val %d, want (1,8) 255", x, y, val)
	}
}

func TestBuildRotationMapsRemapEqualsRotate(t *testing.T) {
	src := gradient(7, 7, 1)
	g := cudawarping.Upload(src)
	dsize := image.Point{X: 7, Y: 7}
	direct := g.Rotate(dsize, 25, 1, 2, cudawarping.InterLinear, nil).Download()
	xmap, ymap := cudawarping.BuildRotationMaps(25, 1, 2, dsize, nil)
	viaMaps := g.Remap(xmap, ymap, cudawarping.InterLinear, cudawarping.BorderConstant, 0, nil).Download()
	if !equalMat(direct, viaMaps) {
		t.Fatal("BuildRotationMaps + Remap did not match Rotate")
	}
}
