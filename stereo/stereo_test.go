package stereo

import (
	"testing"

	cv "github.com/malcolmston/opencv"
)

// texture returns a deterministic, non-repetitive intensity for column x, row y.
// The multiplicative hash makes neighbouring columns distinctive so block
// matching finds a unique correspondence.
func texture(x, y int) uint8 {
	v := (x*167 + y*83 + ((x * x) % 91) + ((y * x) % 71)) % 256
	if v < 0 {
		v += 256
	}
	return uint8(v)
}

// syntheticPair builds a rectified stereo pair. The right image is a textured
// background; the left image is the same texture except that the rectangular
// region [ry, ry+rh) × [rx, rx+rw) is copied from `disp` columns to the right,
// i.e. left[x] = right[x-disp] there. So the region has a known disparity while
// the surrounding background has disparity 0, and a flat block (all one value)
// is inset to exercise the texture/invalid path.
func syntheticPair(w, h, disp, rx, ry, rw, rh int) (left, right *cv.Mat) {
	right = cv.NewMat(h, w, 1)
	left = cv.NewMat(h, w, 1)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			right.Data[y*w+x] = texture(x, y)
		}
	}
	// Background of left mirrors right (disparity 0).
	copy(left.Data, right.Data)
	// Foreground region: shift texture right by `disp`.
	for y := ry; y < ry+rh; y++ {
		for x := rx; x < rx+rw; x++ {
			sx := x - disp
			if sx < 0 {
				sx = 0
			}
			left.Data[y*w+x] = texture(sx, y)
		}
	}
	// A flat, textureless patch near the region centre to test invalidation.
	for y := ry + 2; y < ry+6; y++ {
		for x := rx + 2; x < rx+6; x++ {
			left.Data[y*w+x] = 128
			right.Data[y*w+x] = 128
		}
	}
	return left, right
}

// modeDisparity returns the most common disparity value over the interior of the
// region (avoiding its edges and the flat inset) and the fraction of interior
// pixels that were valid.
func modeDisparity(t *testing.T, d *cv.Mat, rx, ry, rw, rh int) (mode int, validFrac float64) {
	t.Helper()
	counts := map[int]int{}
	total, valid := 0, 0
	for y := ry + 7; y < ry+rh-2; y++ {
		for x := rx + 7; x < rx+rw-2; x++ {
			total++
			v := d.Data[y*d.Cols+x]
			if v == InvalidDisparity {
				continue
			}
			valid++
			counts[int(v)]++
		}
	}
	best, bestN := 0, -1
	for k, n := range counts {
		if n > bestN {
			best, bestN = k, n
		}
	}
	if total == 0 {
		return 0, 0
	}
	return best, float64(valid) / float64(total)
}

func TestStereoBMRecoversDisparity(t *testing.T) {
	const w, h, disp = 64, 40, 8
	const rx, ry, rw, rh = 24, 8, 32, 24
	left, right := syntheticPair(w, h, disp, rx, ry, rw, rh)

	bm := StereoBM{NumDisparities: 16, BlockSize: 7}
	d := bm.Compute(left, right)
	if d.Rows != h || d.Cols != w || d.Channels != 1 {
		t.Fatalf("unexpected output shape %dx%dx%d", d.Rows, d.Cols, d.Channels)
	}

	mode, frac := modeDisparity(t, d, rx, ry, rw, rh)
	if mode < disp-1 || mode > disp+1 {
		t.Errorf("StereoBM recovered disparity %d, want %d ±1", mode, disp)
	}
	if frac < 0.5 {
		t.Errorf("StereoBM valid fraction in region = %.2f, want >= 0.5", frac)
	}
}

func TestStereoSGBMRecoversDisparity(t *testing.T) {
	const w, h, disp = 64, 40, 8
	const rx, ry, rw, rh = 24, 8, 32, 24
	left, right := syntheticPair(w, h, disp, rx, ry, rw, rh)

	sg := StereoSGBM{NumDisparities: 16, BlockSize: 5}
	d := sg.Compute(left, right)
	if d.Rows != h || d.Cols != w {
		t.Fatalf("unexpected output shape %dx%d", d.Rows, d.Cols)
	}

	mode, frac := modeDisparity(t, d, rx, ry, rw, rh)
	if mode < disp-1 || mode > disp+1 {
		t.Errorf("StereoSGBM recovered disparity %d, want %d ±1", mode, disp)
	}
	if frac < 0.5 {
		t.Errorf("StereoSGBM valid fraction in region = %.2f, want >= 0.5", frac)
	}
}

func TestLeftBorderInvalid(t *testing.T) {
	const w, h = 48, 20
	left, right := syntheticPair(w, h, 6, 20, 4, 20, 12)
	bm := StereoBM{NumDisparities: 16, BlockSize: 5}
	d := bm.Compute(left, right)
	// The leftmost NumDisparities-1 columns cannot be searched and must be invalid.
	for y := 0; y < h; y++ {
		for x := 0; x < 15; x++ {
			if d.Data[y*w+x] != InvalidDisparity {
				t.Fatalf("expected invalid at border (x=%d,y=%d), got %d", x, y, d.Data[y*w+x])
			}
		}
	}
}

func TestUniformRegionInvalid(t *testing.T) {
	// A completely flat pair has no texture anywhere -> everything invalid.
	const w, h = 40, 24
	left := cv.NewMat(h, w, 1)
	right := cv.NewMat(h, w, 1)
	left.SetTo(100)
	right.SetTo(100)
	bm := StereoBM{NumDisparities: 16, BlockSize: 5}
	d := bm.Compute(left, right)
	for i, v := range d.Data {
		if v != InvalidDisparity {
			t.Fatalf("expected all-invalid on uniform input, got %d at %d", v, i)
		}
	}
}

func TestReprojectConstantDisparityConstantDepth(t *testing.T) {
	const w, h = 16, 12
	const dval uint8 = 8
	d := cv.NewMat(h, w, 1)
	d.SetTo(dval)

	// A rectification-style Q: depth depends only on disparity.
	const f, cx, cy, tx = 500.0, 8.0, 6.0, 0.1
	Q := [4][4]float64{
		{1, 0, 0, -cx},
		{0, 1, 0, -cy},
		{0, 0, 0, f},
		{0, 0, 1 / tx, 0},
	}
	pts := ReprojectImageTo3D(d, Q)
	if len(pts) != w*h {
		t.Fatalf("expected %d points, got %d", w*h, len(pts))
	}
	z0 := pts[0][2]
	for i, p := range pts {
		if diff := p[2] - z0; diff > 1e-9 || diff < -1e-9 {
			t.Fatalf("depth not constant: pt %d has Z=%v want %v", i, p[2], z0)
		}
	}
	// Sanity: Z = f / (d/tx) = f*tx/d.
	wantZ := f * tx / float64(dval)
	if diff := z0 - wantZ; diff > 1e-6 || diff < -1e-6 {
		t.Fatalf("Z = %v, want %v", z0, wantZ)
	}
	// X should vary with column (proves per-pixel reprojection, not a constant).
	if pts[0][0] == pts[w-1][0] {
		t.Fatalf("X did not vary across a row: %v", pts[0][0])
	}
}

func TestReprojectZeroW(t *testing.T) {
	d := cv.NewMat(2, 2, 1)
	d.SetTo(5)
	// Q whose last row is all zero -> W==0 for every pixel.
	var Q [4][4]float64
	Q[0][0] = 1
	pts := ReprojectImageTo3D(d, Q)
	for i, p := range pts {
		if p != [3]float64{0, 0, 0} {
			t.Fatalf("expected zero point for W==0 at %d, got %v", i, p)
		}
	}
}

func TestFilterSpecklesDisparity(t *testing.T) {
	const w, h = 20, 20
	d := cv.NewMat(h, w, 1)
	// A large valid blob of disparity 10 across the top half.
	for y := 0; y < 10; y++ {
		for x := 0; x < w; x++ {
			d.Data[y*w+x] = 10
		}
	}
	// A tiny isolated speckle of disparity 30 in the (invalid) bottom half.
	d.Data[15*w+15] = 30
	d.Data[15*w+16] = 30

	FilterSpecklesDisparity(d, InvalidDisparity, 5, 4)

	if d.Data[15*w+15] != InvalidDisparity || d.Data[15*w+16] != InvalidDisparity {
		t.Errorf("small speckle was not removed: %d %d", d.Data[15*w+15], d.Data[15*w+16])
	}
	// The large blob must survive.
	if d.Data[5*w+5] != 10 {
		t.Errorf("large blob was wrongly removed, got %d", d.Data[5*w+5])
	}
}

func TestRectifyStubClones(t *testing.T) {
	left := cv.NewMat(4, 4, 1)
	right := cv.NewMat(4, 4, 1)
	left.SetTo(1)
	right.SetTo(2)
	rl, rr := Rectify(left, right)
	if rl == left || rr == right {
		t.Fatal("Rectify should return independent clones")
	}
	rl.Data[0] = 99
	if left.Data[0] == 99 {
		t.Fatal("Rectify clone shares storage with input")
	}
}

func TestChannelAndSizeValidation(t *testing.T) {
	rgbL := cv.NewMat(20, 40, 3)
	rgbR := cv.NewMat(20, 40, 3)
	// Three-channel inputs are accepted (converted to gray) and produce output.
	d := StereoBM{NumDisparities: 8, BlockSize: 5}.Compute(rgbL, rgbR)
	if d.Rows != 20 || d.Cols != 40 {
		t.Fatalf("unexpected shape from RGB input: %dx%d", d.Rows, d.Cols)
	}

	// Mismatched sizes must panic.
	func() {
		defer func() {
			if recover() == nil {
				t.Error("expected panic on size mismatch")
			}
		}()
		StereoBM{NumDisparities: 8, BlockSize: 5}.Compute(cv.NewMat(10, 10, 1), cv.NewMat(12, 10, 1))
	}()
}
