package stitching

import (
	"image"
	"math"
	"testing"

	cv "github.com/malcolmston/opencv"
)

// smoothImage builds a deterministic, band-limited grayscale image whose smooth
// variation keeps the resampling error of a warp round-trip small.
func smoothImage(rows, cols int) *cv.Mat {
	m := cv.NewMat(rows, cols, 1)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			v := 128 +
				40*math.Sin(float64(x)/11.0) +
				30*math.Cos(float64(y)/9.0) +
				20*math.Sin(float64(x+y)/17.0)
			m.Data[y*cols+x] = clampUint8(v)
		}
	}
	return m
}

func roundTripError(t *testing.T, w Warper, focal float64) float64 {
	t.Helper()
	src := smoothImage(80, 120)
	warped, corner := w.Warp(src, focal)
	if warped.Cols <= 0 || warped.Rows <= 0 {
		t.Fatalf("%s: warp produced empty image", w.Name())
	}
	back := w.WarpBackward(warped, focal, corner, src.Cols, src.Rows)
	if back.Cols != src.Cols || back.Rows != src.Rows {
		t.Fatalf("%s: backward size = %dx%d, want %dx%d", w.Name(), back.Cols, back.Rows, src.Cols, src.Rows)
	}
	var sum, count float64
	for y := 8; y < src.Rows-8; y++ {
		for x := 8; x < src.Cols-8; x++ {
			d := math.Abs(float64(back.Data[y*src.Cols+x]) - float64(src.Data[y*src.Cols+x]))
			sum += d
			count++
		}
	}
	return sum / count
}

func TestCylindricalWarpRoundTrip(t *testing.T) {
	if e := roundTripError(t, CylindricalWarper{}, 140); e > 6 {
		t.Errorf("cylindrical round-trip mean abs error = %.3f, want <= 6", e)
	}
}

func TestSphericalWarpRoundTrip(t *testing.T) {
	if e := roundTripError(t, SphericalWarper{}, 160); e > 6 {
		t.Errorf("spherical round-trip mean abs error = %.3f, want <= 6", e)
	}
}

func TestPlaneWarpIdentity(t *testing.T) {
	src := smoothImage(40, 50)
	warped, corner := PlaneWarper{}.Warp(src, 100)
	if warped.Rows != src.Rows || warped.Cols != src.Cols {
		t.Fatalf("plane warp resized image to %dx%d", warped.Cols, warped.Rows)
	}
	// Plane warp is the identity, so the interior must be reproduced exactly.
	for y := 1; y < src.Rows-1; y++ {
		for x := 1; x < src.Cols-1; x++ {
			if warped.Data[y*src.Cols+x] != src.Data[y*src.Cols+x] {
				t.Fatalf("plane warp changed pixel (%d,%d)", x, y)
			}
		}
	}
	_ = corner
}

func TestCylindricalWarpPointMonotone(t *testing.T) {
	// The forward map must be monotone in x: larger source columns map to larger
	// warped u coordinates.
	w := CylindricalWarper{}
	prev := math.Inf(-1)
	for x := 0.0; x <= 120; x += 4 {
		u, _ := w.WarpPoint(x, 40, 140, 60, 40)
		if u <= prev {
			t.Fatalf("warp x not monotone at x=%.0f: u=%.3f prev=%.3f", x, u, prev)
		}
		prev = u
	}
}

func TestWarpCornerNegative(t *testing.T) {
	// The warped region is centred on the principal point, so its top-left corner
	// should be up-and-left of the origin (negative components).
	src := smoothImage(60, 60)
	_, corner := SphericalWarper{}.Warp(src, 100)
	if corner.X > 0 || corner.Y > 0 {
		t.Errorf("warp corner = %v, want non-positive components", corner)
	}
	_ = image.Point{}
}
