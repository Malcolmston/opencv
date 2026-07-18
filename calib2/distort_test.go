package calib2

import (
	"math"
	"testing"

	cv "github.com/malcolmston/opencv"
)

func TestDistortUndistortRoundTrip(t *testing.T) {
	k := CameraMatrix{Fx: 800, Fy: 800, Cx: 320, Cy: 240}
	d := DistortionCoeffs{K1: -0.28, K2: 0.10, P1: 0.001, P2: -0.0015, K3: 0.02}
	for _, pt := range []cv.Point2f{{X: 100, Y: 90}, {X: 320, Y: 240}, {X: 500, Y: 400}, {X: 250, Y: 300}} {
		dp := DistortPoint(pt, k, d)
		back := UndistortPoint(dp, k, d)
		if math.Abs(back.X-pt.X) > 1e-4 || math.Abs(back.Y-pt.Y) > 1e-4 {
			t.Errorf("round trip for %v: distort->%v->undistort %v", pt, dp, back)
		}
	}
}

func TestUndistortNoDistortionIdentity(t *testing.T) {
	k := CameraMatrix{Fx: 800, Fy: 800, Cx: 320, Cy: 240}
	pt := cv.Point2f{X: 123, Y: 456}
	if got := UndistortPoint(pt, k, DistortionCoeffs{}); got != pt {
		t.Errorf("no-distortion undistort changed point: %v", got)
	}
	if got := DistortPoint(pt, k, DistortionCoeffs{}); got != pt {
		t.Errorf("no-distortion distort changed point: %v", got)
	}
}

func TestPrincipalPointUnaffected(t *testing.T) {
	// The principal point maps to itself under radial/tangential distortion.
	k := CameraMatrix{Fx: 800, Fy: 800, Cx: 320, Cy: 240}
	d := DistortionCoeffs{K1: -0.3, K2: 0.12}
	pt := cv.Point2f{X: 320, Y: 240}
	if got := DistortPoint(pt, k, d); math.Abs(got.X-320) > 1e-9 || math.Abs(got.Y-240) > 1e-9 {
		t.Errorf("principal point moved to %v", got)
	}
}

func TestUndistortImageIdentity(t *testing.T) {
	// With zero distortion the undistort map is the identity, so the output
	// equals the input (up to interpolation on integer coordinates).
	k := CameraMatrix{Fx: 50, Fy: 50, Cx: 15.5, Cy: 11.5}
	src := cv.NewMat(24, 32, 1)
	for y := 0; y < src.Rows; y++ {
		for x := 0; x < src.Cols; x++ {
			src.Set(y, x, 0, uint8((x*7+y*13)%256))
		}
	}
	out := UndistortImage(src, k, DistortionCoeffs{})
	if out.Rows != src.Rows || out.Cols != src.Cols {
		t.Fatalf("size changed: %dx%d", out.Rows, out.Cols)
	}
	var maxDiff int
	for y := 0; y < src.Rows; y++ {
		for x := 0; x < src.Cols; x++ {
			d := int(out.At(y, x, 0)) - int(src.At(y, x, 0))
			if d < 0 {
				d = -d
			}
			if d > maxDiff {
				maxDiff = d
			}
		}
	}
	if maxDiff != 0 {
		t.Errorf("identity undistort changed pixels, max diff %d", maxDiff)
	}
}

func TestRemapShiftKnown(t *testing.T) {
	// A constant-shift map moves the image; bilinear interpolation of an
	// integer shift reproduces exact source pixels.
	src := cv.NewMat(4, 4, 1)
	for y := 0; y < 4; y++ {
		for x := 0; x < 4; x++ {
			src.Set(y, x, 0, uint8(y*4+x))
		}
	}
	mapX := make([][]float64, 4)
	mapY := make([][]float64, 4)
	for v := 0; v < 4; v++ {
		mapX[v] = make([]float64, 4)
		mapY[v] = make([]float64, 4)
		for u := 0; u < 4; u++ {
			mapX[v][u] = float64(u) // sample column u
			mapY[v][u] = float64(v)
		}
	}
	out := Remap(src, mapX, mapY)
	for y := 0; y < 4; y++ {
		for x := 0; x < 4; x++ {
			if out.At(y, x, 0) != src.At(y, x, 0) {
				t.Fatalf("identity remap mismatch at %d,%d", y, x)
			}
		}
	}
}
