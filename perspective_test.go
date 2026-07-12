package cv

import (
	"math"
	"testing"
)

// applyPerspective maps a point through a PerspectiveMatrix (test helper).
func applyPerspective(m PerspectiveMatrix, x, y float64) (float64, float64) {
	w := m[6]*x + m[7]*y + m[8]
	return (m[0]*x + m[1]*y + m[2]) / w, (m[3]*x + m[4]*y + m[5]) / w
}

func TestGetPerspectiveTransformMapsCorners(t *testing.T) {
	src := [4]Point{{20, 20}, {80, 30}, {75, 85}, {25, 75}}
	dst := [4]Point{{10, 10}, {90, 10}, {90, 90}, {10, 90}}
	m := GetPerspectiveTransform(src, dst)
	for i := range src {
		gx, gy := applyPerspective(m, float64(src[i].X), float64(src[i].Y))
		if math.Abs(gx-float64(dst[i].X)) > 1e-6 || math.Abs(gy-float64(dst[i].Y)) > 1e-6 {
			t.Errorf("corner %d maps to (%v,%v), want (%d,%d)", i, gx, gy, dst[i].X, dst[i].Y)
		}
	}
}

func TestWarpPerspectiveQuadToRectangle(t *testing.T) {
	src := NewMat(100, 100, 1)
	quad := []Point{{20, 20}, {80, 30}, {75, 85}, {25, 75}}
	FillPoly(src, [][]Point{quad}, NewScalar(255))

	srcQ := [4]Point{{20, 20}, {80, 30}, {75, 85}, {25, 75}}
	dstQ := [4]Point{{10, 10}, {90, 10}, {90, 90}, {10, 90}}
	m := GetPerspectiveTransform(srcQ, dstQ)
	warped := WarpPerspective(src, m, 100, 100, InterNearest)

	// The interior of the destination rectangle should be white.
	if warped.At(50, 50, 0) != 255 {
		t.Errorf("warped centre = %d, want 255", warped.At(50, 50, 0))
	}
	if warped.At(20, 20, 0) != 255 {
		t.Errorf("warped rect interior = %d, want 255", warped.At(20, 20, 0))
	}
	// Well outside the rectangle should be black.
	if warped.At(3, 3, 0) != 0 {
		t.Errorf("warped corner = %d, want 0", warped.At(3, 3, 0))
	}
}

func TestWarpPerspectiveIdentity(t *testing.T) {
	src := NewMat(10, 10, 1)
	for i := range src.Data {
		src.Data[i] = uint8(i)
	}
	corners := [4]Point{{0, 0}, {9, 0}, {9, 9}, {0, 9}}
	m := GetPerspectiveTransform(corners, corners)
	out := WarpPerspective(src, m, 10, 10, InterNearest)
	for i := range src.Data {
		if out.Data[i] != src.Data[i] {
			t.Fatalf("identity warp changed pixel %d: %d != %d", i, out.Data[i], src.Data[i])
		}
	}
}

func TestRemapFlipHorizontal(t *testing.T) {
	src := grayFromValues(1, 3, []uint8{10, 20, 30})
	mapX := NewFloatMat(1, 3)
	mapY := NewFloatMat(1, 3)
	for x := 0; x < 3; x++ {
		mapX.Data[x] = float64(2 - x)
		mapY.Data[x] = 0
	}
	out := Remap(src, mapX, mapY, InterNearest)
	if out.Data[0] != 30 || out.Data[1] != 20 || out.Data[2] != 10 {
		t.Errorf("Remap flip = %v, want [30 20 10]", out.Data)
	}
}

func TestPyrDownUpDimensions(t *testing.T) {
	src := NewMat(8, 8, 1)
	src.SetTo(100)
	down := PyrDown(src)
	if down.Rows != 4 || down.Cols != 4 {
		t.Errorf("PyrDown dims = %dx%d, want 4x4", down.Rows, down.Cols)
	}
	// A constant image stays constant after down-sampling.
	if down.At(1, 1, 0) != 100 {
		t.Errorf("PyrDown constant = %d, want 100", down.At(1, 1, 0))
	}
	up := PyrUp(down)
	if up.Rows != 8 || up.Cols != 8 {
		t.Errorf("PyrUp dims = %dx%d, want 8x8", up.Rows, up.Cols)
	}
	// Interior of the up-sampled constant image is preserved.
	if v := up.At(4, 4, 0); v < 98 || v > 102 {
		t.Errorf("PyrUp constant interior = %d, want ~100", v)
	}
}

func TestDistanceTransform(t *testing.T) {
	m := NewMat(5, 5, 1)
	m.SetTo(255)
	m.Set(0, 0, 0, 0) // single background pixel
	dt := DistanceTransform(m)
	if dt.At(0, 0) != 0 {
		t.Errorf("distance at background = %v, want 0", dt.At(0, 0))
	}
	if math.Abs(dt.At(0, 1)-1) > 1e-9 {
		t.Errorf("distance at (0,1) = %v, want 1", dt.At(0, 1))
	}
	if math.Abs(dt.At(1, 1)-math.Sqrt2) > 1e-9 {
		t.Errorf("distance at (1,1) = %v, want sqrt2", dt.At(1, 1))
	}
}
