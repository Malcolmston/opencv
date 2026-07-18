package stitch

import (
	"math"
	"testing"

	cv "github.com/malcolmston/opencv"
)

func TestProjectCenterMapsToOrigin(t *testing.T) {
	u, v := ProjectCylindrical(5, 7, 30, 5, 7)
	if math.Abs(u) > 1e-12 || math.Abs(v) > 1e-12 {
		t.Fatalf("cylindrical centre = (%g,%g), want (0,0)", u, v)
	}
	u, v = ProjectSpherical(5, 7, 30, 5, 7)
	if math.Abs(u) > 1e-12 || math.Abs(v) > 1e-12 {
		t.Fatalf("spherical centre = (%g,%g), want (0,0)", u, v)
	}
}

func TestProjectCylindricalMonotonic(t *testing.T) {
	// u increases with x; small-angle u ≈ (x-cx).
	u1, _ := ProjectCylindrical(6, 5, 100, 5, 5)
	u0, _ := ProjectCylindrical(5, 5, 100, 5, 5)
	if u1 <= u0 {
		t.Fatal("cylindrical u must increase with x")
	}
	if math.Abs(u1-1) > 0.01 {
		t.Fatalf("small-angle u = %g, want ≈1", u1)
	}
}

func TestWarpCylindricalSolidColor(t *testing.T) {
	img := cv.NewMat(21, 21, 3)
	col := []uint8{100, 150, 200}
	for p := 0; p < img.Rows*img.Cols; p++ {
		copy(img.Data[p*3:p*3+3], col)
	}
	out, _, _ := WarpCylindrical(img, 40)
	if out.Empty() {
		t.Fatal("warp produced empty image")
	}
	cx, cy := out.Cols/2, out.Rows/2
	for c := 0; c < 3; c++ {
		if d := int(out.At(cy, cx, c)) - int(col[c]); d < -1 || d > 1 {
			t.Fatalf("centre channel %d = %d, want ≈%d", c, out.At(cy, cx, c), col[c])
		}
	}
}

func TestWarpSphericalNonEmpty(t *testing.T) {
	img := cv.NewMat(15, 15, 1)
	for i := range img.Data {
		img.Data[i] = 128
	}
	out, ox, oy := WarpSpherical(img, 25)
	if out.Empty() {
		t.Fatal("spherical warp empty")
	}
	if ox > 0 || oy > 0 {
		t.Fatalf("offsets should be non-positive for a centred image, got (%d,%d)", ox, oy)
	}
}

func TestWarpPerspectiveToCanvasIdentity(t *testing.T) {
	img := cv.NewMat(4, 4, 3)
	for i := range img.Data {
		img.Data[i] = uint8(i % 256)
	}
	layer := WarpPerspectiveToCanvas(img, IdentityHomography(), Bounds{0, 0, 4, 4})
	if layer.Image == nil {
		t.Fatal("nil layer image")
	}
	for i := range img.Data {
		if layer.Image.Data[i] != img.Data[i] {
			t.Fatalf("identity warp changed pixel %d: %d vs %d", i, layer.Image.Data[i], img.Data[i])
		}
	}
	// All interior pixels covered with positive weight.
	for p := 0; p < layer.Weight.Rows*layer.Weight.Cols; p++ {
		if layer.Weight.Data[p] <= 0 {
			t.Fatalf("weight[%d] = %g, want > 0", p, layer.Weight.Data[p])
		}
	}
}

func TestWarpPerspectiveToCanvasTranslation(t *testing.T) {
	img := cv.NewMat(3, 3, 1)
	copy(img.Data, []uint8{1, 2, 3, 4, 5, 6, 7, 8, 9})
	h := TranslationHomography(2, 1)
	canvas := WarpedBounds(3, 3, h)
	layer := WarpPerspectiveToCanvas(img, h, canvas)
	// Global (2,1) is source (0,0)=1; canvas origin is (2,1) so layer pixel (0,0).
	if layer.Image.At(0, 0, 0) != 1 {
		t.Fatalf("translated top-left = %d, want 1", layer.Image.At(0, 0, 0))
	}
}
