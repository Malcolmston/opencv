package cv

import "testing"

func TestResizeDimensions(t *testing.T) {
	m := NewMat(4, 6, 3)
	out := Resize(m, 12, 8, InterNearest)
	if out.Cols != 12 || out.Rows != 8 || out.Channels != 3 {
		t.Fatalf("resize dims = %dx%dx%d", out.Rows, out.Cols, out.Channels)
	}
}

func TestResizeBilinearValue(t *testing.T) {
	// 2x2 image:
	//   0   100
	//   100 200
	m := grayFromValues(2, 2, []uint8{0, 100, 100, 200})
	out := Resize(m, 4, 4, InterLinear)
	// dst(1,1) maps to source (0.25,0.25):
	// top = 0*0.75 + 100*0.25 = 25; bot = 100*0.75 + 200*0.25 = 125;
	// value = 25*0.75 + 125*0.25 = 50.
	if got := out.At(1, 1, 0); got != 50 {
		t.Errorf("bilinear dst(1,1) = %d, want 50", got)
	}
}

func TestResizeNearestCorners(t *testing.T) {
	m := grayFromValues(2, 2, []uint8{10, 20, 30, 40})
	out := Resize(m, 4, 4, InterNearest)
	if out.At(0, 0, 0) != 10 || out.At(3, 3, 0) != 40 {
		t.Errorf("nearest corners = %d,%d", out.At(0, 0, 0), out.At(3, 3, 0))
	}
}

func TestFlip(t *testing.T) {
	m := grayFromValues(2, 2, []uint8{1, 2, 3, 4})
	h := Flip(m, FlipHorizontal)
	if h.At(0, 0, 0) != 2 || h.At(0, 1, 0) != 1 {
		t.Errorf("horizontal flip = %v", h.Data)
	}
	v := Flip(m, FlipVertical)
	if v.At(0, 0, 0) != 3 || v.At(1, 0, 0) != 1 {
		t.Errorf("vertical flip = %v", v.Data)
	}
}

func TestRotate90(t *testing.T) {
	// 2x3 image:
	//  1 2 3
	//  4 5 6
	m := grayFromValues(2, 3, []uint8{1, 2, 3, 4, 5, 6})
	cw := Rotate(m, Rotate90CW)
	if cw.Rows != 3 || cw.Cols != 2 {
		t.Fatalf("rotate90 dims = %dx%d", cw.Rows, cw.Cols)
	}
	// Top-left after CW rotation is the old bottom-left = 4.
	if cw.At(0, 0, 0) != 4 {
		t.Errorf("rotate90CW top-left = %d, want 4", cw.At(0, 0, 0))
	}
	// 180 rotation reverses everything.
	r180 := Rotate(m, Rotate180)
	if r180.At(0, 0, 0) != 6 || r180.At(1, 2, 0) != 1 {
		t.Errorf("rotate180 = %v", r180.Data)
	}
}

func TestTranspose(t *testing.T) {
	m := grayFromValues(2, 3, []uint8{1, 2, 3, 4, 5, 6})
	tr := Transpose(m)
	if tr.Rows != 3 || tr.Cols != 2 {
		t.Fatalf("transpose dims = %dx%d", tr.Rows, tr.Cols)
	}
	if tr.At(0, 0, 0) != 1 || tr.At(2, 1, 0) != 6 || tr.At(1, 0, 0) != 2 {
		t.Errorf("transpose content = %v", tr.Data)
	}
}

func TestWarpAffineIdentity(t *testing.T) {
	m := grayFromValues(2, 2, []uint8{1, 2, 3, 4})
	id := AffineMatrix{1, 0, 0, 0, 1, 0}
	out := WarpAffine(m, id, 2, 2, InterNearest)
	for i := range m.Data {
		if out.Data[i] != m.Data[i] {
			t.Fatalf("identity warp changed data at %d", i)
		}
	}
}

func TestWarpAffineTranslate(t *testing.T) {
	m := NewMat(4, 4, 1)
	m.Set(1, 1, 0, 200)
	// Translate by (+1, +1).
	tr := AffineMatrix{1, 0, 1, 0, 1, 1}
	out := WarpAffine(m, tr, 4, 4, InterNearest)
	if out.At(2, 2, 0) != 200 {
		t.Errorf("translate: expected 200 at (2,2), got %d", out.At(2, 2, 0))
	}
}

func TestGetRotationMatrix2DAndWarp180(t *testing.T) {
	m := NewMat(3, 3, 1)
	m.Set(0, 0, 0, 255)
	rot := GetRotationMatrix2D(1, 1, 180, 1)
	out := WarpAffine(m, rot, 3, 3, InterNearest)
	// A 180 rotation about the centre maps (0,0) to (2,2).
	if out.At(2, 2, 0) != 255 {
		t.Errorf("rotation 180: expected 255 at (2,2), got %d", out.At(2, 2, 0))
	}
}
