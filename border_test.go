package cv

import "testing"

func TestBorderInterpolate(t *testing.T) {
	cases := []struct {
		p, length int
		typ       BorderType
		want      int
	}{
		{-1, 5, BorderReplicate, 0},
		{5, 5, BorderReplicate, 4},
		{-1, 5, BorderReflect, 0},
		{-1, 5, BorderReflect101, 1},
		{-1, 5, BorderWrap, 4},
		{-1, 5, BorderConstant, -1},
		{6, 5, BorderReflect101, 2},
	}
	for _, c := range cases {
		if got := BorderInterpolate(c.p, c.length, c.typ); got != c.want {
			t.Errorf("BorderInterpolate(%d,%d,%d) = %d, want %d", c.p, c.length, c.typ, got, c.want)
		}
	}
}

func TestCopyMakeBorderReplicate(t *testing.T) {
	src := NewMat(2, 2, 1)
	copy(src.Data, []uint8{1, 2, 3, 4})
	dst := CopyMakeBorder(src, 1, 1, 1, 1, BorderReplicate, Scalar{})
	if dst.Rows != 4 || dst.Cols != 4 {
		t.Fatalf("dims = %dx%d", dst.Rows, dst.Cols)
	}
	// Corner replicates the nearest source pixel (1).
	if dst.At(0, 0, 0) != 1 {
		t.Errorf("top-left = %d, want 1", dst.At(0, 0, 0))
	}
	if dst.At(3, 3, 0) != 4 {
		t.Errorf("bottom-right = %d, want 4", dst.At(3, 3, 0))
	}
}

func TestCopyMakeBorderConstant(t *testing.T) {
	src := NewMat(1, 1, 1)
	src.Data[0] = 100
	dst := CopyMakeBorder(src, 1, 1, 1, 1, BorderConstant, NewScalar(7))
	if dst.At(0, 0, 0) != 7 {
		t.Errorf("border = %d, want 7", dst.At(0, 0, 0))
	}
	if dst.At(1, 1, 0) != 100 {
		t.Errorf("center = %d, want 100", dst.At(1, 1, 0))
	}
}
