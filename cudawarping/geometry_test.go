package cudawarping_test

import (
	"testing"

	cv "github.com/malcolmston/opencv"
	"github.com/malcolmston/opencv/cudawarping"
)

func TestTransposeMatchesCV(t *testing.T) {
	src := gradient(4, 6, 3)
	g := cudawarping.Upload(src)
	got := g.Transpose(nil).Download()
	want := cv.Transpose(src)
	if !equalMat(got, want) {
		t.Fatal("Transpose did not match cv.Transpose")
	}
	if got.Rows != 6 || got.Cols != 4 {
		t.Fatalf("transposed size = (%d,%d), want (6,4)", got.Rows, got.Cols)
	}
}

func TestTransposeTwiceIsIdentity(t *testing.T) {
	src := gradient(5, 3, 2)
	g := cudawarping.Upload(src)
	out := g.Transpose(nil).Transpose(nil).Download()
	if !equalMat(src, out) {
		t.Fatal("transposing twice should restore the image")
	}
}

func TestFlipMatchesCV(t *testing.T) {
	src := gradient(4, 5, 3)
	g := cudawarping.Upload(src)
	cases := []struct {
		flipCode int
		code     cv.FlipCode
	}{
		{0, cv.FlipVertical},
		{1, cv.FlipHorizontal},
		{-1, cv.FlipBoth},
	}
	for _, tc := range cases {
		got := g.Flip(tc.flipCode, nil).Download()
		want := cv.Flip(src, tc.code)
		if !equalMat(got, want) {
			t.Fatalf("Flip(%d) did not match cv.Flip", tc.flipCode)
		}
	}
}

func TestPyrDownUpMatchesCV(t *testing.T) {
	src := gradient(8, 8, 3)
	g := cudawarping.Upload(src)
	down := g.PyrDown(nil).Download()
	if !equalMat(down, cv.PyrDown(src)) {
		t.Fatal("PyrDown did not match cv.PyrDown")
	}
	up := g.PyrUp(nil).Download()
	if !equalMat(up, cv.PyrUp(src)) {
		t.Fatal("PyrUp did not match cv.PyrUp")
	}
	if down.Rows != 4 || down.Cols != 4 {
		t.Fatalf("PyrDown size = (%d,%d), want (4,4)", down.Rows, down.Cols)
	}
	if up.Rows != 16 || up.Cols != 16 {
		t.Fatalf("PyrUp size = (%d,%d), want (16,16)", up.Rows, up.Cols)
	}
}

func TestCopyMakeBorderReplicate(t *testing.T) {
	src := cv.NewMat(2, 2, 1)
	src.SetTo(5)
	g := cudawarping.Upload(src)
	out := g.CopyMakeBorder(1, 1, 2, 2, cudawarping.BorderReplicate, 0, nil).Download()
	if out.Rows != 4 || out.Cols != 6 {
		t.Fatalf("bordered size = (%d,%d), want (4,6)", out.Rows, out.Cols)
	}
	// Every pixel is 5, so replicate keeps 5 in the border too.
	for _, p := range [][2]int{{0, 0}, {3, 5}, {0, 5}, {3, 0}} {
		if out.At(p[0], p[1], 0) != 5 {
			t.Fatalf("replicate corner (%d,%d) = %d, want 5", p[0], p[1], out.At(p[0], p[1], 0))
		}
	}
}

func TestCopyMakeBorderConstant(t *testing.T) {
	src := cv.NewMat(2, 2, 1)
	src.SetTo(5)
	g := cudawarping.Upload(src)
	out := g.CopyMakeBorder(1, 0, 1, 0, cudawarping.BorderConstant, 99, nil).Download()
	if out.At(0, 0, 0) != 99 {
		t.Fatalf("constant border corner = %d, want 99", out.At(0, 0, 0))
	}
	// The interior (shifted by top=1,left=1) is the original.
	if out.At(1, 1, 0) != 5 {
		t.Fatalf("interior = %d, want 5", out.At(1, 1, 0))
	}
}

func TestCopyMakeBorderNegativePanics(t *testing.T) {
	g := cudawarping.Upload(gradient(3, 3, 1))
	defer func() {
		if recover() == nil {
			t.Fatal("negative border width should panic")
		}
	}()
	g.CopyMakeBorder(-1, 0, 0, 0, cudawarping.BorderConstant, 0, nil)
}
