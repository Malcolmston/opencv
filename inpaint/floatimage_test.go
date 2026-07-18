package inpaint

import "testing"

func TestGradientKnownAnswer(t *testing.T) {
	// I(y,x) = 10*x + 5*y. Interior derivatives are exactly 10 and 5.
	img := rampMat(6, 6, 10, 5, 0)
	gx := GradientX(img)
	gy := GradientY(img)
	if got := gx.At(2, 2, 0); got != 10 {
		t.Fatalf("GradientX interior = %v, want 10", got)
	}
	if got := gy.At(2, 2, 0); got != 5 {
		t.Fatalf("GradientY interior = %v, want 5", got)
	}
}

func TestLaplacianOfLinearIsZero(t *testing.T) {
	img := rampMat(6, 6, 7, 3, 0)
	lap := Laplacian(img)
	for y := 1; y < 5; y++ {
		for x := 1; x < 5; x++ {
			if v := lap.At(y, x, 0); v != 0 {
				t.Fatalf("Laplacian(%d,%d) = %v, want 0", y, x, v)
			}
		}
	}
}

func TestDivergenceEqualsLaplacian(t *testing.T) {
	img := rampMat(6, 6, 4, 9, 10)
	lap := Laplacian(img)
	div := Divergence(GradientX(img), GradientY(img))
	for y := 1; y < 5; y++ {
		for x := 1; x < 5; x++ {
			if div.At(y, x, 0) != lap.At(y, x, 0) {
				t.Fatalf("div(%d,%d)=%v lap=%v", y, x, div.At(y, x, 0), lap.At(y, x, 0))
			}
		}
	}
}

func TestFloatImageRoundTrip(t *testing.T) {
	img := uniformMat(3, 3, 3, 77)
	f := FloatImageFromMat(img)
	f.Set(1, 1, 0, 123.6)
	back := f.ToMat()
	if back.At(1, 1, 0) != 124 { // rounded
		t.Fatalf("ToMat rounding = %d, want 124", back.At(1, 1, 0))
	}
	if back.At(0, 0, 2) != 77 {
		t.Fatalf("unchanged sample = %d, want 77", back.At(0, 0, 2))
	}
}
