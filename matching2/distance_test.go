package matching2

import (
	"math"
	"testing"
)

func TestL2Distance(t *testing.T) {
	a := []float64{0, 0, 0}
	b := []float64{3, 4, 0}
	if got := L2Distance(a, b); !approx(got, 5, 1e-12) {
		t.Fatalf("L2Distance = %v, want 5", got)
	}
	if got := L2SquaredDistance(a, b); !approx(got, 25, 1e-12) {
		t.Fatalf("L2SquaredDistance = %v, want 25", got)
	}
}

func TestL1Distance(t *testing.T) {
	a := []float64{1, 2, 3}
	b := []float64{4, 0, 3}
	if got := L1Distance(a, b); !approx(got, 5, 1e-12) {
		t.Fatalf("L1Distance = %v, want 5", got)
	}
}

func TestFloatDistanceDispatch(t *testing.T) {
	a := []float64{0, 0}
	b := []float64{3, 4}
	if got := FloatDistance(NormL2, a, b); !approx(got, 5, 1e-12) {
		t.Fatalf("NormL2 = %v, want 5", got)
	}
	if got := FloatDistance(NormL2Sqr, a, b); !approx(got, 25, 1e-12) {
		t.Fatalf("NormL2Sqr = %v, want 25", got)
	}
	if got := FloatDistance(NormL1, a, b); !approx(got, 7, 1e-12) {
		t.Fatalf("NormL1 = %v, want 7", got)
	}
}

func TestHammingDistance(t *testing.T) {
	a := []byte{0b0000_1111, 0b1010_1010}
	b := []byte{0b0000_0000, 0b0101_0101}
	// 4 differing bits in the first byte, 8 in the second.
	if got := HammingDistance(a, b); got != 12 {
		t.Fatalf("HammingDistance = %d, want 12", got)
	}
}

func TestNormalizeL2(t *testing.T) {
	d := []float64{3, 4}
	got := NormalizeL2(d)
	var s float64
	for _, x := range got {
		s += x * x
	}
	if !approx(math.Sqrt(s), 1, 1e-12) {
		t.Fatalf("NormalizeL2 not unit length: %v", got)
	}
	// Zero vector is returned unchanged.
	if z := NormalizeL2([]float64{0, 0}); z[0] != 0 || z[1] != 0 {
		t.Fatalf("NormalizeL2 zero = %v", z)
	}
}
