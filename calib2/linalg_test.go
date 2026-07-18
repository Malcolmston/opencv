package calib2

import (
	"math"
	"testing"
)

func TestMatrixMulKnown(t *testing.T) {
	a := NewMatrix(2, 3)
	copy(a.Data, []float64{1, 2, 3, 4, 5, 6})
	b := NewMatrix(3, 2)
	copy(b.Data, []float64{7, 8, 9, 10, 11, 12})
	c := a.Mul(b)
	// [[1,2,3],[4,5,6]] * [[7,8],[9,10],[11,12]] = [[58,64],[139,154]]
	want := []float64{58, 64, 139, 154}
	for i := range want {
		if math.Abs(c.Data[i]-want[i]) > 1e-12 {
			t.Fatalf("Mul = %v want %v", c.Data, want)
		}
	}
}

func TestMatrixTranspose(t *testing.T) {
	a := NewMatrix(2, 3)
	copy(a.Data, []float64{1, 2, 3, 4, 5, 6})
	tr := a.Transpose()
	if tr.Rows != 3 || tr.Cols != 2 {
		t.Fatalf("transpose shape %dx%d", tr.Rows, tr.Cols)
	}
	if tr.At(2, 1) != 6 || tr.At(0, 1) != 4 {
		t.Errorf("transpose values wrong")
	}
}

func TestSolveLinearKnown(t *testing.T) {
	// 2x + y = 5 ; x + 3y = 10 -> x=1, y=3.
	a := [][]float64{{2, 1}, {1, 3}}
	b := []float64{5, 10}
	x, ok := solveLinear(a, b)
	if !ok {
		t.Fatal("solve failed")
	}
	if math.Abs(x[0]-1) > 1e-12 || math.Abs(x[1]-3) > 1e-12 {
		t.Errorf("solve = %v want [1 3]", x)
	}
}

func TestSolveLinearSingular(t *testing.T) {
	a := [][]float64{{1, 2}, {2, 4}}
	b := []float64{3, 6}
	if _, ok := solveLinear(a, b); ok {
		t.Error("singular system reported solvable")
	}
}

func TestJacobiEigenKnown(t *testing.T) {
	// Diagonal-ish symmetric matrix with known eigenvalues 2 and 4 (plus 5).
	a := [][]float64{
		{3, 1, 0},
		{1, 3, 0},
		{0, 0, 5},
	}
	vals, vecs := jacobiEigen(a)
	// Eigenvalues of [[3,1],[1,3]] are 2 and 4.
	got := append([]float64{}, vals...)
	// Sort for comparison.
	for i := 0; i < len(got); i++ {
		for j := i + 1; j < len(got); j++ {
			if got[j] < got[i] {
				got[i], got[j] = got[j], got[i]
			}
		}
	}
	want := []float64{2, 4, 5}
	for i := range want {
		if math.Abs(got[i]-want[i]) > 1e-9 {
			t.Fatalf("eigenvalues %v want %v", got, want)
		}
	}
	// Reconstruct A = V diag Vᵀ.
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			var s float64
			for k := 0; k < 3; k++ {
				s += vecs[i][k] * vals[k] * vecs[j][k]
			}
			if math.Abs(s-a[i][j]) > 1e-9 {
				t.Fatalf("reconstruction mismatch at %d,%d: %g vs %g", i, j, s, a[i][j])
			}
		}
	}
}

func TestSVD3Reconstruction(t *testing.T) {
	a := [3][3]float64{
		{1, 2, 0},
		{0, 3, 1},
		{4, 0, 2},
	}
	u, s, v := svd3(a)
	// u diag(s) vᵀ == a
	var recon [3][3]float64
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			var acc float64
			for k := 0; k < 3; k++ {
				acc += u[i][k] * s[k] * v[j][k]
			}
			recon[i][j] = acc
		}
	}
	if mat3MaxDiff(recon, a) > 1e-9 {
		t.Fatalf("svd reconstruction diff %g", mat3MaxDiff(recon, a))
	}
	// Singular values descending and non-negative.
	if s[0] < s[1] || s[1] < s[2] || s[2] < 0 {
		t.Errorf("singular values not sorted/non-negative: %v", s)
	}
}

func TestIdentityMatrix(t *testing.T) {
	id := IdentityMatrix(3)
	if id.At(0, 0) != 1 || id.At(1, 1) != 1 || id.At(0, 1) != 0 {
		t.Error("identity wrong")
	}
}
