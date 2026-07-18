package matching2

import "testing"

func TestMat3MulIdentity(t *testing.T) {
	a := [3][3]float64{{1, 2, 3}, {4, 5, 6}, {7, 8, 10}}
	got := Mat3Mul(a, Mat3Identity())
	if got != a {
		t.Fatalf("A*I = %v, want %v", got, a)
	}
}

func TestMat3Inverse(t *testing.T) {
	a := [3][3]float64{{4, 7, 2}, {3, 6, 1}, {2, 5, 3}}
	inv, ok := Mat3Inverse(a)
	if !ok {
		t.Fatal("expected invertible")
	}
	prod := Mat3Mul(a, inv)
	id := Mat3Identity()
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			if !approx(prod[i][j], id[i][j], 1e-9) {
				t.Fatalf("A*A^-1 not identity: %v", prod)
			}
		}
	}
}

func TestMat3InverseSingular(t *testing.T) {
	singular := [3][3]float64{{1, 2, 3}, {2, 4, 6}, {1, 1, 1}}
	if _, ok := Mat3Inverse(singular); ok {
		t.Fatal("expected singular matrix to fail")
	}
}

func TestMat3DetAndTranspose(t *testing.T) {
	a := [3][3]float64{{1, 0, 0}, {0, 2, 0}, {0, 0, 3}}
	if got := Mat3Det(a); !approx(got, 6, 1e-12) {
		t.Fatalf("det = %v, want 6", got)
	}
	b := [3][3]float64{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}
	tr := Mat3Transpose(b)
	if tr[0][1] != 4 || tr[1][0] != 2 {
		t.Fatalf("transpose wrong: %v", tr)
	}
}

func TestNullVector(t *testing.T) {
	// Rows orthogonal to (1,1,1)/sqrt(3); the null space is that direction.
	a := [][]float64{
		{1, -1, 0},
		{0, 1, -1},
	}
	v := matching2nullVector(a)
	// v should be proportional to (1,1,1).
	if !approx(v[0], v[1], 1e-9) || !approx(v[1], v[2], 1e-9) {
		t.Fatalf("null vector not (1,1,1) direction: %v", v)
	}
}

func TestSVD3Reconstructs(t *testing.T) {
	a := [3][3]float64{{2, 0, 1}, {0, 3, 0}, {1, 0, 2}}
	u, s, v := matching2svd3(a)
	// Reconstruct U*diag(S)*V^T.
	d := [3][3]float64{{s[0], 0, 0}, {0, s[1], 0}, {0, 0, s[2]}}
	rec := Mat3Mul(u, Mat3Mul(d, Mat3Transpose(v)))
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			if !approx(rec[i][j], a[i][j], 1e-9) {
				t.Fatalf("SVD reconstruction wrong at %d,%d: %v vs %v", i, j, rec[i][j], a[i][j])
			}
		}
	}
	// Singular values are non-increasing.
	if s[0] < s[1] || s[1] < s[2] {
		t.Fatalf("singular values not ordered: %v", s)
	}
}
