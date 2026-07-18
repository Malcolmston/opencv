package tracking

import "testing"

func TestMatrixMul(t *testing.T) {
	a := MatrixFromRows([][]float64{{1, 2}, {3, 4}})
	b := MatrixFromRows([][]float64{{5, 6}, {7, 8}})
	c := a.Mul(b)
	want := [][]float64{{19, 22}, {43, 50}}
	for i := 0; i < 2; i++ {
		for j := 0; j < 2; j++ {
			if !approx(c.At(i, j), want[i][j], 1e-9) {
				t.Fatalf("Mul[%d][%d] = %v, want %v", i, j, c.At(i, j), want[i][j])
			}
		}
	}
}

func TestMatrixInverse(t *testing.T) {
	m := MatrixFromRows([][]float64{
		{4, 7, 2},
		{3, 6, 1},
		{2, 5, 3},
	})
	inv := m.Inverse()
	prod := m.Mul(inv)
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			exp := 0.0
			if i == j {
				exp = 1
			}
			if !approx(prod.At(i, j), exp, 1e-9) {
				t.Fatalf("M*M^-1 [%d][%d] = %v, want %v", i, j, prod.At(i, j), exp)
			}
		}
	}
}

func TestMatrixTransposeAddScale(t *testing.T) {
	m := MatrixFromRows([][]float64{{1, 2, 3}, {4, 5, 6}})
	tr := m.Transpose()
	if tr.Rows != 3 || tr.Cols != 2 || tr.At(2, 1) != 6 {
		t.Fatalf("Transpose wrong: %+v", tr)
	}
	s := m.Scale(2)
	sum := m.Add(m)
	for i := range sum.Data {
		if !approx(sum.Data[i], s.Data[i], 1e-9) {
			t.Fatalf("Add != Scale(2) at %d", i)
		}
	}
	if !approx(m.Sub(m).Data[0], 0, 1e-9) {
		t.Fatalf("Sub self should be zero")
	}
}

func TestIdentityMatrix(t *testing.T) {
	id := IdentityMatrix(3)
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			exp := 0.0
			if i == j {
				exp = 1
			}
			if id.At(i, j) != exp {
				t.Fatalf("identity wrong at %d,%d", i, j)
			}
		}
	}
}
