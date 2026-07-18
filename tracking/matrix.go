package tracking

import (
	"fmt"
	"math"
)

// Matrix is a small dense row-major matrix of float64 values. It provides the
// linear-algebra operations the recursive filters in this package need; it is
// not a general BLAS replacement but is exact and deterministic for the modest
// sizes used by [KalmanFilter].
type Matrix struct {
	// Rows is the number of rows.
	Rows int
	// Cols is the number of columns.
	Cols int
	// Data holds the elements in row-major order, length Rows*Cols.
	Data []float64
}

// NewMatrix allocates a zero-filled matrix with the given shape. It panics if
// either dimension is not positive.
func NewMatrix(rows, cols int) *Matrix {
	if rows <= 0 || cols <= 0 {
		panic("tracking: NewMatrix requires positive dimensions")
	}
	return &Matrix{Rows: rows, Cols: cols, Data: make([]float64, rows*cols)}
}

// IdentityMatrix returns the n-by-n identity matrix.
func IdentityMatrix(n int) *Matrix {
	m := NewMatrix(n, n)
	for i := 0; i < n; i++ {
		m.Data[i*n+i] = 1
	}
	return m
}

// MatrixFromRows builds a matrix from a slice of equal-length rows. It panics if
// rows is empty or ragged.
func MatrixFromRows(rows [][]float64) *Matrix {
	if len(rows) == 0 || len(rows[0]) == 0 {
		panic("tracking: MatrixFromRows requires a non-empty grid")
	}
	c := len(rows[0])
	m := NewMatrix(len(rows), c)
	for i, row := range rows {
		if len(row) != c {
			panic("tracking: MatrixFromRows requires rectangular input")
		}
		copy(m.Data[i*c:(i+1)*c], row)
	}
	return m
}

// At returns the element at (r, c). It panics if the indices are out of range.
func (m *Matrix) At(r, c int) float64 {
	if r < 0 || r >= m.Rows || c < 0 || c >= m.Cols {
		panic("tracking: Matrix.At out of range")
	}
	return m.Data[r*m.Cols+c]
}

// Set stores v at (r, c). It panics if the indices are out of range.
func (m *Matrix) Set(r, c int, v float64) {
	if r < 0 || r >= m.Rows || c < 0 || c >= m.Cols {
		panic("tracking: Matrix.Set out of range")
	}
	m.Data[r*m.Cols+c] = v
}

// Clone returns an independent copy of the matrix.
func (m *Matrix) Clone() *Matrix {
	out := NewMatrix(m.Rows, m.Cols)
	copy(out.Data, m.Data)
	return out
}

// Mul returns the matrix product m*o. It panics if the inner dimensions differ.
func (m *Matrix) Mul(o *Matrix) *Matrix {
	if m.Cols != o.Rows {
		panic(fmt.Sprintf("tracking: Matrix.Mul dimension mismatch %dx%d * %dx%d", m.Rows, m.Cols, o.Rows, o.Cols))
	}
	out := NewMatrix(m.Rows, o.Cols)
	for i := 0; i < m.Rows; i++ {
		for k := 0; k < m.Cols; k++ {
			a := m.Data[i*m.Cols+k]
			if a == 0 {
				continue
			}
			for j := 0; j < o.Cols; j++ {
				out.Data[i*o.Cols+j] += a * o.Data[k*o.Cols+j]
			}
		}
	}
	return out
}

// Add returns the element-wise sum m+o. It panics if the shapes differ.
func (m *Matrix) Add(o *Matrix) *Matrix {
	m.requireSameShape(o, "Add")
	out := NewMatrix(m.Rows, m.Cols)
	for i := range m.Data {
		out.Data[i] = m.Data[i] + o.Data[i]
	}
	return out
}

// Sub returns the element-wise difference m-o. It panics if the shapes differ.
func (m *Matrix) Sub(o *Matrix) *Matrix {
	m.requireSameShape(o, "Sub")
	out := NewMatrix(m.Rows, m.Cols)
	for i := range m.Data {
		out.Data[i] = m.Data[i] - o.Data[i]
	}
	return out
}

// Scale returns the matrix with every element multiplied by s.
func (m *Matrix) Scale(s float64) *Matrix {
	out := NewMatrix(m.Rows, m.Cols)
	for i := range m.Data {
		out.Data[i] = m.Data[i] * s
	}
	return out
}

// Transpose returns the transpose of the matrix.
func (m *Matrix) Transpose() *Matrix {
	out := NewMatrix(m.Cols, m.Rows)
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			out.Data[j*m.Rows+i] = m.Data[i*m.Cols+j]
		}
	}
	return out
}

// Inverse returns the inverse of a square matrix, computed by Gauss-Jordan
// elimination with partial pivoting. It panics if the matrix is not square or is
// singular.
func (m *Matrix) Inverse() *Matrix {
	if m.Rows != m.Cols {
		panic("tracking: Matrix.Inverse requires a square matrix")
	}
	n := m.Rows
	// Augmented [m | I].
	a := make([][]float64, n)
	for i := 0; i < n; i++ {
		a[i] = make([]float64, 2*n)
		copy(a[i][:n], m.Data[i*n:(i+1)*n])
		a[i][n+i] = 1
	}
	for col := 0; col < n; col++ {
		// Partial pivot.
		piv := col
		best := math.Abs(a[col][col])
		for r := col + 1; r < n; r++ {
			if v := math.Abs(a[r][col]); v > best {
				best = v
				piv = r
			}
		}
		if best < 1e-15 {
			panic("tracking: Matrix.Inverse of a singular matrix")
		}
		a[col], a[piv] = a[piv], a[col]
		pv := a[col][col]
		for j := 0; j < 2*n; j++ {
			a[col][j] /= pv
		}
		for r := 0; r < n; r++ {
			if r == col {
				continue
			}
			f := a[r][col]
			if f == 0 {
				continue
			}
			for j := 0; j < 2*n; j++ {
				a[r][j] -= f * a[col][j]
			}
		}
	}
	out := NewMatrix(n, n)
	for i := 0; i < n; i++ {
		copy(out.Data[i*n:(i+1)*n], a[i][n:])
	}
	return out
}

// requireSameShape panics if o does not have the same shape as m.
func (m *Matrix) requireSameShape(o *Matrix, who string) {
	if m.Rows != o.Rows || m.Cols != o.Cols {
		panic("tracking: Matrix." + who + " shape mismatch")
	}
}
