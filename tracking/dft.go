package tracking

import "math"

// ComplexMat is a dense, row-major matrix of complex128 samples. It is the
// working type for the Fourier-domain correlation filters in this package
// ([TrackerMOSSE], [TrackerDCF], [TrackerKCFHOG] and [TrackerCSRT]). The value
// for row y, column x is at index y*Cols+x.
//
// Unlike [cv.Mat] (8-bit) and [cv.FloatMat] (real), a ComplexMat can hold the
// output of a discrete Fourier transform, whose values are complex. Build one
// with [NewComplexMat] or [RealToComplex] and transform it with [FFT2] / [IFFT2].
type ComplexMat struct {
	// Rows is the number of rows (height).
	Rows int
	// Cols is the number of columns (width).
	Cols int
	// Data holds Rows*Cols samples in row-major order.
	Data []complex128
}

// NewComplexMat allocates a zero-filled ComplexMat. It panics if a dimension is
// not positive.
func NewComplexMat(rows, cols int) *ComplexMat {
	if rows <= 0 || cols <= 0 {
		panic("tracking: NewComplexMat requires positive dimensions")
	}
	return &ComplexMat{Rows: rows, Cols: cols, Data: make([]complex128, rows*cols)}
}

// At returns the sample at row y, column x.
func (c *ComplexMat) At(y, x int) complex128 { return c.Data[y*c.Cols+x] }

// Clone returns a deep copy with its own backing storage.
func (c *ComplexMat) Clone() *ComplexMat {
	out := NewComplexMat(c.Rows, c.Cols)
	copy(out.Data, c.Data)
	return out
}

// RealToComplex lifts a real grid (length rows*cols, row-major) into a
// ComplexMat with zero imaginary parts. It panics if len(real) != rows*cols.
func RealToComplex(real []float64, rows, cols int) *ComplexMat {
	if len(real) != rows*cols {
		panic("tracking: RealToComplex length mismatch")
	}
	out := NewComplexMat(rows, cols)
	for i, v := range real {
		out.Data[i] = complex(v, 0)
	}
	return out
}

// Real returns the real parts of the matrix as a fresh row-major slice.
func (c *ComplexMat) Real() []float64 {
	out := make([]float64, len(c.Data))
	for i, v := range c.Data {
		out[i] = real(v)
	}
	return out
}

// NextPow2 returns the smallest power of two that is >= n (and at least 1). It
// is used to pad correlation-filter model sizes so the radix-2 [FFT2] applies.
func NextPow2(n int) int {
	p := 1
	for p < n {
		p <<= 1
	}
	return p
}

// isPow2 reports whether n is a positive power of two.
func isPow2(n int) bool { return n > 0 && n&(n-1) == 0 }

// fft1d performs an in-place radix-2 Cooley-Tukey FFT (or inverse FFT when
// inverse is true, including the 1/n scaling) on a, whose length must be a power
// of two.
func fft1d(a []complex128, inverse bool) {
	n := len(a)
	// Bit-reversal permutation.
	for i, j := 1, 0; i < n; i++ {
		bit := n >> 1
		for ; j&bit != 0; bit >>= 1 {
			j ^= bit
		}
		j ^= bit
		if i < j {
			a[i], a[j] = a[j], a[i]
		}
	}
	for length := 2; length <= n; length <<= 1 {
		ang := 2 * math.Pi / float64(length)
		if !inverse {
			ang = -ang
		}
		wlen := complex(math.Cos(ang), math.Sin(ang))
		for i := 0; i < n; i += length {
			w := complex(1, 0)
			half := length >> 1
			for k := 0; k < half; k++ {
				u := a[i+k]
				v := a[i+k+half] * w
				a[i+k] = u + v
				a[i+k+half] = u - v
				w *= wlen
			}
		}
	}
	if inverse {
		inv := complex(1/float64(n), 0)
		for i := range a {
			a[i] *= inv
		}
	}
}

// transform2d applies a separable 2D FFT (rows then columns) to a copy of m. Its
// dimensions must both be powers of two.
func transform2d(m *ComplexMat, inverse bool) *ComplexMat {
	if !isPow2(m.Rows) || !isPow2(m.Cols) {
		panic("tracking: FFT2 requires power-of-two dimensions (use NextPow2)")
	}
	out := m.Clone()
	row := make([]complex128, out.Cols)
	for y := 0; y < out.Rows; y++ {
		copy(row, out.Data[y*out.Cols:(y+1)*out.Cols])
		fft1d(row, inverse)
		copy(out.Data[y*out.Cols:(y+1)*out.Cols], row)
	}
	col := make([]complex128, out.Rows)
	for x := 0; x < out.Cols; x++ {
		for y := 0; y < out.Rows; y++ {
			col[y] = out.Data[y*out.Cols+x]
		}
		fft1d(col, inverse)
		for y := 0; y < out.Rows; y++ {
			out.Data[y*out.Cols+x] = col[y]
		}
	}
	return out
}

// FFT2 returns the forward 2D discrete Fourier transform of m. Both dimensions
// must be powers of two (see [NextPow2]); it panics otherwise.
func FFT2(m *ComplexMat) *ComplexMat { return transform2d(m, false) }

// IFFT2 returns the inverse 2D discrete Fourier transform of m, including the
// 1/(rows*cols) normalisation, so IFFT2(FFT2(x)) == x up to rounding. Both
// dimensions must be powers of two.
func IFFT2(m *ComplexMat) *ComplexMat { return transform2d(m, true) }

// HannWindow2D returns a rows×cols separable Hann (raised-cosine) window in
// row-major order. Multiplying a patch by this window before an FFT suppresses
// the boundary discontinuity that circular correlation would otherwise treat as
// a step edge; it is standard preprocessing for MOSSE/KCF-style filters.
func HannWindow2D(rows, cols int) []float64 {
	hy := make([]float64, rows)
	for y := 0; y < rows; y++ {
		if rows == 1 {
			hy[y] = 1
		} else {
			hy[y] = 0.5 * (1 - math.Cos(2*math.Pi*float64(y)/float64(rows-1)))
		}
	}
	hx := make([]float64, cols)
	for x := 0; x < cols; x++ {
		if cols == 1 {
			hx[x] = 1
		} else {
			hx[x] = 0.5 * (1 - math.Cos(2*math.Pi*float64(x)/float64(cols-1)))
		}
	}
	out := make([]float64, rows*cols)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			out[y*cols+x] = hy[y] * hx[x]
		}
	}
	return out
}

// GaussianResponse returns a rows×cols real grid holding a 2D Gaussian of
// standard deviation sigma peaking (value 1) at the grid centre, in row-major
// order. It is the desired correlation output ("regression target") for a MOSSE
// filter, whose peak marks the object centre.
func GaussianResponse(rows, cols int, sigma float64) []float64 {
	cx := float64(cols-1) / 2
	cy := float64(rows-1) / 2
	out := make([]float64, rows*cols)
	den := 2 * sigma * sigma
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			dx := float64(x) - cx
			dy := float64(y) - cy
			out[y*cols+x] = math.Exp(-(dx*dx + dy*dy) / den)
		}
	}
	return out
}

// gaussianResponseOrigin is like [GaussianResponse] but the peak sits at the
// top-left (0,0) with the grid treated as circular (fftshift-free). It is the
// regression target for the kernelised KCF filters, whose zero-lag correlation
// sits at index 0.
func gaussianResponseOrigin(rows, cols int, sigma float64) []float64 {
	out := make([]float64, rows*cols)
	den := 2 * sigma * sigma
	for y := 0; y < rows; y++ {
		dy := float64(y)
		if y > rows/2 {
			dy = float64(y - rows)
		}
		for x := 0; x < cols; x++ {
			dx := float64(x)
			if x > cols/2 {
				dx = float64(x - cols)
			}
			out[y*cols+x] = math.Exp(-(dx*dx + dy*dy) / den)
		}
	}
	return out
}
