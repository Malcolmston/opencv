package imgprocx

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// PhaseCorrelate estimates the translational shift between two same-sized images
// a and b from the phase of their cross-power spectrum, mirroring
// cv2.phaseCorrelate. It converts both images to luminance, takes their discrete
// Fourier transforms, forms the normalised cross-power spectrum
//
//	R = (A · conj(B)) / |A · conj(B)|,
//
// and locates the peak of its inverse transform. The returned shift (dx, dy) is
// the translation that maps a onto b: a feature at (x, y) in a lies near
// (x+dx, y+dy) in b. response is the height of the correlation peak, in [0,1],
// and measures how sharply defined the shift is (near 1 for a clean translation).
//
// Because it uses the circular (periodic) Fourier transform, PhaseCorrelate
// recovers integer shifts exactly for images related by a circular shift. a and
// b must have identical dimensions; it panics otherwise.
func PhaseCorrelate(a, b *cv.Mat) (shift Point2f, response float64) {
	if a.Rows != b.Rows || a.Cols != b.Cols {
		panic("imgprocx: PhaseCorrelate requires images of equal size")
	}
	rows, cols := a.Rows, a.Cols
	fa := matToComplex(a)
	fb := matToComplex(b)
	dft2d(fa, rows, cols, false)
	dft2d(fb, rows, cols, false)
	// Cross-power spectrum, magnitude-normalised.
	r := make([]complex128, rows*cols)
	for i := range r {
		cross := fa[i] * conj(fb[i])
		mag := math.Hypot(real(cross), imag(cross))
		if mag < 1e-12 {
			r[i] = 0
			continue
		}
		r[i] = complex(real(cross)/mag, imag(cross)/mag)
	}
	dft2d(r, rows, cols, true)
	// Find the peak of the real part.
	peak := math.Inf(-1)
	var px, py int
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			v := real(r[y*cols+x])
			if v > peak {
				peak = v
				px, py = x, y
			}
		}
	}
	// Convert the wrapped peak index into a signed shift. The inverse transform
	// places the peak at (cols-dx, rows-dy), so a peak index beyond the halfway
	// point corresponds to a negative signed value; the shift that maps a onto b
	// is the negation of that signed value.
	sx := px
	if sx > cols/2 {
		sx -= cols
	}
	sy := py
	if sy > rows/2 {
		sy -= rows
	}
	return Point2f{X: float64(-sx), Y: float64(-sy)}, peak
}

// matToComplex builds a row-major complex slice from the luminance of img.
func matToComplex(img *cv.Mat) []complex128 {
	data, rows, cols := toGrayPlane(img)
	out := make([]complex128, rows*cols)
	for i := range out {
		out[i] = complex(data[i], 0)
	}
	return out
}

// conj returns the complex conjugate of z.
func conj(z complex128) complex128 {
	return complex(real(z), -imag(z))
}

// dft2d performs an in-place separable 2-D discrete Fourier transform of the
// rows×cols complex grid stored row-major in g. When inverse is true it computes
// the inverse transform (with 1/N normalisation).
func dft2d(g []complex128, rows, cols int, inverse bool) {
	// Transform each row.
	row := make([]complex128, cols)
	for y := 0; y < rows; y++ {
		copy(row, g[y*cols:y*cols+cols])
		dft1d(row, inverse)
		copy(g[y*cols:y*cols+cols], row)
	}
	// Transform each column.
	col := make([]complex128, rows)
	for x := 0; x < cols; x++ {
		for y := 0; y < rows; y++ {
			col[y] = g[y*cols+x]
		}
		dft1d(col, inverse)
		for y := 0; y < rows; y++ {
			g[y*cols+x] = col[y]
		}
	}
}

// dft1d performs an in-place naive discrete Fourier transform of x (O(n²)),
// which is ample for the small images used with phase correlation. When inverse
// is true it computes the inverse transform with 1/n normalisation.
func dft1d(x []complex128, inverse bool) {
	n := len(x)
	sign := -2 * math.Pi
	if inverse {
		sign = 2 * math.Pi
	}
	out := make([]complex128, n)
	for k := 0; k < n; k++ {
		var sum complex128
		for t := 0; t < n; t++ {
			ang := sign * float64(k) * float64(t) / float64(n)
			sum += x[t] * complex(math.Cos(ang), math.Sin(ang))
		}
		if inverse {
			sum /= complex(float64(n), 0)
		}
		out[k] = sum
	}
	copy(x, out)
}
