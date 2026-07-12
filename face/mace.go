package face

import (
	"math"
	"math/cmplx"

	cv "github.com/malcolmston/opencv"
)

// This file implements the Minimum Average Correlation Energy (MACE) filter,
// which OpenCV exposes as cv::face::MACE. MACE synthesises a single correlation
// filter from several training images of one subject so that correlating an
// authentic image with the filter produces a sharp, delta-like peak, whereas an
// impostor yields a diffuse response. Authenticity is judged by the
// peak-to-sidelobe ratio (PSR) of the correlation output. Everything — the 2D
// DFT, the complex linear algebra and the filter synthesis — is built here from
// the standard library.
//
// Filter synthesis follows the closed form of Mahalanobis, Kumar & Casasent
// (1987): with X the column matrix of vectorised image spectra, D the diagonal
// average power-spectrum matrix and u the vector of desired peak values (ones),
//
//	H = D⁻¹·X·(Xᴴ·D⁻¹·X)⁻¹·u
//
// The small inversion is over the (T×T) matrix for T training images.

// MACE is a synthesised Minimum Average Correlation Energy correlation filter.
// Construct one with [NewMACE] and fit it with [MACE.Train]; the zero value is
// not usable.
type MACE struct {
	size    int          // images are resampled to size×size before transforming
	filter  []complex128 // conjugated frequency-domain filter, length size*size
	trained bool
	psrThr  float64 // authenticity threshold on the PSR
}

// NewMACE returns an untrained MACE filter that operates on size×size images
// (every training and query image is reduced to luma and resampled to this
// square geometry). size must be positive; powers of two are fastest but any
// size works. The default authenticity threshold is a PSR of 20, adjustable
// with [MACE.SetThreshold].
func NewMACE(size int) *MACE {
	if size < 2 {
		panic("face: NewMACE size must be >= 2")
	}
	return &MACE{size: size, psrThr: 20}
}

// SetThreshold sets the peak-to-sidelobe ratio above which [MACE.Same] reports a
// query as authentic.
func (m *MACE) SetThreshold(psr float64) { m.psrThr = psr }

// Train synthesises the correlation filter from one subject's images. Each image
// is reduced to luma, resampled to size×size, DFT-transformed and used as a
// constraint that the correlation peak at the origin equals one while the
// average correlation energy is minimised. It panics if given no images.
func (m *MACE) Train(images []*cv.Mat) {
	if len(images) == 0 {
		panic("face: MACE.Train requires at least one image")
	}
	n := m.size * m.size
	t := len(images)

	// X: n×t matrix of vectorised, centred spectra (column j = image j).
	X := make([][]complex128, n)
	for i := 0; i < n; i++ {
		X[i] = make([]complex128, t)
	}
	for j, img := range images {
		spec := m.spectrum(img)
		for i := 0; i < n; i++ {
			X[i][j] = spec[i]
		}
	}

	// D: diagonal average power spectrum, D_ii = (1/t) Σ_j |X_ij|² (+ε).
	D := make([]float64, n)
	for i := 0; i < n; i++ {
		var p float64
		for j := 0; j < t; j++ {
			a := X[i][j]
			p += real(a)*real(a) + imag(a)*imag(a)
		}
		D[i] = p/float64(t) + 1e-6
	}

	// Dinv·X (n×t).
	DiX := make([][]complex128, n)
	for i := 0; i < n; i++ {
		DiX[i] = make([]complex128, t)
		inv := complex(1/D[i], 0)
		for j := 0; j < t; j++ {
			DiX[i][j] = inv * X[i][j]
		}
	}

	// G = Xᴴ·Dinv·X  (t×t, Hermitian).
	G := make([][]complex128, t)
	for a := 0; a < t; a++ {
		G[a] = make([]complex128, t)
		for b := 0; b < t; b++ {
			var s complex128
			for i := 0; i < n; i++ {
				s += cmplx.Conj(X[i][a]) * DiX[i][b]
			}
			G[a][b] = s
		}
	}
	// Regularise the diagonal for a stable inverse.
	for a := 0; a < t; a++ {
		G[a][a] += complex(1e-6, 0)
	}

	Ginv, ok := invertComplex(G)
	if !ok {
		// Fall back to a matched filter (average spectrum) if G is singular.
		m.filter = make([]complex128, n)
		for i := 0; i < n; i++ {
			var s complex128
			for j := 0; j < t; j++ {
				s += X[i][j]
			}
			m.filter[i] = cmplx.Conj(s / complex(float64(t), 0))
		}
		m.trained = true
		return
	}

	// c = Ginv·u, u = ones(t).
	c := make([]complex128, t)
	for a := 0; a < t; a++ {
		var s complex128
		for b := 0; b < t; b++ {
			s += Ginv[a][b]
		}
		c[a] = s
	}

	// H = Dinv·X·c  (length n). Store the conjugate so correlation is a simple
	// per-frequency product in [MACE.correlate].
	m.filter = make([]complex128, n)
	for i := 0; i < n; i++ {
		var s complex128
		for j := 0; j < t; j++ {
			s += DiX[i][j] * c[j]
		}
		m.filter[i] = cmplx.Conj(s)
	}
	m.trained = true
}

// PSR correlates img with the trained filter and returns the peak-to-sidelobe
// ratio of the correlation plane: the correlation peak minus the mean of the
// surrounding sidelobe region, divided by that region's standard deviation. A
// high PSR indicates an authentic match. It panics if the filter is untrained.
func (m *MACE) PSR(img *cv.Mat) float64 {
	if !m.trained {
		panic("face: MACE.PSR before Train")
	}
	plane := m.correlate(img)
	return peakToSidelobe(plane, m.size)
}

// Same reports whether img is authentic for this filter, i.e. its [MACE.PSR]
// exceeds the threshold set by [MACE.SetThreshold] (default 20).
func (m *MACE) Same(img *cv.Mat) bool {
	return m.PSR(img) >= m.psrThr
}

// spectrum reduces img to luma, resamples it to size×size, energy-normalises it
// and returns its centred 2D DFT as a length size*size vector.
func (m *MACE) spectrum(img *cv.Mat) []complex128 {
	v := imageVector(img, m.size, m.size)
	// Remove the DC level and unit-normalise energy so brightness/contrast do
	// not dominate the synthesis or the later correlation.
	var mean float64
	for _, x := range v {
		mean += x
	}
	mean /= float64(len(v))
	var energy float64
	for i := range v {
		v[i] -= mean
		energy += v[i] * v[i]
	}
	if energy > 1e-12 {
		inv := 1 / math.Sqrt(energy)
		for i := range v {
			v[i] *= inv
		}
	}
	spatial := make([]complex128, len(v))
	for i, x := range v {
		spatial[i] = complex(x, 0)
	}
	return dft2D(spatial, m.size, m.size, false)
}

// correlate multiplies the query spectrum by the (conjugated) filter and
// inverse-transforms, returning the real correlation plane of length size*size.
func (m *MACE) correlate(img *cv.Mat) []float64 {
	spec := m.spectrum(img)
	prod := make([]complex128, len(spec))
	for i := range spec {
		prod[i] = spec[i] * m.filter[i]
	}
	inv := dft2D(prod, m.size, m.size, true)
	out := make([]float64, len(inv))
	for i, c := range inv {
		out[i] = real(c)
	}
	return out
}

// peakToSidelobe finds the correlation peak and returns (peak − μ)/σ where μ and
// σ are the mean and standard deviation over the plane excluding an exclusion
// window around the peak.
func peakToSidelobe(plane []float64, size int) float64 {
	peakIdx := 0
	peakVal := math.Inf(-1)
	for i, v := range plane {
		av := math.Abs(v)
		if av > peakVal {
			peakVal = av
			peakIdx = i
		}
	}
	py := peakIdx / size
	px := peakIdx % size
	const exclude = 2
	var sum, sumSq float64
	var n int
	for y := 0; y < size; y++ {
		for x := 0; x < size; x++ {
			if abs(y-py) <= exclude && abs(x-px) <= exclude {
				continue
			}
			v := plane[y*size+x]
			sum += v
			sumSq += v * v
			n++
		}
	}
	if n == 0 {
		return 0
	}
	mean := sum / float64(n)
	varc := sumSq/float64(n) - mean*mean
	if varc < 1e-12 {
		return 0
	}
	return (peakVal - mean) / math.Sqrt(varc)
}

func abs(a int) int {
	if a < 0 {
		return -a
	}
	return a
}

// dft2D computes the (optionally inverse) 2D discrete Fourier transform of a
// rows×cols complex field stored row-major, by applying a 1D DFT along the rows
// and then the columns. The transform is O((rows·cols)·(rows+cols)); it is exact
// for any size and needs no power-of-two constraint, which keeps the correlation
// filter usable on arbitrary square patches.
func dft2D(in []complex128, rows, cols int, inverse bool) []complex128 {
	tmp := make([]complex128, len(in))
	// DFT each row.
	rowBuf := make([]complex128, cols)
	for y := 0; y < rows; y++ {
		copy(rowBuf, in[y*cols:(y+1)*cols])
		r := dft1D(rowBuf, inverse)
		copy(tmp[y*cols:(y+1)*cols], r)
	}
	// DFT each column.
	out := make([]complex128, len(in))
	colBuf := make([]complex128, rows)
	for x := 0; x < cols; x++ {
		for y := 0; y < rows; y++ {
			colBuf[y] = tmp[y*cols+x]
		}
		c := dft1D(colBuf, inverse)
		for y := 0; y < rows; y++ {
			out[y*cols+x] = c[y]
		}
	}
	return out
}

// dft1D computes the 1D DFT (or inverse DFT, scaled by 1/N) of x directly.
func dft1D(x []complex128, inverse bool) []complex128 {
	n := len(x)
	out := make([]complex128, n)
	sign := -1.0
	if inverse {
		sign = 1.0
	}
	for k := 0; k < n; k++ {
		var s complex128
		base := sign * 2 * math.Pi * float64(k) / float64(n)
		for t := 0; t < n; t++ {
			ang := base * float64(t)
			s += x[t] * cmplx.Rect(1, ang)
		}
		out[k] = s
	}
	if inverse {
		scale := complex(1/float64(n), 0)
		for k := range out {
			out[k] *= scale
		}
	}
	return out
}

// invertComplex inverts the square complex matrix a via Gauss–Jordan
// elimination with partial pivoting, reporting ok=false when a is singular.
func invertComplex(a [][]complex128) ([][]complex128, bool) {
	n := len(a)
	if n == 0 {
		return nil, false
	}
	m := make([][]complex128, n)
	for i := 0; i < n; i++ {
		m[i] = make([]complex128, 2*n)
		copy(m[i], a[i])
		m[i][n+i] = 1
	}
	for col := 0; col < n; col++ {
		pivot := col
		best := cmplx.Abs(m[col][col])
		for r := col + 1; r < n; r++ {
			if v := cmplx.Abs(m[r][col]); v > best {
				best = v
				pivot = r
			}
		}
		if best < 1e-15 {
			return nil, false
		}
		m[col], m[pivot] = m[pivot], m[col]
		inv := 1 / m[col][col]
		for c := 0; c < 2*n; c++ {
			m[col][c] *= inv
		}
		for r := 0; r < n; r++ {
			if r == col {
				continue
			}
			f := m[r][col]
			if f == 0 {
				continue
			}
			for c := 0; c < 2*n; c++ {
				m[r][c] -= f * m[col][c]
			}
		}
	}
	out := make([][]complex128, n)
	for i := 0; i < n; i++ {
		out[i] = make([]complex128, n)
		copy(out[i], m[i][n:])
	}
	return out, true
}
