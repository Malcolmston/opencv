package structured_light

import (
	"math"
	"math/cmplx"

	cv "github.com/malcolmston/opencv"
)

// dft computes the (unnormalized) discrete Fourier transform of x by direct
// summation. It is O(n²), which is ample for the single-line transforms used by
// Fourier-transform profilometry and keeps the package dependency-free (no FFT
// library). The output has the same length as x.
func dft(x []complex128) []complex128 {
	n := len(x)
	out := make([]complex128, n)
	for k := 0; k < n; k++ {
		var s complex128
		w := -2 * math.Pi * float64(k) / float64(n)
		for t := 0; t < n; t++ {
			s += x[t] * cmplx.Rect(1, w*float64(t))
		}
		out[k] = s
	}
	return out
}

// idft inverts [dft], including the 1/n normalization, so that idft(dft(x)) == x
// up to floating-point error.
func idft(x []complex128) []complex128 {
	n := len(x)
	out := make([]complex128, n)
	inv := complex(1/float64(n), 0)
	for k := 0; k < n; k++ {
		var s complex128
		w := 2 * math.Pi * float64(k) / float64(n)
		for t := 0; t < n; t++ {
			s += x[t] * cmplx.Rect(1, w*float64(t))
		}
		out[k] = s * inv
	}
	return out
}

// detectCarrier returns the positive-frequency bin (in [1, n/2)) of maximum
// magnitude in signal, i.e. the dominant fringe carrier. It returns 1 for a
// signal too short to have a carrier bin.
func detectCarrier(signal []float64) int {
	n := len(signal)
	if n < 4 {
		return 1
	}
	x := make([]complex128, n)
	for i, v := range signal {
		x[i] = complex(v, 0)
	}
	spec := dft(x)
	best, bestMag := 1, -1.0
	for k := 1; k < n/2; k++ {
		if m := cmplx.Abs(spec[k]); m > bestMag {
			bestMag, best = m, k
		}
	}
	return best
}

// ftpLine extracts the wrapped phase of one real fringe line by the classic
// Fourier-transform-profilometry filter: transform the line, keep only the
// positive-frequency sideband in [carrier-band, carrier+band] (which discards
// the DC term and the conjugate lobe, forming an analytic signal), invert, and
// take the argument. The returned phase has the length of signal and includes
// the carrier ramp.
func ftpLine(signal []float64, carrier, band int) []float64 {
	n := len(signal)
	x := make([]complex128, n)
	for i, v := range signal {
		x[i] = complex(v, 0)
	}
	spec := dft(x)
	filtered := make([]complex128, n)
	lo := carrier - band
	if lo < 1 {
		lo = 1
	}
	hi := carrier + band
	if hi > n-1 {
		hi = n - 1
	}
	for k := lo; k <= hi; k++ {
		filtered[k] = spec[k]
	}
	y := idft(filtered)
	ph := make([]float64, n)
	for i := range y {
		ph[i] = math.Atan2(imag(y[i]), real(y[i]))
	}
	return ph
}

// FTPWrappedPhase decodes a single fringe image into a wrapped phase map using
// Fourier-transform profilometry — a one-shot alternative to multi-image
// phase shifting. The carrier fringe frequency is detected automatically from a
// central line and a proportional sideband is retained. When horizontal is
// false the fringes are vertical and each image row is transformed along x;
// when true each column is transformed along y. img may be single- or
// multi-channel (reduced to luma). The result is a row-major []float64 in
// (-π, π]; unwrap it (e.g. with [UnwrapPhaseMap] or [QualityGuidedUnwrap]) to
// recover absolute phase.
func FTPWrappedPhase(img *cv.Mat, horizontal bool) []float64 {
	g := toGray(img)
	rows, cols := g.Rows, g.Cols
	out := make([]float64, rows*cols)
	if !horizontal {
		line := make([]float64, cols)
		mid := rows / 2
		for x := 0; x < cols; x++ {
			line[x] = float64(g.Data[mid*cols+x])
		}
		carrier := detectCarrier(line)
		band := carrier / 2
		if band < 1 {
			band = 1
		}
		for y := 0; y < rows; y++ {
			for x := 0; x < cols; x++ {
				line[x] = float64(g.Data[y*cols+x])
			}
			ph := ftpLine(line, carrier, band)
			for x := 0; x < cols; x++ {
				out[y*cols+x] = ph[x]
			}
		}
		return out
	}
	line := make([]float64, rows)
	mid := cols / 2
	for y := 0; y < rows; y++ {
		line[y] = float64(g.Data[y*cols+mid])
	}
	carrier := detectCarrier(line)
	band := carrier / 2
	if band < 1 {
		band = 1
	}
	for x := 0; x < cols; x++ {
		for y := 0; y < rows; y++ {
			line[y] = float64(g.Data[y*cols+x])
		}
		ph := ftpLine(line, carrier, band)
		for y := 0; y < rows; y++ {
			out[y*cols+x] = ph[y]
		}
	}
	return out
}

// FTPWrappedPhaseBand is [FTPWrappedPhase] with an explicit carrier bin and
// sideband half-width instead of automatic detection, for callers that know the
// projected fringe frequency or need a tighter band to reject harmonics. carrier
// and band are in cycles-per-line; band must be at least 1. Orientation and the
// output convention match [FTPWrappedPhase]. It panics if carrier<1 or band<1.
func FTPWrappedPhaseBand(img *cv.Mat, horizontal bool, carrier, band int) []float64 {
	if carrier < 1 || band < 1 {
		panic("structured_light: FTPWrappedPhaseBand requires carrier>=1 and band>=1")
	}
	g := toGray(img)
	rows, cols := g.Rows, g.Cols
	out := make([]float64, rows*cols)
	if !horizontal {
		line := make([]float64, cols)
		for y := 0; y < rows; y++ {
			for x := 0; x < cols; x++ {
				line[x] = float64(g.Data[y*cols+x])
			}
			ph := ftpLine(line, carrier, band)
			for x := 0; x < cols; x++ {
				out[y*cols+x] = ph[x]
			}
		}
		return out
	}
	line := make([]float64, rows)
	for x := 0; x < cols; x++ {
		for y := 0; y < rows; y++ {
			line[y] = float64(g.Data[y*cols+x])
		}
		ph := ftpLine(line, carrier, band)
		for y := 0; y < rows; y++ {
			out[y*cols+x] = ph[y]
		}
	}
	return out
}
