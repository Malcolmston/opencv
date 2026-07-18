package moments2

import (
	"math"
	"math/cmplx"

	cv "github.com/malcolmston/opencv"
)

// ResampleContour resamples a closed contour to exactly n points spaced equally
// along its arc length, returning the new points as [cv.Point2f]. It panics if n
// is less than one or the contour has fewer than two points.
func ResampleContour(contour []cv.Point, n int) []cv.Point2f {
	if n < 1 {
		panic("moments2: ResampleContour requires n >= 1")
	}
	if len(contour) < 2 {
		panic("moments2: ResampleContour requires at least two points")
	}
	m := len(contour)
	// Cumulative arc length around the closed contour.
	cum := make([]float64, m+1)
	for i := 0; i < m; i++ {
		j := (i + 1) % m
		dx := float64(contour[j].X - contour[i].X)
		dy := float64(contour[j].Y - contour[i].Y)
		cum[i+1] = cum[i] + math.Hypot(dx, dy)
	}
	total := cum[m]
	out := make([]cv.Point2f, n)
	if total == 0 {
		for i := range out {
			out[i] = cv.Point2f{X: float64(contour[0].X), Y: float64(contour[0].Y)}
		}
		return out
	}
	step := total / float64(n)
	seg := 0
	for i := 0; i < n; i++ {
		target := float64(i) * step
		for seg < m && cum[seg+1] < target {
			seg++
		}
		if seg >= m {
			seg = m - 1
		}
		segLen := cum[seg+1] - cum[seg]
		t := 0.0
		if segLen > 0 {
			t = (target - cum[seg]) / segLen
		}
		a := contour[seg]
		b := contour[(seg+1)%m]
		out[i] = cv.Point2f{
			X: float64(a.X) + t*float64(b.X-a.X),
			Y: float64(a.Y) + t*float64(b.Y-a.Y),
		}
	}
	return out
}

// FourierDescriptors returns the discrete Fourier transform of a closed contour
// whose points are treated as complex numbers x+iy. The returned slice has the
// same length as the input and encodes the boundary shape; low-index
// coefficients capture coarse structure and high-index ones fine detail. It
// panics on an empty contour.
func FourierDescriptors(contour []cv.Point2f) []complex128 {
	n := len(contour)
	if n == 0 {
		panic("moments2: FourierDescriptors on empty contour")
	}
	z := make([]complex128, n)
	for i, p := range contour {
		z[i] = complex(p.X, p.Y)
	}
	out := make([]complex128, n)
	for k := 0; k < n; k++ {
		var acc complex128
		for t := 0; t < n; t++ {
			angle := -2 * math.Pi * float64(k) * float64(t) / float64(n)
			acc += z[t] * cmplx.Rect(1, angle)
		}
		out[k] = acc
	}
	return out
}

// FourierMagnitudeSpectrum returns the moduli of a sequence of Fourier
// descriptors, discarding phase.
func FourierMagnitudeSpectrum(fd []complex128) []float64 {
	out := make([]float64, len(fd))
	for i, c := range fd {
		out[i] = cmplx.Abs(c)
	}
	return out
}

// NormalizedFourierDescriptors returns a translation-, scale-, rotation- and
// start-point-invariant descriptor derived from fd. Translation invariance
// comes from dropping the zero-frequency term; scale invariance from dividing by
// the magnitude of the first non-zero descriptor; and rotation and start-point
// invariance from taking magnitudes. The result has length len(fd)-1 with the
// leading element equal to 1. It returns nil for fewer than two descriptors.
func NormalizedFourierDescriptors(fd []complex128) []float64 {
	n := len(fd)
	if n < 2 {
		return nil
	}
	norm := cmplx.Abs(fd[1])
	out := make([]float64, n-1)
	if norm == 0 {
		return out
	}
	for k := 1; k < n; k++ {
		out[k-1] = cmplx.Abs(fd[k]) / norm
	}
	return out
}

// FourierDescriptorDistance returns the Euclidean distance between two
// normalized Fourier descriptor vectors, comparing only the overlapping leading
// coefficients when the lengths differ.
func FourierDescriptorDistance(a, b []float64) float64 {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	var sum float64
	for i := 0; i < n; i++ {
		d := a[i] - b[i]
		sum += d * d
	}
	return math.Sqrt(sum)
}

// ReconstructContour approximates a contour from its Fourier descriptors,
// keeping only the numDescriptors lowest-frequency coefficients (paired from the
// low and high ends of the spectrum) and evaluating the inverse transform at
// numPoints equally spaced parameter values. It is the inverse companion of
// [FourierDescriptors] and is useful for visualising how many harmonics a shape
// requires. It panics if numPoints is less than one.
func ReconstructContour(fd []complex128, numPoints, numDescriptors int) []cv.Point2f {
	if numPoints < 1 {
		panic("moments2: ReconstructContour requires numPoints >= 1")
	}
	n := len(fd)
	if n == 0 {
		return make([]cv.Point2f, numPoints)
	}
	if numDescriptors < 1 {
		numDescriptors = 1
	}
	if numDescriptors > n {
		numDescriptors = n
	}
	// Frequencies to keep: index 0..half from the low end and their negative
	// counterparts from the high end of the spectrum.
	keep := make([]bool, n)
	keep[0] = true
	for k := 1; k < numDescriptors; k++ {
		keep[k%n] = true
		keep[(n-k)%n] = true
	}
	out := make([]cv.Point2f, numPoints)
	for i := 0; i < numPoints; i++ {
		t := float64(i) / float64(numPoints) * float64(n)
		var acc complex128
		for k := 0; k < n; k++ {
			if !keep[k] {
				continue
			}
			angle := 2 * math.Pi * float64(k) * t / float64(n)
			acc += fd[k] * cmplx.Rect(1, angle)
		}
		acc /= complex(float64(n), 0)
		out[i] = cv.Point2f{X: real(acc), Y: imag(acc)}
	}
	return out
}
