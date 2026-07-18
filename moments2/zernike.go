package moments2

import (
	"math"
	"math/cmplx"

	cv "github.com/malcolmston/opencv"
)

// RadialPolynomial evaluates the Zernike radial polynomial R_n^m(rho) for a
// radius rho in [0, 1]. It returns 0 when (n-|m|) is odd or |m| > n, the cases
// in which the polynomial is undefined or identically zero.
func RadialPolynomial(n, m int, rho float64) float64 {
	if m < 0 {
		m = -m
	}
	if m > n || (n-m)%2 != 0 {
		return 0
	}
	var r float64
	half := (n - m) / 2
	for k := 0; k <= half; k++ {
		num := math.Pow(-1, float64(k)) * moments2factorial(n-k)
		den := moments2factorial(k) *
			moments2factorial((n+m)/2-k) *
			moments2factorial((n-m)/2-k)
		r += (num / den) * math.Pow(rho, float64(n-2*k))
	}
	return r
}

// ZernikeCoefficient is a single complex Zernike moment A_nm together with its
// order n and repetition m.
type ZernikeCoefficient struct {
	// N is the radial order of the moment.
	N int
	// M is the azimuthal repetition of the moment.
	M int
	// Value is the complex moment A_nm.
	Value complex128
}

// Magnitude returns the modulus of the coefficient, which is invariant to
// rotation of the underlying image.
func (z ZernikeCoefficient) Magnitude() float64 { return cmplx.Abs(z.Value) }

// Phase returns the argument of the coefficient in radians.
func (z ZernikeCoefficient) Phase() float64 { return cmplx.Phase(z.Value) }

// ZernikeMoment computes the complex Zernike moment A_nm of a single-channel
// image over the unit disk inscribed in the image. Pixel intensities are the
// mass; coordinates are mapped so the largest image dimension spans the disk
// diameter and pixels outside the disk are ignored. It returns 0 when
// (n-|m|) is odd or |m| > n. It panics if src is not single-channel.
func ZernikeMoment(src *cv.Mat, n, m int) complex128 {
	moments2requireGray(src, "ZernikeMoment")
	am := m
	if am < 0 {
		am = -am
	}
	if am > n || (n-am)%2 != 0 {
		return 0
	}
	cx := float64(src.Cols-1) / 2
	cy := float64(src.Rows-1) / 2
	radius := float64(src.Cols)
	if src.Rows > src.Cols {
		radius = float64(src.Rows)
	}
	radius /= 2
	invR2 := 1 / (radius * radius)
	var acc complex128
	fm := float64(m)
	for y := 0; y < src.Rows; y++ {
		yn := (float64(y) - cy) / radius
		row := y * src.Cols
		for x := 0; x < src.Cols; x++ {
			v := float64(src.Data[row+x])
			if v == 0 {
				continue
			}
			xn := (float64(x) - cx) / radius
			rho := math.Hypot(xn, yn)
			if rho > 1 {
				continue
			}
			theta := math.Atan2(yn, xn)
			rad := RadialPolynomial(n, m, rho)
			acc += complex(v*rad, 0) * cmplx.Conj(cmplx.Rect(1, fm*theta))
		}
	}
	return complex(float64(n+1)/math.Pi, 0) * acc * complex(invR2, 0)
}

// ZernikeMagnitude returns the modulus of the Zernike moment A_nm, a rotation
// invariant shape feature. It panics if src is not single-channel.
func ZernikeMagnitude(src *cv.Mat, n, m int) float64 {
	return cmplx.Abs(ZernikeMoment(src, n, m))
}

// ZernikeMoments computes every valid Zernike moment up to and including radial
// order maxOrder, returning them ordered by n then by m from 0 to n. Only the
// non-negative repetitions are returned because A_n,-m is the complex conjugate
// of A_nm. It panics if src is not single-channel or maxOrder is negative.
func ZernikeMoments(src *cv.Mat, maxOrder int) []ZernikeCoefficient {
	moments2requireGray(src, "ZernikeMoments")
	if maxOrder < 0 {
		panic("moments2: ZernikeMoments requires maxOrder >= 0")
	}
	var out []ZernikeCoefficient
	for n := 0; n <= maxOrder; n++ {
		for m := 0; m <= n; m++ {
			if (n-m)%2 != 0 {
				continue
			}
			out = append(out, ZernikeCoefficient{N: n, M: m, Value: ZernikeMoment(src, n, m)})
		}
	}
	return out
}

// PseudoZernikeRadial evaluates the pseudo-Zernike radial polynomial R_n^m(rho)
// for a radius rho in [0, 1]. Unlike the Zernike polynomials these are defined
// for every m with |m| <= n regardless of the parity of (n-m). It returns 0
// when |m| > n.
func PseudoZernikeRadial(n, m int, rho float64) float64 {
	if m < 0 {
		m = -m
	}
	if m > n {
		return 0
	}
	var r float64
	for s := 0; s <= n-m; s++ {
		num := math.Pow(-1, float64(s)) * moments2factorial(2*n+1-s)
		den := moments2factorial(s) *
			moments2factorial(n-m-s) *
			moments2factorial(n+m+1-s)
		r += (num / den) * math.Pow(rho, float64(n-s))
	}
	return r
}

// PseudoZernikeMoment computes the complex pseudo-Zernike moment A_nm of a
// single-channel image over the inscribed unit disk. Pseudo-Zernike moments
// provide more low-order descriptors than ordinary Zernike moments and are more
// robust to noise. It returns 0 when |m| > n. It panics if src is not
// single-channel.
func PseudoZernikeMoment(src *cv.Mat, n, m int) complex128 {
	moments2requireGray(src, "PseudoZernikeMoment")
	am := m
	if am < 0 {
		am = -am
	}
	if am > n {
		return 0
	}
	cx := float64(src.Cols-1) / 2
	cy := float64(src.Rows-1) / 2
	radius := float64(src.Cols)
	if src.Rows > src.Cols {
		radius = float64(src.Rows)
	}
	radius /= 2
	invR2 := 1 / (radius * radius)
	var acc complex128
	fm := float64(m)
	for y := 0; y < src.Rows; y++ {
		yn := (float64(y) - cy) / radius
		row := y * src.Cols
		for x := 0; x < src.Cols; x++ {
			v := float64(src.Data[row+x])
			if v == 0 {
				continue
			}
			xn := (float64(x) - cx) / radius
			rho := math.Hypot(xn, yn)
			if rho > 1 {
				continue
			}
			theta := math.Atan2(yn, xn)
			rad := PseudoZernikeRadial(n, m, rho)
			acc += complex(v*rad, 0) * cmplx.Conj(cmplx.Rect(1, fm*theta))
		}
	}
	return complex(float64(n+1)/math.Pi, 0) * acc * complex(invR2, 0)
}
