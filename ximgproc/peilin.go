package ximgproc

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// Affine is a 2×3 affine transform in row-major form. The point (x, y) maps to
//
//	x' = M[0][0]·x + M[0][1]·y + M[0][2],
//	y' = M[1][0]·x + M[1][1]·y + M[1][2].
type Affine [2][3]float64

// Apply transforms the point (x, y) by the affine matrix and returns (x', y').
func (m Affine) Apply(x, y float64) (float64, float64) {
	return m[0][0]*x + m[0][1]*y + m[0][2], m[1][0]*x + m[1][1]*y + m[1][2]
}

// PeiLinNormalization computes the Pei–Lin normalization transform of img: the
// 2×3 affine matrix that maps the image into a canonical frame in which its
// intensity distribution has zero mean offset, isotropic (whitened) second-order
// moments and a fixed orientation, following Pei and Lin, "Image normalization
// for pattern recognition" (1995).
//
// The transform is built by (1) translating so the intensity centroid sits at
// the image centre, (2) applying the inverse square root of the second-moment
// (covariance) matrix so the shape becomes rotationally symmetric, and (3)
// rotating so the dominant third-moment direction points along +x, which
// resolves the remaining orientation ambiguity. Applying the returned [Affine]
// to pixel coordinates (e.g. with a warp) yields the normalized image; the
// transform maps the centroid exactly to the image centre.
//
// img may be 1- or 3-channel (colour is reduced to luma); pixel intensity acts
// as the moment mass. It panics if img has zero total intensity. Exact
// bit-for-bit parity with OpenCV's implementation (particularly the sign/tilt
// convention) is approximate and listed as deferred in the package
// documentation.
func PeiLinNormalization(img *cv.Mat) Affine {
	rows, cols := img.Rows, img.Cols
	g := toGrayFloat(img)

	// Raw moments.
	var m00, m10, m01 float64
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			w := g[y*cols+x]
			if w == 0 {
				continue
			}
			m00 += w
			m10 += w * float64(x)
			m01 += w * float64(y)
		}
	}
	if m00 == 0 {
		panic("ximgproc: PeiLinNormalization requires non-zero total intensity")
	}
	xc := m10 / m00
	yc := m01 / m00

	// Central second-order moments, normalised by total mass.
	var mu20, mu11, mu02 float64
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			w := g[y*cols+x]
			if w == 0 {
				continue
			}
			ddx := float64(x) - xc
			ddy := float64(y) - yc
			mu20 += w * ddx * ddx
			mu11 += w * ddx * ddy
			mu02 += w * ddy * ddy
		}
	}
	mu20 /= m00
	mu11 /= m00
	mu02 /= m00

	// Eigen-decomposition of the symmetric covariance [[mu20,mu11],[mu11,mu02]].
	theta := 0.5 * math.Atan2(2*mu11, mu20-mu02)
	common := math.Sqrt(math.Pow((mu20-mu02)/2, 2) + mu11*mu11)
	l1 := (mu20+mu02)/2 + common
	l2 := (mu20+mu02)/2 - common
	if l1 <= 0 {
		l1 = 1
	}
	if l2 <= 0 {
		l2 = 1
	}
	ct, st := math.Cos(theta), math.Sin(theta)
	is1 := 1.0 / math.Sqrt(l1)
	is2 := 1.0 / math.Sqrt(l2)
	// W = R · diag(is1,is2) · Rᵀ  (inverse square root of the covariance).
	w00 := ct*ct*is1 + st*st*is2
	w01 := ct * st * (is1 - is2)
	w10 := w01
	w11 := st*st*is1 + ct*ct*is2

	// Third-moment orientation fix, evaluated in the whitened frame.
	var t30, t21, t12, t03 float64
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			w := g[y*cols+x]
			if w == 0 {
				continue
			}
			ddx := float64(x) - xc
			ddy := float64(y) - yc
			u := w00*ddx + w01*ddy
			v := w10*ddx + w11*ddy
			t30 += w * u * u * u
			t21 += w * u * u * v
			t12 += w * u * v * v
			t03 += w * v * v * v
		}
	}
	phi := math.Atan2(t03+t21, t30+t12)
	cp, sp := math.Cos(-phi), math.Sin(-phi)
	// Rfix · W.
	a00 := cp*w00 - sp*w10
	a01 := cp*w01 - sp*w11
	a10 := sp*w00 + cp*w10
	a11 := sp*w01 + cp*w11

	// Translate the centroid to the image centre.
	cxCen := float64(cols) / 2
	cyCen := float64(rows) / 2
	tx := cxCen - (a00*xc + a01*yc)
	ty := cyCen - (a10*xc + a11*yc)

	return Affine{
		{a00, a01, tx},
		{a10, a11, ty},
	}
}
