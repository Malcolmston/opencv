package transforms2

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// A projective transform is represented by the parent package's
// [cv.PerspectiveMatrix], a 3x3 homography stored row-major that maps a point
// (X, Y) to (x'/w', y'/w') where (x', y', w') = H * (X, Y, 1).

// PerspectiveIdentity returns the homography that leaves points unchanged.
func PerspectiveIdentity() cv.PerspectiveMatrix {
	return cv.PerspectiveMatrix{1, 0, 0, 0, 1, 0, 0, 0, 1}
}

// AffineToPerspective embeds a 2x3 affine transform into a 3x3 homography.
func AffineToPerspective(a cv.AffineMatrix) cv.PerspectiveMatrix {
	return cv.PerspectiveMatrix{a[0], a[1], a[2], a[3], a[4], a[5], 0, 0, 1}
}

// ApplyPerspective maps the point (x, y) through the homography m. It returns
// the mapped point; when the homogeneous weight is zero it returns the point at
// infinity as (+Inf, +Inf).
func ApplyPerspective(m cv.PerspectiveMatrix, x, y float64) (float64, float64) {
	w := m[6]*x + m[7]*y + m[8]
	if w == 0 {
		return math.Inf(1), math.Inf(1)
	}
	return (m[0]*x + m[1]*y + m[2]) / w, (m[3]*x + m[4]*y + m[5]) / w
}

// ComposePerspective returns the homography equivalent to applying b first and
// then a (a after b), i.e. the matrix product a*b.
func ComposePerspective(a, b cv.PerspectiveMatrix) cv.PerspectiveMatrix {
	var r cv.PerspectiveMatrix
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			r[i*3+j] = a[i*3+0]*b[0*3+j] + a[i*3+1]*b[1*3+j] + a[i*3+2]*b[2*3+j]
		}
	}
	return r
}

// InvertPerspective returns the inverse homography and reports whether m is
// invertible.
func InvertPerspective(m cv.PerspectiveMatrix) (cv.PerspectiveMatrix, bool) {
	a, b, c := m[0], m[1], m[2]
	d, e, f := m[3], m[4], m[5]
	g, h, i := m[6], m[7], m[8]
	det := a*(e*i-f*h) - b*(d*i-f*g) + c*(d*h-e*g)
	if math.Abs(det) < 1e-15 {
		return cv.PerspectiveMatrix{}, false
	}
	id := 1 / det
	var inv cv.PerspectiveMatrix
	inv[0] = (e*i - f*h) * id
	inv[1] = (c*h - b*i) * id
	inv[2] = (b*f - c*e) * id
	inv[3] = (f*g - d*i) * id
	inv[4] = (a*i - c*g) * id
	inv[5] = (c*d - a*f) * id
	inv[6] = (d*h - e*g) * id
	inv[7] = (b*g - a*h) * id
	inv[8] = (a*e - b*d) * id
	return inv, true
}

// GetPerspectiveTransform computes the homography mapping the four source points
// src to the four destination points dst, in corresponding order and with no
// three collinear. It panics if the resulting linear system is singular.
func GetPerspectiveTransform(src, dst [4]cv.Point2f) cv.PerspectiveMatrix {
	a := make([][]float64, 8)
	b := make([]float64, 8)
	for i := 0; i < 4; i++ {
		X, Y := src[i].X, src[i].Y
		u, v := dst[i].X, dst[i].Y
		a[2*i] = []float64{X, Y, 1, 0, 0, 0, -u * X, -u * Y}
		b[2*i] = u
		a[2*i+1] = []float64{0, 0, 0, X, Y, 1, -v * X, -v * Y}
		b[2*i+1] = v
	}
	h, ok := transforms2solve(a, b)
	if !ok {
		panic("transforms2: GetPerspectiveTransform points are degenerate")
	}
	return cv.PerspectiveMatrix{h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], 1}
}

// WarpPerspective applies the homography m to src, producing a width x height
// output. Each destination pixel is inverse-mapped into src and resampled with
// the chosen interpolation and border handling. It panics if m is not
// invertible or the size is non-positive.
func WarpPerspective(src *cv.Mat, m cv.PerspectiveMatrix, width, height int, interp Interpolation, border BorderMode, fill float64) *cv.Mat {
	inv, ok := InvertPerspective(m)
	if !ok {
		panic("transforms2: WarpPerspective transform is not invertible")
	}
	return transforms2warpInverse(src, width, height, interp, border, fill, func(x, y float64) (float64, float64) {
		return ApplyPerspective(inv, x, y)
	})
}
