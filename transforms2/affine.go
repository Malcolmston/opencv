package transforms2

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// An affine transform is represented by the parent package's [cv.AffineMatrix],
// a 2x3 matrix [a b tx; c d ty] stored row-major that maps a source point
// (X, Y) to (a*X + b*Y + tx, c*X + d*Y + ty).

// AffineIdentity returns the affine transform that leaves points unchanged.
func AffineIdentity() cv.AffineMatrix {
	return cv.AffineMatrix{1, 0, 0, 0, 1, 0}
}

// AffineTranslation returns the affine transform that shifts points by
// (tx, ty).
func AffineTranslation(tx, ty float64) cv.AffineMatrix {
	return cv.AffineMatrix{1, 0, tx, 0, 1, ty}
}

// AffineScaling returns the affine transform that scales points by sx along x
// and sy along y about the origin.
func AffineScaling(sx, sy float64) cv.AffineMatrix {
	return cv.AffineMatrix{sx, 0, 0, 0, sy, 0}
}

// AffineRotation returns the affine transform that rotates points about the
// origin by angleDeg degrees (counter-clockwise in OpenCV's convention).
func AffineRotation(angleDeg float64) cv.AffineMatrix {
	r := angleDeg * math.Pi / 180
	c := math.Cos(r)
	s := math.Sin(r)
	return cv.AffineMatrix{c, s, 0, -s, c, 0}
}

// AffineRotationAround returns the affine transform that rotates about the
// point (cx, cy) by angleDeg degrees (counter-clockwise) and uniformly scales
// by scale, matching cv.GetRotationMatrix2D.
func AffineRotationAround(cx, cy, angleDeg, scale float64) cv.AffineMatrix {
	r := angleDeg * math.Pi / 180
	a := scale * math.Cos(r)
	b := scale * math.Sin(r)
	return cv.AffineMatrix{a, b, (1-a)*cx - b*cy, -b, a, b*cx + (1-a)*cy}
}

// AffineShear returns the affine transform that shears by shx along x (as a
// function of y) and shy along y (as a function of x).
func AffineShear(shx, shy float64) cv.AffineMatrix {
	return cv.AffineMatrix{1, shx, 0, shy, 1, 0}
}

// ApplyAffine maps the point (x, y) through the affine transform m.
func ApplyAffine(m cv.AffineMatrix, x, y float64) (float64, float64) {
	return m[0]*x + m[1]*y + m[2], m[3]*x + m[4]*y + m[5]
}

// ComposeAffine returns the affine transform equivalent to applying b first and
// then a (a after b).
func ComposeAffine(a, b cv.AffineMatrix) cv.AffineMatrix {
	return cv.AffineMatrix{
		a[0]*b[0] + a[1]*b[3], a[0]*b[1] + a[1]*b[4], a[0]*b[2] + a[1]*b[5] + a[2],
		a[3]*b[0] + a[4]*b[3], a[3]*b[1] + a[4]*b[4], a[3]*b[2] + a[4]*b[5] + a[5],
	}
}

// InvertAffine returns the inverse of m and reports whether its 2x2 linear part
// is invertible.
func InvertAffine(m cv.AffineMatrix) (cv.AffineMatrix, bool) {
	det := m[0]*m[4] - m[1]*m[3]
	if math.Abs(det) < 1e-15 {
		return cv.AffineMatrix{}, false
	}
	id := 1 / det
	i0 := m[4] * id
	i1 := -m[1] * id
	i3 := -m[3] * id
	i4 := m[0] * id
	i2 := -(i0*m[2] + i1*m[5])
	i5 := -(i3*m[2] + i4*m[5])
	return cv.AffineMatrix{i0, i1, i2, i3, i4, i5}, true
}

// GetAffineTransform computes the affine transform that maps the three source
// points src to the three destination points dst. The three points must not be
// collinear. It panics if the system is singular.
func GetAffineTransform(src, dst [3]cv.Point2f) cv.AffineMatrix {
	a := [3][3]float64{
		{src[0].X, src[0].Y, 1},
		{src[1].X, src[1].Y, 1},
		{src[2].X, src[2].Y, 1},
	}
	bx := [3]float64{dst[0].X, dst[1].X, dst[2].X}
	by := [3]float64{dst[0].Y, dst[1].Y, dst[2].Y}
	rx, ok1 := transforms2solve3(a, bx)
	ry, ok2 := transforms2solve3(a, by)
	if !ok1 || !ok2 {
		panic("transforms2: GetAffineTransform source points are collinear")
	}
	return cv.AffineMatrix{rx[0], rx[1], rx[2], ry[0], ry[1], ry[2]}
}

// WarpAffine applies the affine transform m to src, producing a width x height
// output. Each destination pixel is inverse-mapped into src and resampled with
// the chosen interpolation and border handling. It panics if m is not
// invertible or the size is non-positive.
func WarpAffine(src *cv.Mat, m cv.AffineMatrix, width, height int, interp Interpolation, border BorderMode, fill float64) *cv.Mat {
	inv, ok := InvertAffine(m)
	if !ok {
		panic("transforms2: WarpAffine transform is not invertible")
	}
	return transforms2warpInverse(src, width, height, interp, border, fill, func(x, y float64) (float64, float64) {
		return ApplyAffine(inv, x, y)
	})
}
