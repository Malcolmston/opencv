package colorspaces2

import (
	cv "github.com/malcolmston/opencv"
)

// Matrix3 is a 3x3 matrix of float64 in row-major order, used for the linear
// colour transforms in this package (RGB<->XYZ and chromatic adaptation).
type Matrix3 [3][3]float64

// MulVec multiplies the matrix by the column vector (v.X, v.Y, v.Z) and returns
// the resulting [XYZ] triple.
func (m Matrix3) MulVec(v XYZ) XYZ {
	return XYZ{
		X: m[0][0]*v.X + m[0][1]*v.Y + m[0][2]*v.Z,
		Y: m[1][0]*v.X + m[1][1]*v.Y + m[1][2]*v.Z,
		Z: m[2][0]*v.X + m[2][1]*v.Y + m[2][2]*v.Z,
	}
}

// Mul returns the matrix product m*n.
func (m Matrix3) Mul(n Matrix3) Matrix3 {
	var out Matrix3
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			out[i][j] = m[i][0]*n[0][j] + m[i][1]*n[1][j] + m[i][2]*n[2][j]
		}
	}
	return out
}

// Transpose returns the transpose of the matrix.
func (m Matrix3) Transpose() Matrix3 {
	var out Matrix3
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			out[i][j] = m[j][i]
		}
	}
	return out
}

// Inverse returns the matrix inverse and true, or the zero matrix and false if
// the matrix is singular.
func (m Matrix3) Inverse() (Matrix3, bool) {
	a, b, c := m[0][0], m[0][1], m[0][2]
	d, e, f := m[1][0], m[1][1], m[1][2]
	g, h, i := m[2][0], m[2][1], m[2][2]
	det := a*(e*i-f*h) - b*(d*i-f*g) + c*(d*h-e*g)
	if det == 0 {
		return Matrix3{}, false
	}
	inv := 1 / det
	var out Matrix3
	out[0][0] = (e*i - f*h) * inv
	out[0][1] = (c*h - b*i) * inv
	out[0][2] = (b*f - c*e) * inv
	out[1][0] = (f*g - d*i) * inv
	out[1][1] = (a*i - c*g) * inv
	out[1][2] = (c*d - a*f) * inv
	out[2][0] = (d*h - e*g) * inv
	out[2][1] = (b*g - a*h) * inv
	out[2][2] = (a*e - b*d) * inv
	return out, true
}

// Standard reference white points in CIE XYZ (Y normalised to 1.0).
var (
	// WhitePointD65 is the CIE D65 daylight illuminant (the sRGB white point).
	WhitePointD65 = XYZ{X: 0.95047, Y: 1.0, Z: 1.08883}
	// WhitePointD50 is the CIE D50 illuminant used by the ICC profile
	// connection space.
	WhitePointD50 = XYZ{X: 0.96422, Y: 1.0, Z: 0.82521}
	// WhitePointA is the CIE A illuminant (incandescent / tungsten, ~2856K).
	WhitePointA = XYZ{X: 1.09850, Y: 1.0, Z: 0.35585}
	// WhitePointE is the equal-energy illuminant E.
	WhitePointE = XYZ{X: 1.0, Y: 1.0, Z: 1.0}
)

// Cone-response matrices for chromatic adaptation.
var (
	bradfordMat = Matrix3{
		{0.8951000, 0.2664000, -0.1614000},
		{-0.7502000, 1.7135000, 0.0367000},
		{0.0389000, -0.0685000, 1.0296000},
	}
	vonKriesMat = Matrix3{
		{0.4002400, 0.7076000, -0.0808100},
		{-0.2263000, 1.1653200, 0.0457000},
		{0.0000000, 0.0000000, 0.9182200},
	}
)

// adaptationMatrix builds the linear XYZ->XYZ chromatic-adaptation matrix for a
// given cone-response matrix, source white and destination white.
func adaptationMatrix(cone Matrix3, src, dst XYZ) Matrix3 {
	coneInv, ok := cone.Inverse()
	if !ok {
		return Matrix3{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}
	}
	s := cone.MulVec(src)
	d := cone.MulVec(dst)
	diag := Matrix3{
		{safeRatio(d.X, s.X), 0, 0},
		{0, safeRatio(d.Y, s.Y), 0},
		{0, 0, safeRatio(d.Z, s.Z)},
	}
	return coneInv.Mul(diag).Mul(cone)
}

// safeRatio returns a/b, or 1 when b is zero.
func safeRatio(a, b float64) float64 {
	if b == 0 {
		return 1
	}
	return a / b
}

// BradfordMatrix returns the Bradford chromatic-adaptation matrix that maps XYZ
// colours viewed under the source white to their appearance under the
// destination white.
func BradfordMatrix(srcWhite, dstWhite XYZ) Matrix3 {
	return adaptationMatrix(bradfordMat, srcWhite, dstWhite)
}

// VonKriesMatrix returns the von Kries (HPE cone) chromatic-adaptation matrix
// mapping from the source white to the destination white.
func VonKriesMatrix(srcWhite, dstWhite XYZ) Matrix3 {
	return adaptationMatrix(vonKriesMat, srcWhite, dstWhite)
}

// BradfordAdapt adapts a single [XYZ] colour from the source white point to the
// destination white point using the Bradford transform.
func BradfordAdapt(c, srcWhite, dstWhite XYZ) XYZ {
	return BradfordMatrix(srcWhite, dstWhite).MulVec(c)
}

// VonKriesAdapt adapts a single [XYZ] colour from the source white point to the
// destination white point using the von Kries transform.
func VonKriesAdapt(c, srcWhite, dstWhite XYZ) XYZ {
	return VonKriesMatrix(srcWhite, dstWhite).MulVec(c)
}

// ChromaticAdaptMat returns a new Mat in which every RGB pixel of src has been
// Bradford-adapted from srcWhite to dstWhite (converting through XYZ). The
// input is treated as sRGB and the result is re-encoded as sRGB.
func ChromaticAdaptMat(src *cv.Mat, srcWhite, dstWhite XYZ) *cv.Mat {
	colorspaces2RequireRGB(src, "ChromaticAdaptMat")
	m := BradfordMatrix(srcWhite, dstWhite)
	dst := cv.NewMat(src.Rows, src.Cols, 3)
	for y := 0; y < src.Rows; y++ {
		for x := 0; x < src.Cols; x++ {
			in := colorspaces2ReadRGB(src, y, x)
			out := XYZToRGB(m.MulVec(RGBToXYZ(in)))
			colorspaces2WriteRGB(dst, y, x, out)
		}
	}
	return dst
}
