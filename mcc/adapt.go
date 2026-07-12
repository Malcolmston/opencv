package mcc

import "math"

// White points (CIE XYZ tristimulus, 2° standard observer, normalised so that
// Y=1). These are the illuminants used for chromatic adaptation and for the
// white-point argument of [XYZToLab] / [LabToXYZ].
//
//   - [WhiteD65] is the sRGB / Rec.709 daylight illuminant used everywhere else
//     in this package.
//   - [WhiteD50] is the ICC / print daylight illuminant.
//   - [WhiteA] is CIE illuminant A, a 2856 K tungsten source.
var (
	// WhiteD65 is CIE illuminant D65 (6504 K daylight).
	WhiteD65 = [3]float64{whiteX, whiteY, whiteZ}
	// WhiteD50 is CIE illuminant D50 (5003 K daylight, the ICC PCS white).
	WhiteD50 = [3]float64{0.96422, 1.00000, 0.82521}
	// WhiteA is CIE illuminant A (2856 K incandescent tungsten).
	WhiteA = [3]float64{1.09850, 1.00000, 0.35585}
)

// AdaptationMethod selects the cone-response model used by [ChromaticAdaptation]
// and [AdaptationMatrix] when converting a color from one white point to
// another.
type AdaptationMethod int

const (
	// Bradford is the spectrally-sharpened cone-response transform used by ICC
	// profiles; it is the most widely used and usually the most accurate.
	Bradford AdaptationMethod = iota
	// VonKries uses the Hunt–Pointer–Estévez cone primaries — the classical von
	// Kries coefficient law.
	VonKries
	// XYZScaling adapts by scaling XYZ directly (the trivial "wrong von Kries"
	// transform); it is provided for completeness and comparison.
	XYZScaling
)

// coneMatrix returns the 3x3 matrix M that maps CIE XYZ to the cone (LMS)
// response space for the given adaptation method.
func coneMatrix(m AdaptationMethod) [3][3]float64 {
	switch m {
	case Bradford:
		return [3][3]float64{
			{0.8951000, 0.2664000, -0.1614000},
			{-0.7502000, 1.7135000, 0.0367000},
			{0.0389000, -0.0685000, 1.0296000},
		}
	case VonKries:
		return [3][3]float64{
			{0.4002400, 0.7076000, -0.0808100},
			{-0.2263000, 1.1653200, 0.0457000},
			{0.0000000, 0.0000000, 0.9182200},
		}
	default: // XYZScaling
		return [3][3]float64{
			{1, 0, 0},
			{0, 1, 0},
			{0, 0, 1},
		}
	}
}

// AdaptationMatrix returns the 3x3 matrix that adapts a CIE XYZ color measured
// under the source white point to its appearance under the destination white
// point, using the given cone-response [AdaptationMethod]. Multiply an XYZ
// column vector by this matrix (see [ApplyMatrix3]) to adapt it. The matrix is
// exactly the identity when the two white points are equal.
func AdaptationMatrix(srcWhite, dstWhite [3]float64, method AdaptationMethod) [3][3]float64 {
	m := coneMatrix(method)
	mInv := invert3x3(m)
	src := ApplyMatrix3(m, srcWhite)
	dst := ApplyMatrix3(m, dstWhite)
	// Diagonal ratio of destination to source cone responses.
	d := [3][3]float64{
		{safeRatio(dst[0], src[0]), 0, 0},
		{0, safeRatio(dst[1], src[1]), 0},
		{0, 0, safeRatio(dst[2], src[2])},
	}
	return mat3Mul(mInv, mat3Mul(d, m))
}

// ChromaticAdaptation adapts a single CIE XYZ color from the source white point
// to the destination white point using the given [AdaptationMethod]. It is a
// convenience wrapper over [AdaptationMatrix] and [ApplyMatrix3].
func ChromaticAdaptation(xyz, srcWhite, dstWhite [3]float64, method AdaptationMethod) [3]float64 {
	return ApplyMatrix3(AdaptationMatrix(srcWhite, dstWhite, method), xyz)
}

// safeRatio returns a/b, guarding a zero denominator (which only arises for a
// degenerate white point) by returning 1.
func safeRatio(a, b float64) float64 {
	if b == 0 {
		return 1
	}
	return a / b
}

// ApplyMatrix3 multiplies the 3x3 matrix m by the column vector v and returns
// the resulting 3-vector.
func ApplyMatrix3(m [3][3]float64, v [3]float64) [3]float64 {
	return [3]float64{
		m[0][0]*v[0] + m[0][1]*v[1] + m[0][2]*v[2],
		m[1][0]*v[0] + m[1][1]*v[1] + m[1][2]*v[2],
		m[2][0]*v[0] + m[2][1]*v[1] + m[2][2]*v[2],
	}
}

// mat3Mul returns the matrix product a*b of two 3x3 matrices.
func mat3Mul(a, b [3][3]float64) [3][3]float64 {
	var out [3][3]float64
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			out[i][j] = a[i][0]*b[0][j] + a[i][1]*b[1][j] + a[i][2]*b[2][j]
		}
	}
	return out
}

// invert3x3 returns the inverse of a 3x3 matrix via the adjugate/determinant.
// A singular matrix yields the zero matrix; the cone-response matrices used
// here are always invertible.
func invert3x3(m [3][3]float64) [3][3]float64 {
	a, b, c := m[0][0], m[0][1], m[0][2]
	d, e, f := m[1][0], m[1][1], m[1][2]
	g, h, i := m[2][0], m[2][1], m[2][2]
	det := a*(e*i-f*h) - b*(d*i-f*g) + c*(d*h-e*g)
	if math.Abs(det) < 1e-15 {
		return [3][3]float64{}
	}
	inv := 1 / det
	return [3][3]float64{
		{(e*i - f*h) * inv, (c*h - b*i) * inv, (b*f - c*e) * inv},
		{(f*g - d*i) * inv, (a*i - c*g) * inv, (c*d - a*f) * inv},
		{(d*h - e*g) * inv, (b*g - a*h) * inv, (a*e - b*d) * inv},
	}
}
