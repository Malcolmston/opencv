package contours2

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// ContourMoments computes the spatial, central and normalised central moments
// of a polygon (contour) using Green's theorem, matching the closed-form
// contour-moment formulas used by OpenCV's moments. M00 is the polygon area.
// The result is orientation-independent: the moments are normalised so that M00
// is non-negative regardless of the winding direction. Fewer than three points
// yield an all-zero result.
func ContourMoments(contour []cv.Point) Moments {
	n := len(contour)
	var m Moments
	if n < 3 {
		return m
	}
	var a00, a10, a01, a20, a11, a02, a30, a21, a12, a03 float64
	xi := float64(contour[n-1].X)
	yi := float64(contour[n-1].Y)
	for i := 0; i < n; i++ {
		xi1 := float64(contour[i].X)
		yi1 := float64(contour[i].Y)
		xii := xi * xi
		yii := yi * yi
		xi1i := xi1 * xi1
		yi1i := yi1 * yi1
		a := xi*yi1 - xi1*yi

		a00 += a
		a10 += a * (xi + xi1)
		a01 += a * (yi + yi1)
		a20 += a * (xii + xi*xi1 + xi1i)
		a11 += a * (2*xi*yi + xi*yi1 + xi1*yi + 2*xi1*yi1)
		a02 += a * (yii + yi*yi1 + yi1i)
		a30 += a * (xi + xi1) * (xii + xi1i)
		a21 += a * (xii*(3*yi+yi1) + 2*xi*xi1*(yi+yi1) + xi1i*(yi+3*yi1))
		a12 += a * (yii*(3*xi+xi1) + 2*yi*yi1*(xi+xi1) + yi1i*(xi+3*xi1))
		a03 += a * (yi + yi1) * (yii + yi1i)

		xi, yi = xi1, yi1
	}

	if a00 == 0 {
		return m
	}
	// Normalise winding so the area is positive.
	if a00 < 0 {
		a00, a10, a01, a20, a11, a02, a30, a21, a12, a03 =
			-a00, -a10, -a01, -a20, -a11, -a02, -a30, -a21, -a12, -a03
	}

	const (
		i2  = 1.0 / 2
		i6  = 1.0 / 6
		i12 = 1.0 / 12
		i24 = 1.0 / 24
		i20 = 1.0 / 20
		i60 = 1.0 / 60
	)
	m.M00 = a00 * i2
	m.M10 = a10 * i6
	m.M01 = a01 * i6
	m.M20 = a20 * i12
	m.M11 = a11 * i24
	m.M02 = a02 * i12
	m.M30 = a30 * i20
	m.M21 = a21 * i60
	m.M12 = a12 * i60
	m.M03 = a03 * i20

	contours2central(&m)
	return m
}

// ImageMoments computes the moments of a single-channel image, weighting each
// pixel (x, y) by its sample value. For a binary mask this yields the geometric
// moments of the white region. It panics if src is nil, empty or not
// single-channel. This mirrors OpenCV's moments called on a raster.
func ImageMoments(src *cv.Mat) Moments {
	contours2requireGray(src, "ImageMoments")
	var m Moments
	for y := 0; y < src.Rows; y++ {
		fy := float64(y)
		for x := 0; x < src.Cols; x++ {
			v := float64(src.Data[y*src.Cols+x])
			if v == 0 {
				continue
			}
			fx := float64(x)
			m.M00 += v
			m.M10 += fx * v
			m.M01 += fy * v
			m.M20 += fx * fx * v
			m.M11 += fx * fy * v
			m.M02 += fy * fy * v
			m.M30 += fx * fx * fx * v
			m.M21 += fx * fx * fy * v
			m.M12 += fx * fy * fy * v
			m.M03 += fy * fy * fy * v
		}
	}
	contours2central(&m)
	return m
}

// contours2central fills the central (Mu*) and normalised central (Nu*) moments
// from the spatial (M*) moments already stored in m.
func contours2central(m *Moments) {
	if m.M00 == 0 {
		return
	}
	cx := m.M10 / m.M00
	cy := m.M01 / m.M00
	m.Mu20 = m.M20 - cx*m.M10
	m.Mu11 = m.M11 - cx*m.M01
	m.Mu02 = m.M02 - cy*m.M01
	m.Mu30 = m.M30 - 3*cx*m.M20 + 2*cx*cx*m.M10
	m.Mu21 = m.M21 - 2*cx*m.M11 - cy*m.M20 + 2*cx*cx*m.M01
	m.Mu12 = m.M12 - 2*cy*m.M11 - cx*m.M02 + 2*cy*cy*m.M10
	m.Mu03 = m.M03 - 3*cy*m.M02 + 2*cy*cy*m.M01

	inv := 1 / (m.M00 * m.M00)
	inv25 := inv / math.Sqrt(math.Abs(m.M00))
	m.Nu20 = m.Mu20 * inv
	m.Nu11 = m.Mu11 * inv
	m.Nu02 = m.Mu02 * inv
	m.Nu30 = m.Mu30 * inv25
	m.Nu21 = m.Mu21 * inv25
	m.Nu12 = m.Mu12 * inv25
	m.Nu03 = m.Mu03 * inv25
}

// HuMoments returns the seven Hu invariant moments computed from a set of
// normalised central moments, matching OpenCV's HuMoments. They are invariant
// to translation, scale and rotation (the seventh also changes sign under
// reflection).
func HuMoments(m Moments) [7]float64 {
	var h [7]float64
	n20, n02, n11 := m.Nu20, m.Nu02, m.Nu11
	n30, n12, n21, n03 := m.Nu30, m.Nu12, m.Nu21, m.Nu03
	h[0] = n20 + n02
	h[1] = (n20-n02)*(n20-n02) + 4*n11*n11
	h[2] = (n30-3*n12)*(n30-3*n12) + (3*n21-n03)*(3*n21-n03)
	h[3] = (n30+n12)*(n30+n12) + (n21+n03)*(n21+n03)
	h[4] = (n30-3*n12)*(n30+n12)*((n30+n12)*(n30+n12)-3*(n21+n03)*(n21+n03)) +
		(3*n21-n03)*(n21+n03)*(3*(n30+n12)*(n30+n12)-(n21+n03)*(n21+n03))
	h[5] = (n20-n02)*((n30+n12)*(n30+n12)-(n21+n03)*(n21+n03)) +
		4*n11*(n30+n12)*(n21+n03)
	h[6] = (3*n21-n03)*(n30+n12)*((n30+n12)*(n30+n12)-3*(n21+n03)*(n21+n03)) -
		(n30-3*n12)*(n21+n03)*(3*(n30+n12)*(n30+n12)-(n21+n03)*(n21+n03))
	return h
}

// HuMoments returns the seven Hu invariant moments of the receiver's normalised
// central moments. It is the method form of the package-level [HuMoments].
func (m Moments) HuMoments() [7]float64 { return HuMoments(m) }

// MatchShapes compares two shapes given by their moments using one of the
// log-scaled Hu-moment metrics ([ContoursMatchI1], [ContoursMatchI2] or
// [ContoursMatchI3]). A smaller value means a closer match; identical shapes
// score 0. This mirrors OpenCV's matchShapes. It panics on an unknown method.
func MatchShapes(a, b Moments, method ShapeMatchMethod) float64 {
	return MatchShapesHu(HuMoments(a), HuMoments(b), method)
}

// MatchShapesHu compares two shapes directly from their Hu-moment vectors using
// the selected metric, matching OpenCV's matchShapes. It panics on an unknown
// method.
func MatchShapesHu(ha, hb [7]float64, method ShapeMatchMethod) float64 {
	var result float64
	for i := 0; i < 7; i++ {
		ma := contours2signedLog(ha[i])
		mb := contours2signedLog(hb[i])
		if ma == 0 || mb == 0 {
			continue
		}
		switch method {
		case ContoursMatchI1:
			result += math.Abs(1/ma - 1/mb)
		case ContoursMatchI2:
			result += math.Abs(ma - mb)
		case ContoursMatchI3:
			d := math.Abs(ma-mb) / math.Abs(ma)
			if d > result {
				result = d
			}
		default:
			panic("contours2: MatchShapes unknown method")
		}
	}
	return result
}

// contours2signedLog returns sign(v)*log10(|v|), used by the Hu-moment shape
// metrics. It returns 0 for a zero input.
func contours2signedLog(v float64) float64 {
	if v == 0 {
		return 0
	}
	l := math.Log10(math.Abs(v))
	if v < 0 {
		return -l
	}
	return l
}
