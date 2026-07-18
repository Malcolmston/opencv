package moments2

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// Moments holds the spatial (raw), central and normalized central moments of an
// image or contour up to the third order, mirroring OpenCV's cv::Moments.
//
// Spatial moments Mpq weight each unit of mass by x^p*y^q. Central moments Mupq
// are the same sums taken about the centroid and are therefore invariant to
// translation. Normalized central moments Nupq additionally divide by a power
// of M00 and are invariant to uniform scale.
type Moments struct {
	// Spatial (raw) moments.
	M00, M10, M01, M20, M11, M02, M30, M21, M12, M03 float64
	// Central moments about the centroid.
	Mu20, Mu11, Mu02, Mu30, Mu21, Mu12, Mu03 float64
	// Normalized central moments.
	Nu20, Nu11, Nu02, Nu30, Nu21, Nu12, Nu03 float64
}

// Centroid returns the mass-weighted centre (M10/M00, M01/M00) as a
// [cv.Point2f]. It returns the origin for a shape of zero total mass.
func (m Moments) Centroid() cv.Point2f {
	if m.M00 == 0 {
		return cv.Point2f{}
	}
	return cv.Point2f{X: m.M10 / m.M00, Y: m.M01 / m.M00}
}

// Area returns the zeroth spatial moment M00, which for a binary mask equals the
// number of foreground pixels and for a contour equals the enclosed area.
func (m Moments) Area() float64 { return m.M00 }

// completeCentralMoments fills in the central and normalized moments from the
// spatial moments already stored in m.
func (m *Moments) completeCentralMoments() {
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
	inv2 := 1 / (m.M00 * m.M00)
	s25 := inv2 / math.Sqrt(m.M00)
	m.Nu20 = m.Mu20 * inv2
	m.Nu11 = m.Mu11 * inv2
	m.Nu02 = m.Mu02 * inv2
	m.Nu30 = m.Mu30 * s25
	m.Nu21 = m.Mu21 * s25
	m.Nu12 = m.Mu12 * s25
	m.Nu03 = m.Mu03 * s25
}

// ImageMoments computes the full set of moments of a single-channel image,
// weighting each pixel (x, y) by its sample value. For a binary mask this yields
// the geometric moments of the foreground region. It panics if src is not
// single-channel.
func ImageMoments(src *cv.Mat) Moments {
	moments2requireGray(src, "ImageMoments")
	var m Moments
	for y := 0; y < src.Rows; y++ {
		fy := float64(y)
		row := y * src.Cols
		for x := 0; x < src.Cols; x++ {
			v := float64(src.Data[row+x])
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
	m.completeCentralMoments()
	return m
}

// MaskMoments computes moments treating every non-zero pixel as a unit of mass,
// ignoring the actual sample value. This is the correct choice for a binary
// segmentation where foreground pixels may not all equal 255. It panics if src
// is not single-channel.
func MaskMoments(src *cv.Mat) Moments {
	moments2requireGray(src, "MaskMoments")
	var m Moments
	for y := 0; y < src.Rows; y++ {
		fy := float64(y)
		row := y * src.Cols
		for x := 0; x < src.Cols; x++ {
			if src.Data[row+x] == 0 {
				continue
			}
			fx := float64(x)
			m.M00++
			m.M10 += fx
			m.M01 += fy
			m.M20 += fx * fx
			m.M11 += fx * fy
			m.M02 += fy * fy
			m.M30 += fx * fx * fx
			m.M21 += fx * fx * fy
			m.M12 += fx * fy * fy
			m.M03 += fy * fy * fy
		}
	}
	m.completeCentralMoments()
	return m
}

// ContourMoments computes the moments of the region enclosed by a polygon using
// Green's theorem, matching OpenCV's contour-based cv::moments. The polygon is
// given by its vertices in order; it is treated as closed. Fewer than three
// vertices yield a zero Moments value.
func ContourMoments(pts []cv.Point) Moments {
	var m Moments
	if len(pts) < 3 {
		return m
	}
	lpt := len(pts)
	var a00, a10, a01, a20, a11, a02, a30, a21, a12, a03 float64
	xiPrev := float64(pts[lpt-1].X)
	yiPrev := float64(pts[lpt-1].Y)
	xiPrev2 := xiPrev * xiPrev
	yiPrev2 := yiPrev * yiPrev
	for i := 0; i < lpt; i++ {
		xi := float64(pts[i].X)
		yi := float64(pts[i].Y)
		xi2 := xi * xi
		yi2 := yi * yi
		dxy := xiPrev*yi - xi*yiPrev
		xiiPrev := xiPrev + xi
		yiiPrev := yiPrev + yi
		a00 += dxy
		a10 += dxy * xiiPrev
		a01 += dxy * yiiPrev
		a20 += dxy * (xiPrev*xiiPrev + xi2)
		a11 += dxy * (xiPrev*(yiiPrev+yiPrev) + xi*(yiiPrev+yi))
		a02 += dxy * (yiPrev*yiiPrev + yi2)
		a30 += dxy * xiiPrev * (xiPrev2 + xi2)
		a03 += dxy * yiiPrev * (yiPrev2 + yi2)
		a21 += dxy * (xiPrev2*(3*yiPrev+yi) + 2*xi*xiPrev*yiiPrev + xi2*(yiPrev+3*yi))
		a12 += dxy * (yiPrev2*(3*xiPrev+xi) + 2*yi*yiPrev*xiiPrev + yi2*(xiPrev+3*xi))
		xiPrev = xi
		yiPrev = yi
		xiPrev2 = xi2
		yiPrev2 = yi2
	}
	if math.Abs(a00) < 1e-12 {
		return m
	}
	var db12, db16, db112, db124, db120, db160 float64
	if a00 > 0 {
		db12, db16, db112 = 0.5, 1.0/6.0, 1.0/12.0
		db124, db120, db160 = 1.0/24.0, 1.0/20.0, 1.0/60.0
	} else {
		db12, db16, db112 = -0.5, -1.0/6.0, -1.0/12.0
		db124, db120, db160 = -1.0/24.0, -1.0/20.0, -1.0/60.0
	}
	m.M00 = a00 * db12
	m.M10 = a10 * db16
	m.M01 = a01 * db16
	m.M20 = a20 * db112
	m.M11 = a11 * db124
	m.M02 = a02 * db112
	m.M30 = a30 * db120
	m.M21 = a21 * db160
	m.M12 = a12 * db160
	m.M03 = a03 * db120
	m.completeCentralMoments()
	return m
}

// RawMoment returns the spatial moment of order (p, q), sum of value*x^p*y^q
// over all pixels, for any non-negative p and q. It panics if src is not
// single-channel.
func RawMoment(src *cv.Mat, p, q int) float64 {
	moments2requireGray(src, "RawMoment")
	var sum float64
	for y := 0; y < src.Rows; y++ {
		yp := math.Pow(float64(y), float64(q))
		row := y * src.Cols
		for x := 0; x < src.Cols; x++ {
			v := float64(src.Data[row+x])
			if v == 0 {
				continue
			}
			sum += v * math.Pow(float64(x), float64(p)) * yp
		}
	}
	return sum
}

// CentralMoment returns the central moment of order (p, q), taken about the
// image centroid, for any non-negative p and q. It panics if src is not
// single-channel.
func CentralMoment(src *cv.Mat, p, q int) float64 {
	moments2requireGray(src, "CentralMoment")
	m00 := RawMoment(src, 0, 0)
	if m00 == 0 {
		return 0
	}
	cx := RawMoment(src, 1, 0) / m00
	cy := RawMoment(src, 0, 1) / m00
	var sum float64
	fp := float64(p)
	fq := float64(q)
	for y := 0; y < src.Rows; y++ {
		dy := math.Pow(float64(y)-cy, fq)
		row := y * src.Cols
		for x := 0; x < src.Cols; x++ {
			v := float64(src.Data[row+x])
			if v == 0 {
				continue
			}
			sum += v * math.Pow(float64(x)-cx, fp) * dy
		}
	}
	return sum
}

// NormalizedCentralMoment returns the normalized central moment of order (p, q),
// equal to mu_pq / M00^(1+(p+q)/2), which is invariant to uniform scaling. It
// panics if src is not single-channel.
func NormalizedCentralMoment(src *cv.Mat, p, q int) float64 {
	moments2requireGray(src, "NormalizedCentralMoment")
	m00 := RawMoment(src, 0, 0)
	if m00 == 0 {
		return 0
	}
	mu := CentralMoment(src, p, q)
	return mu / math.Pow(m00, 1+float64(p+q)/2)
}
