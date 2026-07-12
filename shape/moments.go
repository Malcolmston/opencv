package shape

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// Shape-comparison methods for [MatchShapes], mirroring OpenCV's
// CONTOURS_MATCH_I1/I2/I3. Each compares the log-transformed Hu moments of two
// contours; smaller values mean more similar shapes.
const (
	// ContoursMatchI1 sums |1/m_i^A − 1/m_i^B| over the Hu moments.
	ContoursMatchI1 = 1
	// ContoursMatchI2 sums |m_i^A − m_i^B| over the Hu moments.
	ContoursMatchI2 = 2
	// ContoursMatchI3 takes the maximum of |m_i^A − m_i^B| / |m_i^A|.
	ContoursMatchI3 = 3
)

// ContourMoments computes the spatial, central and normalised central moments
// of a closed polygon (a contour) up to third order using Green's theorem, and
// returns them as a [cv.Moments]. Unlike cv.ImageMoments, which integrates over
// filled pixels, this integrates over the polygon defined by the contour's
// vertices, matching OpenCV's moments() applied to a point set.
//
// The polygon is treated as closed (the last vertex joins the first). Vertex
// winding does not affect the result: the contour is internally oriented so the
// zeroth moment (area) is non-negative. A contour with fewer than three points
// has zero area and yields a zero-value Moments.
func ContourMoments(contour []cv.Point) cv.Moments {
	n := len(contour)
	var m cv.Moments
	if n < 3 {
		return m
	}
	// Orient so the signed area is positive.
	pts := make([]fpoint, n)
	for i, p := range contour {
		pts[i] = fpoint{float64(p.X), float64(p.Y)}
	}
	if signedArea(pts) < 0 {
		for i, j := 0, n-1; i < j; i, j = i+1, j-1 {
			pts[i], pts[j] = pts[j], pts[i]
		}
	}

	var a00, a10, a01, a20, a11, a02, a30, a21, a12, a03 float64
	xPrev, yPrev := pts[n-1].x, pts[n-1].y
	for i := 0; i < n; i++ {
		xi, yi := pts[i].x, pts[i].y
		xi2, yi2 := xi*xi, yi*yi
		dxy := xPrev*yi - xi*yPrev
		xSum := xPrev + xi
		ySum := yPrev + yi
		a00 += dxy
		a10 += dxy * xSum
		a01 += dxy * ySum
		a20 += dxy * (xPrev*xSum + xi2)
		a11 += dxy * (xPrev*(ySum+yPrev) + xi*(ySum+yi))
		a02 += dxy * (yPrev*ySum + yi2)
		a30 += dxy * xSum * (xPrev*xPrev + xi2)
		a03 += dxy * ySum * (yPrev*yPrev + yi2)
		a21 += dxy * (xPrev*xPrev*(3*yPrev+yi) + 2*xPrev*xi*ySum + xi2*(yPrev+3*yi))
		a12 += dxy * (yPrev*yPrev*(3*xPrev+xi) + 2*yPrev*yi*xSum + yi2*(xPrev+3*xi))
		xPrev, yPrev = xi, yi
	}

	m.M00 = a00 / 2
	m.M10 = a10 / 6
	m.M01 = a01 / 6
	m.M20 = a20 / 12
	m.M11 = a11 / 24
	m.M02 = a02 / 12
	m.M30 = a30 / 20
	m.M21 = a21 / 60
	m.M12 = a12 / 60
	m.M03 = a03 / 20

	completeMoments(&m)
	return m
}

// signedArea returns twice the signed area of a polygon (positive for
// counter-clockwise winding in standard axes).
func signedArea(pts []fpoint) float64 {
	var a float64
	n := len(pts)
	for i := 0; i < n; i++ {
		j := (i + 1) % n
		a += pts[i].x*pts[j].y - pts[j].x*pts[i].y
	}
	return a
}

// completeMoments fills the central (Mu*) and normalised central (Nu*) moments
// from the spatial moments already stored in m.
func completeMoments(m *cv.Moments) {
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
	// nu_pq = mu_pq / m00^(1+(p+q)/2); use |m00| so the fractional power is real.
	am00 := math.Abs(m.M00)
	s2 := 1 / (am00 * am00)
	s25 := s2 / math.Sqrt(am00)
	m.Nu20 = m.Mu20 * s2
	m.Nu11 = m.Mu11 * s2
	m.Nu02 = m.Mu02 * s2
	m.Nu30 = m.Mu30 * s25
	m.Nu21 = m.Mu21 * s25
	m.Nu12 = m.Mu12 * s25
	m.Nu03 = m.Mu03 * s25
}

// HuMoments returns the seven Hu invariant moments derived from the normalised
// central moments of m. The Hu moments are invariant to translation, scale and
// rotation (the first six are also reflection-invariant in magnitude; the
// seventh changes sign under reflection), which makes them a compact rotation-
// and scale-independent shape signature.
func HuMoments(m cv.Moments) [7]float64 {
	n20, n02, n11 := m.Nu20, m.Nu02, m.Nu11
	n30, n21, n12, n03 := m.Nu30, m.Nu21, m.Nu12, m.Nu03

	var h [7]float64
	h[0] = n20 + n02
	h[1] = (n20-n02)*(n20-n02) + 4*n11*n11

	t1 := n30 - 3*n12
	t2 := 3*n21 - n03
	h[2] = t1*t1 + t2*t2

	s1 := n30 + n12
	s2 := n21 + n03
	h[3] = s1*s1 + s2*s2

	h[4] = t1*s1*(s1*s1-3*s2*s2) + t2*s2*(3*s1*s1-s2*s2)
	h[5] = (n20-n02)*(s1*s1-s2*s2) + 4*n11*s1*s2
	h[6] = t2*s1*(s1*s1-3*s2*s2) - t1*s2*(3*s1*s1-s2*s2)
	return h
}

// MatchShapes compares two contours by their Hu moments and returns a
// dissimilarity score: 0 for a perfect match, growing as the shapes differ.
// method selects the metric ([ContoursMatchI1], [ContoursMatchI2] or
// [ContoursMatchI3]), matching OpenCV's cv::matchShapes.
//
// Each contour's Hu moments are log-transformed to m_i = sign(h_i)·log10|h_i|
// before comparison, which compresses their wide dynamic range. Because Hu
// moments are invariant to translation, scale and rotation, so is the score:
// congruent shapes at any position, size or orientation compare as (near) zero.
// It panics on an unknown method.
func MatchShapes(c1, c2 []cv.Point, method int) float64 {
	a := huLog(HuMoments(ContourMoments(c1)))
	b := huLog(HuMoments(ContourMoments(c2)))

	var result float64
	switch method {
	case ContoursMatchI1:
		for i := 0; i < 7; i++ {
			if a.nonzero[i] && b.nonzero[i] {
				result += math.Abs(1/a.m[i] - 1/b.m[i])
			}
		}
	case ContoursMatchI2:
		for i := 0; i < 7; i++ {
			if a.nonzero[i] && b.nonzero[i] {
				result += math.Abs(a.m[i] - b.m[i])
			}
		}
	case ContoursMatchI3:
		for i := 0; i < 7; i++ {
			if a.nonzero[i] && b.nonzero[i] {
				d := math.Abs(a.m[i]-b.m[i]) / math.Abs(a.m[i])
				if d > result {
					result = d
				}
			}
		}
	default:
		panic("shape: MatchShapes unknown method")
	}
	return result
}

// huSig holds the log-transformed Hu moments of one contour and which entries
// were non-zero (and therefore participate in the comparison).
type huSig struct {
	m       [7]float64
	nonzero [7]bool
}

// huLog applies OpenCV's sign-preserving log10 transform to Hu moments.
func huLog(h [7]float64) huSig {
	var out huSig
	for i, v := range h {
		if v == 0 {
			continue
		}
		out.nonzero[i] = true
		mag := math.Log10(math.Abs(v))
		if v < 0 {
			mag = -mag
		}
		out.m[i] = mag
	}
	return out
}
