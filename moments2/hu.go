package moments2

import "math"

// HuMoments returns Hu's seven invariant moments computed from the normalized
// central moments of m. The first six are invariant to translation, scale and
// rotation; the seventh additionally changes sign under reflection, allowing
// mirror images to be distinguished. The formulas match OpenCV's cv::HuMoments.
func HuMoments(m Moments) [7]float64 {
	var hu [7]float64
	t0 := m.Nu30 + m.Nu12
	t1 := m.Nu21 + m.Nu03
	q0 := t0 * t0
	q1 := t1 * t1
	n4 := 4 * m.Nu11
	s := m.Nu20 + m.Nu02
	d := m.Nu20 - m.Nu02
	hu[0] = s
	hu[1] = d*d + n4*m.Nu11
	hu[3] = q0 + q1
	hu[5] = d*(q0-q1) + n4*t0*t1
	t0 *= q0 - 3*q1
	t1 *= 3*q0 - q1
	q0 = m.Nu30 - 3*m.Nu12
	q1 = 3*m.Nu21 - m.Nu03
	hu[2] = q0*q0 + q1*q1
	hu[4] = q0*t0 + q1*t1
	hu[6] = q1*t0 - q0*t1
	return hu
}

// LogHuMoments returns a sign-preserving log-magnitude transform of the seven Hu
// moments, -sign(h)*log10(|h|), which compresses their very wide dynamic range
// into a form suitable for direct comparison. A zero Hu value maps to zero.
func LogHuMoments(m Moments) [7]float64 {
	hu := HuMoments(m)
	var out [7]float64
	for i, h := range hu {
		if h == 0 {
			out[i] = 0
			continue
		}
		out[i] = -moments2sign(h) * math.Log10(math.Abs(h))
	}
	return out
}

// MatchMethod selects the dissimilarity formula used by [MatchShapes].
type MatchMethod int

const (
	// MatchI1 sums the absolute differences of the reciprocals of the
	// log-transformed Hu moments (OpenCV CONTOURS_MATCH_I1).
	MatchI1 MatchMethod = 1
	// MatchI2 sums the absolute differences of the log-transformed Hu moments
	// (OpenCV CONTOURS_MATCH_I2).
	MatchI2 MatchMethod = 2
	// MatchI3 takes the maximum relative difference of the log-transformed Hu
	// moments (OpenCV CONTOURS_MATCH_I3).
	MatchI3 MatchMethod = 3
)

// MatchShapes returns a dissimilarity score between two shapes given by their
// moments, using their Hu invariants and the selected method. A score of zero
// means the shapes are identical up to translation, scale and rotation; larger
// values mean less similar. It panics on an unknown method.
func MatchShapes(a, b Moments, method MatchMethod) float64 {
	const eps = 1e-5
	ha := HuMoments(a)
	hb := HuMoments(b)
	var result float64
	switch method {
	case MatchI1:
		for i := 0; i < 7; i++ {
			ama := math.Abs(ha[i])
			amb := math.Abs(hb[i])
			if ama < eps || amb < eps {
				continue
			}
			va := 1 / (moments2sign(ha[i]) * math.Log10(ama))
			vb := 1 / (moments2sign(hb[i]) * math.Log10(amb))
			result += math.Abs(va - vb)
		}
	case MatchI2:
		for i := 0; i < 7; i++ {
			ama := math.Abs(ha[i])
			amb := math.Abs(hb[i])
			if ama < eps || amb < eps {
				continue
			}
			va := moments2sign(ha[i]) * math.Log10(ama)
			vb := moments2sign(hb[i]) * math.Log10(amb)
			result += math.Abs(va - vb)
		}
	case MatchI3:
		for i := 0; i < 7; i++ {
			ama := math.Abs(ha[i])
			amb := math.Abs(hb[i])
			if ama < eps || amb < eps {
				continue
			}
			va := moments2sign(ha[i]) * math.Log10(ama)
			vb := moments2sign(hb[i]) * math.Log10(amb)
			if r := math.Abs((va - vb) / va); r > result {
				result = r
			}
		}
	default:
		panic("moments2: MatchShapes unknown method")
	}
	return result
}
