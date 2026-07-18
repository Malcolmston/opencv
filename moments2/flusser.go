package moments2

// FlusserI1 returns the first Flusser & Suk affine moment invariant, computed
// from the central moments of m and normalized by mu00^4:
//
//	I1 = (mu20*mu02 - mu11^2) / mu00^4.
//
// It is invariant to any affine transform of the shape. It returns 0 for a shape
// of zero mass.
func FlusserI1(m Moments) float64 {
	if m.M00 == 0 {
		return 0
	}
	num := m.Mu20*m.Mu02 - m.Mu11*m.Mu11
	return num / pow(m.M00, 4)
}

// FlusserI2 returns the second Flusser & Suk affine moment invariant, built from
// the third-order central moments and normalized by mu00^10. It is invariant to
// any affine transform. It returns 0 for a shape of zero mass.
func FlusserI2(m Moments) float64 {
	if m.M00 == 0 {
		return 0
	}
	num := m.Mu30*m.Mu30*m.Mu03*m.Mu03 -
		6*m.Mu30*m.Mu21*m.Mu12*m.Mu03 +
		4*m.Mu30*m.Mu12*m.Mu12*m.Mu12 +
		4*m.Mu21*m.Mu21*m.Mu21*m.Mu03 -
		3*m.Mu21*m.Mu21*m.Mu12*m.Mu12
	return num / pow(m.M00, 10)
}

// FlusserI3 returns the third Flusser & Suk affine moment invariant, mixing
// second- and third-order central moments and normalized by mu00^7. It is
// invariant to any affine transform. It returns 0 for a shape of zero mass.
func FlusserI3(m Moments) float64 {
	if m.M00 == 0 {
		return 0
	}
	num := m.Mu20*(m.Mu21*m.Mu03-m.Mu12*m.Mu12) -
		m.Mu11*(m.Mu30*m.Mu03-m.Mu21*m.Mu12) +
		m.Mu02*(m.Mu30*m.Mu12-m.Mu21*m.Mu21)
	return num / pow(m.M00, 7)
}

// FlusserI4 returns the fourth Flusser & Suk affine moment invariant, the most
// complex of the classic set, normalized by mu00^11. It is invariant to any
// affine transform. It returns 0 for a shape of zero mass.
func FlusserI4(m Moments) float64 {
	if m.M00 == 0 {
		return 0
	}
	mu20, mu11, mu02 := m.Mu20, m.Mu11, m.Mu02
	mu30, mu21, mu12, mu03 := m.Mu30, m.Mu21, m.Mu12, m.Mu03
	num := mu20*mu20*mu20*mu03*mu03 -
		6*mu20*mu20*mu11*mu12*mu03 -
		6*mu20*mu20*mu02*mu21*mu03 +
		9*mu20*mu20*mu02*mu12*mu12 +
		12*mu20*mu11*mu11*mu21*mu03 +
		6*mu20*mu11*mu02*mu30*mu03 -
		18*mu20*mu11*mu02*mu21*mu12 -
		8*mu11*mu11*mu11*mu30*mu03 -
		6*mu20*mu02*mu02*mu30*mu12 +
		9*mu20*mu02*mu02*mu21*mu21 +
		12*mu11*mu11*mu02*mu30*mu12 -
		6*mu11*mu02*mu02*mu30*mu21 +
		mu02*mu02*mu02*mu30*mu30
	return num / pow(m.M00, 11)
}

// FlusserInvariants returns the four classic Flusser & Suk affine moment
// invariants as an array [I1, I2, I3, I4].
func FlusserInvariants(m Moments) [4]float64 {
	return [4]float64{FlusserI1(m), FlusserI2(m), FlusserI3(m), FlusserI4(m)}
}

// pow raises base to a small non-negative integer power.
func pow(base float64, exp int) float64 {
	r := 1.0
	for i := 0; i < exp; i++ {
		r *= base
	}
	return r
}
