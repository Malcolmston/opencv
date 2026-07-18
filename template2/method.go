package template2

// Method selects the similarity measure used by [MatchTemplate] and the other
// matching routines.
type Method int

const (
	// MethodSAD is the sum of absolute differences between the template and the
	// overlapped patch. The best match is the minimum; 0 is a perfect match.
	MethodSAD Method = iota
	// MethodSSD is the sum of squared differences. The best match is the
	// minimum; 0 is a perfect match.
	MethodSSD
	// MethodSSDNormed is the sum of squared differences divided by the product
	// of the patch and template L2 norms. Values lie in [0,2]; the best match
	// is the minimum.
	MethodSSDNormed
	// MethodCrossCorr is the raw cross-correlation (sum of element-wise
	// products). The best match is the maximum. It is not contrast-invariant.
	MethodCrossCorr
	// MethodNCC is the cross-correlation normalised by the product of the patch
	// and template L2 norms. Values lie in [-1,1]; the best match is the
	// maximum.
	MethodNCC
	// MethodCorrCoeff is the covariance of the mean-subtracted template and
	// patch (unnormalised correlation coefficient). The best match is the
	// maximum. It is invariant to additive brightness offsets.
	MethodCorrCoeff
	// MethodZNCC is the zero-mean normalised cross-correlation, i.e. the
	// Pearson correlation coefficient between the template and patch. Values
	// lie in [-1,1]; the best match is the maximum. It is invariant to both
	// additive brightness offsets and multiplicative contrast changes.
	MethodZNCC
)

// String returns a short human-readable name for the method.
func (m Method) String() string {
	switch m {
	case MethodSAD:
		return "SAD"
	case MethodSSD:
		return "SSD"
	case MethodSSDNormed:
		return "SSDNormed"
	case MethodCrossCorr:
		return "CrossCorr"
	case MethodNCC:
		return "NCC"
	case MethodCorrCoeff:
		return "CorrCoeff"
	case MethodZNCC:
		return "ZNCC"
	default:
		return "Method(?)"
	}
}

// Valid reports whether m names a known similarity measure.
func (m Method) Valid() bool {
	return m >= MethodSAD && m <= MethodZNCC
}

// HigherIsBetter reports whether larger score values indicate stronger matches
// for this method. It returns true for the correlation measures
// ([MethodCrossCorr], [MethodNCC], [MethodCorrCoeff], [MethodZNCC]) and false
// for the difference measures ([MethodSAD], [MethodSSD], [MethodSSDNormed]).
func (m Method) HigherIsBetter() bool {
	switch m {
	case MethodCrossCorr, MethodNCC, MethodCorrCoeff, MethodZNCC:
		return true
	default:
		return false
	}
}
