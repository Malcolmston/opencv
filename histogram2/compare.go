package histogram2

import "math"

// CompareMethod selects the distance or similarity measure used by
// [CompareHist1D] and [CompareHist2D].
type CompareMethod int

const (
	// CompareCorrelation measures the Pearson correlation of the two
	// histograms; higher (up to 1) means more similar.
	CompareCorrelation CompareMethod = iota
	// CompareChiSquare is the asymmetric chi-square distance; lower (down to
	// 0) means more similar.
	CompareChiSquare
	// CompareIntersection sums the per-bin minima; higher means more similar,
	// with a maximum equal to the smaller histogram's total mass.
	CompareIntersection
	// CompareBhattacharyya is the Bhattacharyya distance; lower (0) means more
	// similar, 1 means no overlap. Equivalent to the Hellinger distance.
	CompareBhattacharyya
	// CompareChiSquareAlt is the symmetric ("alternative") chi-square distance
	// used by OpenCV's HISTCMP_CHISQR_ALT; lower means more similar.
	CompareChiSquareAlt
	// CompareKLDiv is the Kullback-Leibler divergence of the first histogram
	// from the second; lower (0) means more similar.
	CompareKLDiv
)

// Correlation returns the Pearson correlation coefficient of two equal-length
// histograms, matching OpenCV's HISTCMP_CORREL. The result lies in [-1, 1]
// where 1 indicates identical shapes. It panics if the slices differ in length.
func Correlation(a, b []float64) float64 {
	if len(a) != len(b) {
		panic("histogram2: Correlation length mismatch")
	}
	n := float64(len(a))
	var ma, mb float64
	for i := range a {
		ma += a[i]
		mb += b[i]
	}
	ma /= n
	mb /= n
	var num, da, db float64
	for i := range a {
		x := a[i] - ma
		y := b[i] - mb
		num += x * y
		da += x * x
		db += y * y
	}
	denom := math.Sqrt(da * db)
	if denom == 0 {
		return 1
	}
	return num / denom
}

// ChiSquare returns the asymmetric chi-square distance of two equal-length
// histograms, matching OpenCV's HISTCMP_CHISQR: the sum over bins of
// (a-b)^2 / a, skipping bins where a is zero. Lower is more similar. It panics
// if the slices differ in length.
func ChiSquare(a, b []float64) float64 {
	if len(a) != len(b) {
		panic("histogram2: ChiSquare length mismatch")
	}
	var s float64
	for i := range a {
		if a[i] != 0 {
			d := a[i] - b[i]
			s += d * d / a[i]
		}
	}
	return s
}

// ChiSquareAlt returns the symmetric chi-square distance of two equal-length
// histograms, matching OpenCV's HISTCMP_CHISQR_ALT: twice the sum over bins of
// (a-b)^2 / (a+b), skipping bins where a+b is zero. Lower is more similar. It
// panics if the slices differ in length.
func ChiSquareAlt(a, b []float64) float64 {
	if len(a) != len(b) {
		panic("histogram2: ChiSquareAlt length mismatch")
	}
	var s float64
	for i := range a {
		sum := a[i] + b[i]
		if sum != 0 {
			d := a[i] - b[i]
			s += d * d / sum
		}
	}
	return 2 * s
}

// Intersection returns the histogram intersection of two equal-length
// histograms, matching OpenCV's HISTCMP_INTERSECT: the sum over bins of
// min(a, b). Higher is more similar. It panics if the slices differ in length.
func Intersection(a, b []float64) float64 {
	if len(a) != len(b) {
		panic("histogram2: Intersection length mismatch")
	}
	var s float64
	for i := range a {
		s += math.Min(a[i], b[i])
	}
	return s
}

// Bhattacharyya returns the Bhattacharyya distance of two equal-length
// histograms, matching OpenCV's HISTCMP_BHATTACHARYYA (also the Hellinger
// distance): sqrt(1 - BC) where BC is the Bhattacharyya coefficient computed
// against the normalised histograms. The result lies in [0, 1] where 0 means
// identical. It panics if the slices differ in length.
func Bhattacharyya(a, b []float64) float64 {
	if len(a) != len(b) {
		panic("histogram2: Bhattacharyya length mismatch")
	}
	var sa, sb float64
	for i := range a {
		sa += a[i]
		sb += b[i]
	}
	if sa == 0 || sb == 0 {
		if sa == 0 && sb == 0 {
			return 0
		}
		return 1
	}
	var bc float64
	for i := range a {
		bc += math.Sqrt((a[i] / sa) * (b[i] / sb))
	}
	v := 1 - bc
	if v < 0 {
		v = 0
	}
	return math.Sqrt(v)
}

// KLDivergence returns the Kullback-Leibler divergence D(a||b) of two
// equal-length histograms, computed against their normalised distributions.
// Bins where a is zero contribute nothing; if a has mass where b is zero the
// divergence is infinite. It panics if the slices differ in length.
func KLDivergence(a, b []float64) float64 {
	if len(a) != len(b) {
		panic("histogram2: KLDivergence length mismatch")
	}
	var sa, sb float64
	for i := range a {
		sa += a[i]
		sb += b[i]
	}
	if sa == 0 {
		return 0
	}
	if sb == 0 {
		return math.Inf(1)
	}
	var s float64
	for i := range a {
		pa := a[i] / sa
		if pa <= 0 {
			continue
		}
		pb := b[i] / sb
		if pb <= 0 {
			return math.Inf(1)
		}
		s += pa * math.Log(pa/pb)
	}
	return s
}

// EMD1D returns the one-dimensional Earth Mover's Distance (Wasserstein-1
// distance) between two equal-length histograms with unit ground distance
// between adjacent bins. Both histograms are normalised to unit mass first, so
// the result is the sum of absolute differences of their cumulative
// distributions. Lower is more similar. It panics if the slices differ in
// length.
func EMD1D(a, b []float64) float64 {
	if len(a) != len(b) {
		panic("histogram2: EMD1D length mismatch")
	}
	var sa, sb float64
	for i := range a {
		sa += a[i]
		sb += b[i]
	}
	if sa == 0 || sb == 0 {
		return 0
	}
	var acc, work float64
	for i := range a {
		acc += a[i]/sa - b[i]/sb
		work += math.Abs(acc)
	}
	return work
}

// CompareHist1D compares two one-dimensional histograms with the given method.
// It returns [ErrSizeMismatch] if the histograms have different bin counts and
// [ErrInvalidArgument] if method is unknown.
func CompareHist1D(a, b *Histogram1D, method CompareMethod) (float64, error) {
	if a.BinCount != b.BinCount {
		return 0, ErrSizeMismatch
	}
	return histogram2compare(a.Counts, b.Counts, method)
}

// CompareHist2D compares two two-dimensional histograms with the given method,
// flattening them in row-major order. It returns [ErrSizeMismatch] if the
// histograms have different shapes and [ErrInvalidArgument] if method is
// unknown.
func CompareHist2D(a, b *Histogram2D, method CompareMethod) (float64, error) {
	if a.BinsX != b.BinsX || a.BinsY != b.BinsY {
		return 0, ErrSizeMismatch
	}
	return histogram2compare(a.Counts, b.Counts, method)
}

// histogram2compare dispatches to the requested comparison measure.
func histogram2compare(a, b []float64, method CompareMethod) (float64, error) {
	switch method {
	case CompareCorrelation:
		return Correlation(a, b), nil
	case CompareChiSquare:
		return ChiSquare(a, b), nil
	case CompareIntersection:
		return Intersection(a, b), nil
	case CompareBhattacharyya:
		return Bhattacharyya(a, b), nil
	case CompareChiSquareAlt:
		return ChiSquareAlt(a, b), nil
	case CompareKLDiv:
		return KLDivergence(a, b), nil
	default:
		return 0, ErrInvalidArgument
	}
}
