package video

// PointF is a floating-point image coordinate (X is the column, Y is the row).
// It is the sub-pixel analogue of cv.Point and is used by the sub-pixel optical
// flow and 2-D transform-estimation routines in this package.
type PointF struct {
	X float64
	Y float64
}

// TermCriteria describes when an iterative algorithm (mean-shift, ECC, the DIS
// inverse search) should stop. It mirrors cv::TermCriteria: iteration halts once
// MaxCount iterations have run or the per-iteration change falls at or below
// Epsilon, whichever happens first. A non-positive field disables that test; at
// least one of the two must be positive.
type TermCriteria struct {
	// MaxCount is the maximum number of iterations (<= 0 disables the count test).
	MaxCount int
	// Epsilon is the convergence threshold on the iteration step (<= 0 disables
	// the accuracy test).
	Epsilon float64
}

// NewTermCriteria returns a TermCriteria with the given iteration cap and
// accuracy threshold.
func NewTermCriteria(maxCount int, epsilon float64) TermCriteria {
	return TermCriteria{MaxCount: maxCount, Epsilon: epsilon}
}

// reached reports whether iteration should stop after completing iteration iter
// (zero-based) with the given step magnitude.
func (t TermCriteria) reached(iter int, step float64) bool {
	if t.MaxCount > 0 && iter+1 >= t.MaxCount {
		return true
	}
	if t.Epsilon > 0 && step <= t.Epsilon {
		return true
	}
	return false
}

// iterCap returns a usable upper bound on iterations, defaulting to fallback
// when MaxCount is non-positive.
func (t TermCriteria) iterCap(fallback int) int {
	if t.MaxCount > 0 {
		return t.MaxCount
	}
	return fallback
}
