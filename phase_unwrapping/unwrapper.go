package phase_unwrapping

// Unwrapper is the common interface implemented by every 2-D phase-unwrapping
// method in this package. Unwrap takes a wrapped phase map as a [row][col] grid
// and returns the unwrapped absolute phase, defined up to a global 2*pi constant,
// or an error such as [ErrEmptyInput]. The interface lets callers select or swap
// algorithms at run time and iterate over several methods uniformly.
type Unwrapper interface {
	Unwrap(wrapped [][]float64) ([][]float64, error)
}

// HistogramUnwrapper adapts the quality-guided [HistogramPhaseUnwrapping] to the
// [Unwrapper] interface. The zero value is usable and applies default parameters.
type HistogramUnwrapper struct {
	// Params configures the underlying histogram unwrapper; the zero value is
	// normalised to sensible defaults.
	Params Params
}

// Unwrap implements [Unwrapper].
func (h HistogramUnwrapper) Unwrap(wrapped [][]float64) ([][]float64, error) {
	u := NewHistogramPhaseUnwrapping(h.Params)
	return u.UnwrapPhaseMapGrid(wrapped)
}

// LeastSquaresUnwrapper adapts [LeastSquaresUnwrap] (unweighted DCT Poisson) to
// the [Unwrapper] interface. The zero value is usable.
type LeastSquaresUnwrapper struct{}

// Unwrap implements [Unwrapper].
func (LeastSquaresUnwrapper) Unwrap(wrapped [][]float64) ([][]float64, error) {
	return LeastSquaresUnwrap(wrapped)
}

// WeightedLeastSquaresUnwrapper adapts [WeightedLeastSquaresUnwrap] (weighted
// least squares by PCG) to the [Unwrapper] interface.
type WeightedLeastSquaresUnwrapper struct {
	// Weights is the optional per-pixel reliability map (nil means uniform).
	Weights [][]float64
	// MaxIter caps the PCG iterations (non-positive selects a default).
	MaxIter int
	// Tol is the relative residual stopping threshold (non-positive selects a
	// default).
	Tol float64
}

// Unwrap implements [Unwrapper].
func (w WeightedLeastSquaresUnwrapper) Unwrap(wrapped [][]float64) ([][]float64, error) {
	return WeightedLeastSquaresUnwrap(wrapped, w.Weights, w.MaxIter, w.Tol)
}

// GoldsteinUnwrapper adapts [GoldsteinBranchCut] to the [Unwrapper] interface.
// The zero value is usable and imposes no branch-cut distance limit.
type GoldsteinUnwrapper struct {
	// MaxBoxRadius limits how far a residue may pair with an opposite-charge
	// partner before it is cut to the border (non-positive means no limit).
	MaxBoxRadius int
}

// Unwrap implements [Unwrapper].
func (g GoldsteinUnwrapper) Unwrap(wrapped [][]float64) ([][]float64, error) {
	return GoldsteinBranchCut(wrapped, g.MaxBoxRadius)
}

// FlynnUnwrapper adapts [FlynnMinimumDiscontinuity] to the [Unwrapper] interface.
type FlynnUnwrapper struct {
	// Weights is the optional per-pixel reliability map (nil means uniform).
	Weights [][]float64
	// MaxIter caps the refinement sweeps (non-positive selects a default).
	MaxIter int
}

// Unwrap implements [Unwrapper].
func (f FlynnUnwrapper) Unwrap(wrapped [][]float64) ([][]float64, error) {
	return FlynnMinimumDiscontinuity(wrapped, f.Weights, f.MaxIter)
}

// QualityGuidedUnwrapper adapts [QualityGuidedUnwrap] to the [Unwrapper]
// interface. The zero value is usable and selects the default quality map.
type QualityGuidedUnwrapper struct {
	// Quality is the optional higher-is-better reliability map (nil selects the
	// negated phase-derivative variance).
	Quality [][]float64
}

// Unwrap implements [Unwrapper].
func (q QualityGuidedUnwrapper) Unwrap(wrapped [][]float64) ([][]float64, error) {
	return QualityGuidedUnwrap(wrapped, q.Quality)
}

// Compile-time checks that every adapter satisfies Unwrapper.
var (
	_ Unwrapper = HistogramUnwrapper{}
	_ Unwrapper = LeastSquaresUnwrapper{}
	_ Unwrapper = WeightedLeastSquaresUnwrapper{}
	_ Unwrapper = GoldsteinUnwrapper{}
	_ Unwrapper = FlynnUnwrapper{}
	_ Unwrapper = QualityGuidedUnwrapper{}
)
