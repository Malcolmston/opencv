package phase_unwrapping

import (
	"math"
	"testing"

	cv "github.com/malcolmston/opencv"
)

// smoothBowl is a residue-free continuous surface whose range exceeds 2*pi, used
// across the new-method tests.
func smoothBowl(rows, cols int) [][]float64 {
	cy, cx := float64(rows-1)/2, float64(cols-1)/2
	return makeGrid(rows, cols, func(i, j int) float64 {
		dy := float64(i) - cy
		dx := float64(j) - cx
		return 0.02*(dy*dy+dx*dx) + 0.18*float64(i) + 0.13*float64(j)
	})
}

// spiralField builds a wrapped phase spiral around (cy,cx) that contains a
// genuine residue.
func spiralField(rows, cols int, cy, cx float64) [][]float64 {
	return makeGrid(rows, cols, func(i, j int) float64 {
		return Wrap(math.Atan2(float64(i)-cy, float64(j)-cx))
	})
}

// rasterIntegrate unwraps by a fixed raster path (first row, then each row from
// the pixel above): a congruent but naive baseline.
func rasterIntegrate(wrapped [][]float64) [][]float64 {
	rows := len(wrapped)
	cols := len(wrapped[0])
	u := make([][]float64, rows)
	for i := range u {
		u[i] = make([]float64, cols)
	}
	u[0][0] = Wrap(wrapped[0][0])
	for j := 1; j < cols; j++ {
		u[0][j] = u[0][j-1] + Wrap(Wrap(wrapped[0][j])-Wrap(wrapped[0][j-1]))
	}
	for i := 1; i < rows; i++ {
		u[i][0] = u[i-1][0] + Wrap(Wrap(wrapped[i][0])-Wrap(wrapped[i-1][0]))
		for j := 1; j < cols; j++ {
			u[i][j] = u[i][j-1] + Wrap(Wrap(wrapped[i][j])-Wrap(wrapped[i][j-1]))
		}
	}
	return u
}

// congruent reports the maximum pixel discrepancy between Rewrap(unwrapped) and
// the original wrapped map, ignoring NaN pixels.
func maxCongruenceError(unwrapped, wrapped [][]float64) float64 {
	rw := Rewrap(unwrapped)
	maxErr := 0.0
	for i := range wrapped {
		for j := range wrapped[i] {
			if math.IsNaN(unwrapped[i][j]) {
				continue
			}
			e := math.Abs(Wrap(rw[i][j] - Wrap(wrapped[i][j])))
			if e > maxErr {
				maxErr = e
			}
		}
	}
	return maxErr
}

func TestLeastSquaresUnwrap(t *testing.T) {
	original := smoothBowl(28, 34)
	wrapped := WrapPhaseMap(original)
	if n := CountResidues(wrapped); n != 0 {
		t.Fatalf("test surface unexpectedly has %d residues", n)
	}
	got, err := LeastSquaresUnwrap(wrapped)
	if err != nil {
		t.Fatalf("LeastSquaresUnwrap: %v", err)
	}
	gridsEqualUpToConstant(t, "least-squares", got, original, 1e-6)
}

func TestLeastSquaresUnwrapMat(t *testing.T) {
	original := smoothBowl(20, 24)
	wrapped := WrapPhaseMap(original)
	fm := cv.NewFloatMat(20, 24)
	for i := 0; i < 20; i++ {
		for j := 0; j < 24; j++ {
			fm.Data[i*24+j] = wrapped[i][j]
		}
	}
	out, err := LeastSquaresUnwrapMat(fm)
	if err != nil {
		t.Fatalf("LeastSquaresUnwrapMat: %v", err)
	}
	grid := make([][]float64, 20)
	for i := 0; i < 20; i++ {
		grid[i] = make([]float64, 24)
		for j := 0; j < 24; j++ {
			grid[i][j] = out.Data[i*24+j]
		}
	}
	gridsEqualUpToConstant(t, "least-squares-mat", grid, original, 1e-6)
}

func TestWeightedLeastSquaresUniform(t *testing.T) {
	original := smoothBowl(24, 30)
	wrapped := WrapPhaseMap(original)
	got, err := WeightedLeastSquaresUnwrap(wrapped, nil, 0, 0)
	if err != nil {
		t.Fatalf("WeightedLeastSquaresUnwrap: %v", err)
	}
	gridsEqualUpToConstant(t, "wls-uniform", got, original, 1e-6)
}

func TestWeightedLeastSquaresVaryingWeights(t *testing.T) {
	rows, cols := 26, 26
	original := smoothBowl(rows, cols)
	wrapped := WrapPhaseMap(original)
	// Reliability-like weights derived from the pseudocorrelation of the map.
	weights := PseudoCorrelation(wrapped)
	got, err := WeightedLeastSquaresUnwrap(wrapped, weights, 0, 0)
	if err != nil {
		t.Fatalf("WeightedLeastSquaresUnwrap: %v", err)
	}
	// Residue-free: any positive weighting recovers the surface exactly.
	gridsEqualUpToConstant(t, "wls-varying", got, original, 1e-5)
}

func TestGoldsteinResidueFree(t *testing.T) {
	original := smoothBowl(30, 30)
	wrapped := WrapPhaseMap(original)
	got, err := GoldsteinBranchCut(wrapped, 0)
	if err != nil {
		t.Fatalf("GoldsteinBranchCut: %v", err)
	}
	gridsEqualUpToConstant(t, "goldstein-smooth", got, original, 1e-6)
}

func TestGoldsteinSpiralCongruent(t *testing.T) {
	rows, cols := 24, 24
	wrapped := spiralField(rows, cols, 12, 12)
	if CountResidues(wrapped) == 0 {
		t.Fatalf("spiral should contain residues")
	}
	got, err := GoldsteinBranchCut(wrapped, 0)
	if err != nil {
		t.Fatalf("GoldsteinBranchCut: %v", err)
	}
	// The branch-cut result must be congruent to the wrapped input everywhere.
	if e := maxCongruenceError(got, wrapped); e > 1e-9 {
		t.Fatalf("branch-cut result not congruent to input: max error %g", e)
	}
	// A cut must have been placed for the unbalanced residue.
	cuts := BranchCuts(wrapped, 0)
	any := false
	for i := range cuts {
		for j := range cuts[i] {
			any = any || cuts[i][j]
		}
	}
	if !any {
		t.Fatalf("expected at least one branch-cut pixel for a residue map")
	}
}

func TestFlynnResidueFree(t *testing.T) {
	original := smoothBowl(26, 28)
	wrapped := WrapPhaseMap(original)
	got, err := FlynnMinimumDiscontinuity(wrapped, nil, 0)
	if err != nil {
		t.Fatalf("FlynnMinimumDiscontinuity: %v", err)
	}
	gridsEqualUpToConstant(t, "flynn-smooth", got, original, 1e-6)
	if d := TotalDiscontinuity(got); d != 0 {
		t.Fatalf("residue-free Flynn result has %d discontinuities", d)
	}
}

func TestFlynnReducesDiscontinuity(t *testing.T) {
	wrapped := spiralField(28, 28, 14, 14)
	got, err := FlynnMinimumDiscontinuity(wrapped, nil, 0)
	if err != nil {
		t.Fatalf("FlynnMinimumDiscontinuity: %v", err)
	}
	if e := maxCongruenceError(got, wrapped); e > 1e-9 {
		t.Fatalf("Flynn result not congruent to input: %g", e)
	}
	dFlynn := TotalDiscontinuity(got)
	dRaster := TotalDiscontinuity(rasterIntegrate(wrapped))
	if dFlynn > dRaster {
		t.Fatalf("Flynn discontinuity %d exceeds raster baseline %d", dFlynn, dRaster)
	}
}

func TestQualityGuidedUnwrap(t *testing.T) {
	original := smoothBowl(30, 26)
	wrapped := WrapPhaseMap(original)
	// Default quality.
	got, err := QualityGuidedUnwrap(wrapped, nil)
	if err != nil {
		t.Fatalf("QualityGuidedUnwrap: %v", err)
	}
	gridsEqualUpToConstant(t, "quality-default", got, original, 1e-6)
	// Explicit pseudocorrelation quality.
	got2, err := QualityGuidedUnwrap(wrapped, PseudoCorrelation(wrapped))
	if err != nil {
		t.Fatalf("QualityGuidedUnwrap (pc): %v", err)
	}
	gridsEqualUpToConstant(t, "quality-pc", got2, original, 1e-6)
}

func TestMaskedUnwrapRespectsMask(t *testing.T) {
	rows, cols := 24, 24
	original := smoothBowl(rows, cols)
	wrapped := WrapPhaseMap(original)
	// Mask out a block and fill it with garbage: if the unwrapper ever stepped
	// into it, the valid region would be corrupted.
	mask := make([][]bool, rows)
	for i := 0; i < rows; i++ {
		mask[i] = make([]bool, cols)
		for j := 0; j < cols; j++ {
			mask[i][j] = true
		}
	}
	for i := 8; i < 16; i++ {
		for j := 8; j < 16; j++ {
			mask[i][j] = false
			wrapped[i][j] = Wrap(37.0*float64(i) - 91.0*float64(j)) // garbage
		}
	}
	got, err := MaskedUnwrap(wrapped, mask, nil)
	if err != nil {
		t.Fatalf("MaskedUnwrap: %v", err)
	}
	// Masked-out pixels are NaN.
	for i := 8; i < 16; i++ {
		for j := 8; j < 16; j++ {
			if !math.IsNaN(got[i][j]) {
				t.Fatalf("masked pixel (%d,%d) should be NaN, got %g", i, j, got[i][j])
			}
		}
	}
	// The valid region (still one connected component around the hole) must match
	// the original up to a single constant.
	offset := math.NaN()
	maxDev := 0.0
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			if math.IsNaN(got[i][j]) {
				continue
			}
			if math.IsNaN(offset) {
				offset = got[i][j] - original[i][j]
			}
			dev := math.Abs((got[i][j] - original[i][j]) - offset)
			if dev > maxDev {
				maxDev = dev
			}
		}
	}
	if maxDev > 1e-6 {
		t.Fatalf("masked unwrap deviates by %g (garbage leaked?)", maxDev)
	}
}

func TestTemporalUnwrap(t *testing.T) {
	rows, cols := 20, 20
	scales := []float64{1, 2, 4, 8}
	// Base field within (-pi, pi] so the coarsest map is unambiguous.
	base := makeGrid(rows, cols, func(i, j int) float64 {
		return 0.9 * math.Pi * (float64(i)/float64(rows-1) - 0.5 + 0.3*(float64(j)/float64(cols-1)-0.5))
	})
	maps := make([][][]float64, len(scales))
	for k, s := range scales {
		maps[k] = makeGrid(rows, cols, func(i, j int) float64 { return Wrap(s * base[i][j]) })
	}
	got, err := TemporalUnwrap(maps, scales)
	if err != nil {
		t.Fatalf("TemporalUnwrap: %v", err)
	}
	// Expected finest absolute phase = scales[last]*base.
	finest := scales[len(scales)-1]
	expected := makeGrid(rows, cols, func(i, j int) float64 { return finest * base[i][j] })
	gridsEqualUpToConstant(t, "temporal", got, expected, 1e-9)
}

func TestWrapToPiRewrapCongruence(t *testing.T) {
	for _, x := range []float64{-7, -math.Pi, 0, 1.2, math.Pi, 4 * math.Pi, 50.3} {
		w := WrapToPi(x)
		if w < -math.Pi-1e-12 || w > math.Pi+1e-12 {
			t.Fatalf("WrapToPi(%g)=%g out of range", x, w)
		}
	}
	original := smoothBowl(16, 18)
	wrapped := WrapPhaseMap(original)
	est, _ := LeastSquaresUnwrap(wrapped)
	// LeastSquares need not be congruent; Congruence must fix that.
	cong, err := Congruence(est, wrapped)
	if err != nil {
		t.Fatalf("Congruence: %v", err)
	}
	if e := maxCongruenceError(cong, wrapped); e > 1e-9 {
		t.Fatalf("Congruence result not congruent: %g", e)
	}
	// And it stays within pi of the estimate.
	for i := range est {
		for j := range est[i] {
			if math.Abs(cong[i][j]-est[i][j]) > math.Pi+1e-9 {
				t.Fatalf("Congruence moved (%d,%d) by more than pi", i, j)
			}
		}
	}
}

func TestQualityMapsDetectSpike(t *testing.T) {
	rows, cols := 21, 21
	original := makeGrid(rows, cols, func(i, j int) float64 { return 0.0 })
	original[10][10] = 2.5
	wrapped := WrapPhaseMap(original)

	pdv := PhaseDerivativeVariance(wrapped)
	mpg := MaximumPhaseGradient(wrapped)
	pc := PseudoCorrelation(wrapped)
	for _, m := range [][][]float64{pdv, mpg, pc} {
		if len(m) != rows || len(m[0]) != cols {
			t.Fatalf("quality map has wrong shape")
		}
	}
	// Cost maps: higher near the spike than in a flat corner.
	if !(pdv[10][10] > pdv[2][2]) {
		t.Fatalf("PhaseDerivativeVariance should be higher at spike: %g vs %g", pdv[10][10], pdv[2][2])
	}
	if !(mpg[10][10] > mpg[2][2]) {
		t.Fatalf("MaximumPhaseGradient should be higher at spike")
	}
	// Reliability map: lower near the spike.
	if !(pc[10][10] < pc[2][2]) {
		t.Fatalf("PseudoCorrelation should be lower at spike: %g vs %g", pc[10][10], pc[2][2])
	}
}

func TestResidueListAndBranchCuts(t *testing.T) {
	smooth := WrapPhaseMap(smoothBowl(16, 16))
	if rl := ResidueList(smooth); len(rl) != 0 {
		t.Fatalf("smooth map should have no residues, got %d", len(rl))
	}
	if BranchCuts(smooth, 0) != nil && anyTrue(BranchCuts(smooth, 0)) {
		t.Fatalf("smooth map should have no branch cuts")
	}
	spiral := spiralField(20, 20, 10, 10)
	rl := ResidueList(spiral)
	if len(rl) == 0 {
		t.Fatalf("spiral should list at least one residue")
	}
	for _, r := range rl {
		if r.Charge == 0 {
			t.Fatalf("residue list must contain only non-zero charges")
		}
	}
}

func anyTrue(g [][]bool) bool {
	for i := range g {
		for j := range g[i] {
			if g[i][j] {
				return true
			}
		}
	}
	return false
}

func TestUnwrapperInterface(t *testing.T) {
	original := smoothBowl(22, 24)
	wrapped := WrapPhaseMap(original)
	methods := map[string]Unwrapper{
		"histogram":     HistogramUnwrapper{},
		"least-squares": LeastSquaresUnwrapper{},
		"weighted-ls":   WeightedLeastSquaresUnwrapper{},
		"goldstein":     GoldsteinUnwrapper{},
		"flynn":         FlynnUnwrapper{},
		"quality":       QualityGuidedUnwrapper{},
	}
	for name, m := range methods {
		got, err := m.Unwrap(wrapped)
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		gridsEqualUpToConstant(t, name, got, original, 1e-5)
	}
}

func TestNewMethodsEmptyAndShapeErrors(t *testing.T) {
	if _, err := LeastSquaresUnwrap(nil); err != ErrEmptyInput {
		t.Fatalf("LeastSquaresUnwrap(nil): %v", err)
	}
	if _, err := GoldsteinBranchCut([][]float64{{1}}, 0); err != ErrEmptyInput {
		t.Fatalf("GoldsteinBranchCut tiny: %v", err)
	}
	small := WrapPhaseMap(smoothBowl(6, 6))
	if _, err := WeightedLeastSquaresUnwrap(small, [][]float64{{1, 2}}, 0, 0); err != ErrShapeMismatch {
		t.Fatalf("WeightedLeastSquaresUnwrap bad weights: %v", err)
	}
	if _, err := FlynnMinimumDiscontinuity(small, [][]float64{{1}}, 0); err != ErrShapeMismatch {
		t.Fatalf("Flynn bad weights: %v", err)
	}
	if _, err := MaskedUnwrap(small, [][]bool{{true}}, nil); err != ErrShapeMismatch {
		t.Fatalf("MaskedUnwrap bad mask: %v", err)
	}
	if _, err := Congruence(small, [][]float64{{1}}); err != ErrShapeMismatch {
		t.Fatalf("Congruence bad shape: %v", err)
	}
	if _, err := TemporalUnwrap(nil, nil); err != ErrShapeMismatch {
		t.Fatalf("TemporalUnwrap empty: %v", err)
	}
	if _, err := TemporalUnwrap([][][]float64{small}, []float64{1, 2}); err != ErrShapeMismatch {
		t.Fatalf("TemporalUnwrap scale mismatch: %v", err)
	}
}
