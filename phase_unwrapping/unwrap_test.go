package phase_unwrapping

import (
	"math"
	"testing"

	cv "github.com/malcolmston/opencv"
)

// maxDeviationFromConstant returns the maximum absolute deviation of
// (unwrapped - original) from its value at pixel (0,0). A correct unwrap makes
// this difference a single global constant, so the deviation is ~0.
func maxDeviationFromConstant(t *testing.T, unwrapped, original [][]float64) float64 {
	t.Helper()
	rows := len(original)
	cols := len(original[0])
	offset := unwrapped[0][0] - original[0][0]
	maxDev := 0.0
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			dev := math.Abs((unwrapped[i][j] - original[i][j]) - offset)
			if dev > maxDev {
				maxDev = dev
			}
		}
	}
	return maxDev
}

func gridsEqualUpToConstant(t *testing.T, name string, unwrapped, original [][]float64, tol float64) {
	t.Helper()
	if dev := maxDeviationFromConstant(t, unwrapped, original); dev > tol {
		t.Fatalf("%s: unwrapped surface deviates from original by %g (tol %g)", name, dev, tol)
	}
}

func makeGrid(rows, cols int, f func(i, j int) float64) [][]float64 {
	g := make([][]float64, rows)
	for i := 0; i < rows; i++ {
		g[i] = make([]float64, cols)
		for j := 0; j < cols; j++ {
			g[i][j] = f(i, j)
		}
	}
	return g
}

func unwrapGrid(t *testing.T, original [][]float64) [][]float64 {
	t.Helper()
	wrapped := WrapPhaseMap(original)
	h := NewHistogramPhaseUnwrapping(DefaultParams(len(original[0]), len(original)))
	got, err := h.UnwrapPhaseMapGrid(wrapped)
	if err != nil {
		t.Fatalf("UnwrapPhaseMapGrid: %v", err)
	}
	return got
}

func TestUnwrapTiltedRamp(t *testing.T) {
	rows, cols := 32, 40
	original := makeGrid(rows, cols, func(i, j int) float64 {
		return 0.3*float64(i) + 0.25*float64(j) + 1.1
	})
	// Sanity: this surface must be residue-free.
	if n := CountResidues(WrapPhaseMap(original)); n != 0 {
		t.Fatalf("ramp unexpectedly has %d residues", n)
	}
	got := unwrapGrid(t, original)
	gridsEqualUpToConstant(t, "ramp", got, original, 1e-6)
}

func TestUnwrapQuadraticBowl(t *testing.T) {
	rows, cols := 36, 36
	cy, cx := 18.0, 18.0
	original := makeGrid(rows, cols, func(i, j int) float64 {
		dy := float64(i) - cy
		dx := float64(j) - cx
		return 0.02 * (dy*dy + dx*dx)
	})
	if n := CountResidues(WrapPhaseMap(original)); n != 0 {
		t.Fatalf("bowl unexpectedly has %d residues", n)
	}
	got := unwrapGrid(t, original)
	// Verify the range genuinely exceeds 2*pi (a non-trivial unwrap).
	minV, maxV := original[0][0], original[0][0]
	for i := range original {
		for j := range original[i] {
			if original[i][j] < minV {
				minV = original[i][j]
			}
			if original[i][j] > maxV {
				maxV = original[i][j]
			}
		}
	}
	if maxV-minV <= twoPi {
		t.Fatalf("bowl range %g does not exceed 2*pi", maxV-minV)
	}
	gridsEqualUpToConstant(t, "bowl", got, original, 1e-6)
}

func TestUnwrapLocalizedSteepRegion(t *testing.T) {
	rows, cols := 40, 40
	cy, cx := 20.0, 20.0
	// Gentle global tilt plus a smooth but localized steep Gaussian bump. The
	// bump is continuous everywhere, yet its central gradient (~2 rad/pixel) is
	// far steeper than the background, stressing the quality-guided path while
	// staying below pi/pixel so the map remains residue-free.
	original := makeGrid(rows, cols, func(i, j int) float64 {
		dy := float64(i) - cy
		dx := float64(j) - cx
		base := 0.1*float64(i) + 0.08*float64(j)
		bump := 10.0 * math.Exp(-(dy*dy+dx*dx)/(2*3.0*3.0))
		return base + bump
	})
	if n := CountResidues(WrapPhaseMap(original)); n != 0 {
		t.Fatalf("steep surface unexpectedly has %d residues", n)
	}
	got := unwrapGrid(t, original)
	gridsEqualUpToConstant(t, "steep", got, original, 1e-6)
}

// lcg is a tiny deterministic pseudo-random generator so the noisy test is
// reproducible without importing math/rand.
type lcg struct{ state uint64 }

func (r *lcg) next() float64 {
	r.state = r.state*6364136223846793005 + 1442695040888963407
	// Top 24 bits mapped to [-1, 1).
	return float64(r.state>>40)/float64(1<<23) - 1.0
}

func TestUnwrapNoisyButRecoverable(t *testing.T) {
	rows, cols := 30, 30
	rng := &lcg{state: 12345}
	// Continuous "true" surface = smooth ramp + small bounded noise. The noise
	// keeps per-pixel gradients well below pi, so the map stays residue-free and
	// must be recovered exactly (up to a global constant).
	original := makeGrid(rows, cols, func(i, j int) float64 { return 0.0 })
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			original[i][j] = 0.5*float64(i) + 0.4*float64(j) + 0.15*rng.next()
		}
	}
	wrapped := WrapPhaseMap(original)
	if n := CountResidues(wrapped); n != 0 {
		t.Skipf("noisy instance produced %d residues; skipping (setup issue)", n)
	}
	h := NewHistogramPhaseUnwrapping(DefaultParams(cols, rows))
	got, err := h.UnwrapPhaseMapGrid(wrapped)
	if err != nil {
		t.Fatalf("UnwrapPhaseMapGrid: %v", err)
	}
	gridsEqualUpToConstant(t, "noisy", got, original, 1e-6)
}

func TestUnwrapFloatMatMatchesGrid(t *testing.T) {
	rows, cols := 24, 28
	original := makeGrid(rows, cols, func(i, j int) float64 {
		return 0.2*float64(i) + 0.35*float64(j)
	})
	wrapped := WrapPhaseMap(original)

	// FloatMat path.
	fm := cv.NewFloatMat(rows, cols)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			fm.Data[i*cols+j] = wrapped[i][j]
		}
	}
	h := NewHistogramPhaseUnwrapping(DefaultParams(cols, rows))
	outMat, err := h.UnwrapPhaseMap(fm)
	if err != nil {
		t.Fatalf("UnwrapPhaseMap: %v", err)
	}
	if outMat.Rows != rows || outMat.Cols != cols {
		t.Fatalf("output shape %dx%d, want %dx%d", outMat.Rows, outMat.Cols, rows, cols)
	}

	// Grid path.
	h2 := NewHistogramPhaseUnwrapping(DefaultParams(cols, rows))
	outGrid, err := h2.UnwrapPhaseMapGrid(wrapped)
	if err != nil {
		t.Fatalf("UnwrapPhaseMapGrid: %v", err)
	}
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			if math.Abs(outMat.Data[i*cols+j]-outGrid[i][j]) > 1e-12 {
				t.Fatalf("FloatMat and grid paths disagree at (%d,%d): %g vs %g",
					i, j, outMat.Data[i*cols+j], outGrid[i][j])
			}
		}
	}
	// And it must equal the original up to a constant.
	gridsEqualUpToConstant(t, "floatmat", outGrid, original, 1e-6)
}

func TestInverseReliabilityMap(t *testing.T) {
	rows, cols := 20, 20
	// Flat region except a sharp central spike -> high inverse reliability there.
	original := makeGrid(rows, cols, func(i, j int) float64 { return 0.0 })
	original[10][10] = 2.0
	wrapped := WrapPhaseMap(original)

	h := NewHistogramPhaseUnwrapping(DefaultParams(cols, rows))

	// Before any unwrap the map is unavailable.
	if _, err := h.GetInverseReliabilityMap(); err == nil {
		t.Fatalf("expected error before unwrap")
	}

	if _, err := h.UnwrapPhaseMapGrid(wrapped); err != nil {
		t.Fatalf("unwrap: %v", err)
	}
	relMat, err := h.GetInverseReliabilityMap()
	if err != nil {
		t.Fatalf("GetInverseReliabilityMap: %v", err)
	}
	if relMat.Rows != rows || relMat.Cols != cols {
		t.Fatalf("reliability map shape %dx%d, want %dx%d", relMat.Rows, relMat.Cols, rows, cols)
	}
	// The pixels adjacent to the spike must be less reliable (higher value) than
	// a far-away flat pixel.
	near := relMat.At(10, 9)
	far := relMat.At(3, 3)
	if !(near > far) {
		t.Fatalf("expected higher inverse reliability near spike: near=%g far=%g", near, far)
	}

	// The standalone helper must agree with the class output on interior pixels.
	relGrid := ReliabilityMap(wrapped)
	for i := 1; i < rows-1; i++ {
		for j := 1; j < cols-1; j++ {
			if math.Abs(relGrid[i][j]-relMat.At(i, j)) > 1e-12 {
				t.Fatalf("ReliabilityMap disagrees at (%d,%d): %g vs %g",
					i, j, relGrid[i][j], relMat.At(i, j))
			}
		}
	}
}

func TestWrapRange(t *testing.T) {
	for _, x := range []float64{-10, -math.Pi, 0, 1.0, math.Pi, 3 * math.Pi, 100.5} {
		w := Wrap(x)
		if w <= -math.Pi-1e-12 || w > math.Pi+1e-12 {
			t.Fatalf("Wrap(%g)=%g not in (-pi, pi]", x, w)
		}
		// Wrapping is idempotent and congruent mod 2*pi.
		if diff := math.Mod(math.Abs(x-w), twoPi); diff > 1e-9 && math.Abs(diff-twoPi) > 1e-9 {
			t.Fatalf("Wrap(%g)=%g not congruent to x mod 2pi (diff %g)", x, w, diff)
		}
		if math.Abs(Wrap(w)-w) > 1e-12 {
			t.Fatalf("Wrap not idempotent at %g", x)
		}
	}
	// -pi maps to +pi (half-open on the left).
	if math.Abs(Wrap(-math.Pi)-math.Pi) > 1e-12 {
		t.Fatalf("Wrap(-pi) should be +pi, got %g", Wrap(-math.Pi))
	}
}

func TestResidues(t *testing.T) {
	// A smooth ramp has no residues.
	rows, cols := 16, 16
	smooth := WrapPhaseMap(makeGrid(rows, cols, func(i, j int) float64 {
		return 0.3*float64(i) + 0.2*float64(j)
	}))
	if n := CountResidues(smooth); n != 0 {
		t.Fatalf("smooth map should have 0 residues, got %d", n)
	}
	res := Residues(smooth)
	if len(res) != rows-1 || len(res[0]) != cols-1 {
		t.Fatalf("residue grid shape %dx%d, want %dx%d", len(res), len(res[0]), rows-1, cols-1)
	}

	// Construct a phase field with a genuine residue: a spiral phase around a
	// point produces a single +/-1 residue at the centre loop.
	cy, cx := 8.0, 8.0
	spiral := make([][]float64, rows)
	for i := 0; i < rows; i++ {
		spiral[i] = make([]float64, cols)
		for j := 0; j < cols; j++ {
			spiral[i][j] = Wrap(math.Atan2(float64(i)-cy, float64(j)-cx))
		}
	}
	if n := CountResidues(spiral); n == 0 {
		t.Fatalf("spiral phase should contain at least one residue")
	}
}

func TestEmptyInputs(t *testing.T) {
	h := NewHistogramPhaseUnwrapping(Params{}) // zero params -> normalised
	if _, err := h.UnwrapPhaseMapGrid(nil); err != ErrEmptyInput {
		t.Fatalf("nil grid: got %v", err)
	}
	if _, err := h.UnwrapPhaseMapGrid([][]float64{{}}); err != ErrEmptyInput {
		t.Fatalf("empty row: got %v", err)
	}
	if _, err := h.UnwrapPhaseMapGrid([][]float64{{1, 2}, {3}}); err != ErrEmptyInput {
		t.Fatalf("ragged grid: got %v", err)
	}
	if _, err := h.UnwrapPhaseMap(nil); err != ErrEmptyInput {
		t.Fatalf("nil mat: got %v", err)
	}
	if _, err := h.UnwrapPhaseMap(&cv.FloatMat{}); err != ErrEmptyInput {
		t.Fatalf("empty mat: got %v", err)
	}
	if _, err := ReliabilityMapMat(nil); err != ErrEmptyInput {
		t.Fatalf("nil rel mat: got %v", err)
	}
	if ReliabilityMap(nil) != nil {
		t.Fatalf("ReliabilityMap(nil) should be nil")
	}
	if Residues([][]float64{{1}}) != nil {
		t.Fatalf("Residues of tiny map should be nil")
	}
}

func TestParamsNormalisationAndAccessor(t *testing.T) {
	h := NewHistogramPhaseUnwrapping(Params{Width: 5, Height: 4})
	p := h.Params()
	if p.NrOfSmallBins != 10 || p.NrOfLargeBins != 5 {
		t.Fatalf("bins not defaulted: %+v", p)
	}
	if math.Abs(p.HistogramThresh-3*math.Pi*math.Pi) > 1e-9 {
		t.Fatalf("thresh not defaulted: %g", p.HistogramThresh)
	}
	// After an unwrap, Width/Height reflect the processed map.
	_, err := h.UnwrapPhaseMapGrid(makeGrid(4, 5, func(i, j int) float64 { return 0 }))
	if err != nil {
		t.Fatalf("unwrap: %v", err)
	}
	if h.Params().Width != 5 || h.Params().Height != 4 {
		t.Fatalf("params size not updated: %+v", h.Params())
	}
}

func TestCustomHistogramParams(t *testing.T) {
	// Exercise a configuration where all reliabilities land in the large bins
	// (tiny thresh) and a single small/large bin each, to cover binning guards.
	rows, cols := 18, 18
	original := makeGrid(rows, cols, func(i, j int) float64 {
		return 0.25*float64(i) + 0.3*float64(j)
	})
	wrapped := WrapPhaseMap(original)
	h := NewHistogramPhaseUnwrapping(Params{
		Width: cols, Height: rows,
		HistogramThresh: 1e-9, NrOfSmallBins: 1, NrOfLargeBins: 1,
	})
	got, err := h.UnwrapPhaseMapGrid(wrapped)
	if err != nil {
		t.Fatalf("unwrap: %v", err)
	}
	gridsEqualUpToConstant(t, "custom-hist", got, original, 1e-6)
}

func TestTinyMapNoInterior(t *testing.T) {
	// A 2xN map has no interior pixels; reliabilities are all equal, but the
	// unwrap must still be exact for a residue-free ramp.
	original := makeGrid(2, 6, func(i, j int) float64 { return 0.4*float64(j) + float64(i) })
	got := unwrapGrid(t, original)
	gridsEqualUpToConstant(t, "tiny", got, original, 1e-6)
}
