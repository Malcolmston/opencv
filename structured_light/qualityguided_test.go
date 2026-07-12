package structured_light

import (
	"math"
	"testing"
)

func TestQualityGuidedUnwrapTiltedPlane(t *testing.T) {
	// Absolute phase is a steep tilted plane whose total range far exceeds 2π;
	// per-pixel gradients stay below π so the nearest-2π rule is exact.
	rows, cols := 20, 20
	abs := make([]float64, rows*cols)
	wrapped := make([]float64, rows*cols)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			i := y*cols + x
			a := 0.8*float64(x) + 0.6*float64(y)
			abs[i] = a
			wrapped[i] = WrapPhase(a)
		}
	}
	quality := PhaseGradientQuality(wrapped, rows, cols)
	out := QualityGuidedUnwrap(wrapped, quality, rows, cols)

	// out must equal abs up to a single global constant.
	offset := out[0] - abs[0]
	maxErr := 0.0
	lo, hi := out[0], out[0]
	for i := range out {
		if e := math.Abs((out[i] - abs[i]) - offset); e > maxErr {
			maxErr = e
		}
		if out[i] < lo {
			lo = out[i]
		}
		if out[i] > hi {
			hi = out[i]
		}
	}
	if maxErr > 1e-9 {
		t.Fatalf("quality-guided unwrap not constant-offset from truth: err %.3e", maxErr)
	}
	if rng := hi - lo; rng < 2*math.Pi {
		t.Fatalf("recovered range %.2f should exceed 2π", rng)
	}
}
