package structured_light

import (
	"math"
	"testing"
)

// rampLevels builds wrapped phase maps for a normalized coordinate ramp
// t = x/cols at each requested frequency, plus the ground-truth absolute phase
// at the highest frequency.
func rampLevels(rows, cols int, freqs []float64) (levels []FrequencyPhase, truthHigh []float64) {
	n := rows * cols
	truthHigh = make([]float64, n)
	for _, f := range freqs {
		w := make([]float64, n)
		for y := 0; y < rows; y++ {
			for x := 0; x < cols; x++ {
				t := float64(x) / float64(cols)
				abs := 2 * math.Pi * f * t
				w[y*cols+x] = WrapPhase(abs)
			}
		}
		levels = append(levels, FrequencyPhase{Frequency: f, Wrapped: w})
	}
	fHi := freqs[len(freqs)-1]
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			t := float64(x) / float64(cols)
			truthHigh[y*cols+x] = 2 * math.Pi * fHi * t
		}
	}
	return levels, truthHigh
}

func TestMultiFrequencyUnwrapRecoversRamp(t *testing.T) {
	rows, cols := 3, 200
	freqs := []float64{1, 4, 16}
	levels, truth := rampLevels(rows, cols, freqs)

	abs, err := MultiFrequencyUnwrap(levels, rows, cols, false)
	if err != nil {
		t.Fatalf("MultiFrequencyUnwrap error: %v", err)
	}

	maxErr := 0.0
	lo, hi := abs[0], abs[0]
	for i := range abs {
		if e := math.Abs(abs[i] - truth[i]); e > maxErr {
			maxErr = e
		}
		if abs[i] < lo {
			lo = abs[i]
		}
		if abs[i] > hi {
			hi = abs[i]
		}
	}
	if maxErr > 1e-6 {
		t.Fatalf("multi-frequency unwrap max error %.3e exceeds tolerance", maxErr)
	}
	// The recovered range must far exceed 2π (16 fringes ≈ 100 rad).
	if rng := hi - lo; rng < 90 {
		t.Fatalf("recovered range %.2f rad is not >> 2π", rng)
	}
}

func TestHeterodyneUnwrapTwoFrequency(t *testing.T) {
	rows, cols := 2, 128
	freqs := []float64{15, 16}
	levels, truth := rampLevels(rows, cols, freqs)

	abs, err := HeterodyneUnwrap(levels, rows, cols, false)
	if err != nil {
		t.Fatalf("HeterodyneUnwrap error: %v", err)
	}
	maxErr, lo, hi := 0.0, abs[0], abs[0]
	for i := range abs {
		if e := math.Abs(abs[i] - truth[i]); e > maxErr {
			maxErr = e
		}
		if abs[i] < lo {
			lo = abs[i]
		}
		if abs[i] > hi {
			hi = abs[i]
		}
	}
	if maxErr > 1e-6 {
		t.Fatalf("2-freq heterodyne max error %.3e exceeds tolerance", maxErr)
	}
	if rng := hi - lo; rng < 90 {
		t.Fatalf("recovered range %.2f rad is not >> 2π", rng)
	}
}

func TestHeterodyneUnwrapThreeFrequency(t *testing.T) {
	rows, cols := 2, 96
	freqs := []float64{14, 15, 16}
	levels, truth := rampLevels(rows, cols, freqs)

	abs, err := HeterodyneUnwrap(levels, rows, cols, false)
	if err != nil {
		t.Fatalf("HeterodyneUnwrap error: %v", err)
	}
	maxErr := 0.0
	for i := range abs {
		if e := math.Abs(abs[i] - truth[i]); e > maxErr {
			maxErr = e
		}
	}
	if maxErr > 1e-6 {
		t.Fatalf("3-freq heterodyne max error %.3e exceeds tolerance", maxErr)
	}
}

func TestTemporalUnwrapErrors(t *testing.T) {
	good := []float64{0, 0, 0, 0}
	// too few levels
	if _, err := MultiFrequencyUnwrap([]FrequencyPhase{{Frequency: 1, Wrapped: good}}, 2, 2, false); err == nil {
		t.Fatal("expected error for single level")
	}
	// non-increasing frequency
	lv := []FrequencyPhase{{Frequency: 4, Wrapped: good}, {Frequency: 4, Wrapped: good}}
	if _, err := MultiFrequencyUnwrap(lv, 2, 2, false); err == nil {
		t.Fatal("expected error for non-increasing frequency")
	}
	// wrong length
	lv2 := []FrequencyPhase{{Frequency: 1, Wrapped: good}, {Frequency: 2, Wrapped: []float64{0}}}
	if _, err := HeterodyneUnwrap(lv2, 2, 2, false); err == nil {
		t.Fatal("expected error for wrong map length")
	}
	// heterodyne rejects 4 frequencies
	lv4 := make([]FrequencyPhase, 4)
	for i := range lv4 {
		lv4[i] = FrequencyPhase{Frequency: float64(i + 1), Wrapped: good}
	}
	if _, err := HeterodyneUnwrap(lv4, 2, 2, false); err == nil {
		t.Fatal("expected error for 4-frequency heterodyne")
	}
}
