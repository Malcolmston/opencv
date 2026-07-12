package structured_light

import "fmt"

// FrequencyPhase pairs a wrapped phase map with the number of full fringe
// periods (Frequency) its projected pattern spanned across the varying image
// direction. It is the unit of input to the temporal phase-unwrapping routines
// [MultiFrequencyUnwrap] and [HeterodyneUnwrap]. Wrapped is a row-major
// []float64 in (-π, π]; Frequency must be positive.
type FrequencyPhase struct {
	// Frequency is the number of sinusoid periods the pattern spanned.
	Frequency float64
	// Wrapped is the row-major wrapped phase map for that frequency.
	Wrapped []float64
}

// MultiFrequencyUnwrap performs hierarchical (temporal) phase unwrapping across
// a set of frequencies, resolving a fine, high-frequency phase whose absolute
// range far exceeds 2π. The levels must be ordered by strictly increasing
// Frequency and share the length rows*cols.
//
// The lowest-frequency map is unwrapped spatially with [UnwrapPhaseMap] to form
// an unambiguous reference; every higher frequency is then unwrapped per-pixel
// against the running absolute phase via the fringe-order formula (see
// [FrequencyPhase]). Because the higher levels are unwrapped point-by-point,
// the method tolerates noise and true 2π-per-pixel gradients that defeat a
// purely spatial unwrap. The returned slice is the absolute phase at the
// highest frequency.
//
// It returns an error if fewer than two levels are given, the frequencies are
// not strictly increasing or non-positive, or any map has the wrong length.
func MultiFrequencyUnwrap(levels []FrequencyPhase, rows, cols int, horizontal bool) ([]float64, error) {
	if err := validateLevels(levels, rows, cols); err != nil {
		return nil, err
	}
	abs := UnwrapPhaseMap(levels[0].Wrapped, rows, cols, horizontal)
	prevFreq := levels[0].Frequency
	for k := 1; k < len(levels); k++ {
		abs = unwrapWithReference(abs, prevFreq, levels[k].Wrapped, levels[k].Frequency)
		prevFreq = levels[k].Frequency
	}
	return abs, nil
}

// HeterodyneUnwrap unwraps two or three frequencies with the heterodyne (beat)
// method. Rather than scaling frequency ratios directly, it synthesizes a
// low-frequency beat by wrapping the difference of two higher-frequency phases;
// a beat spanning about one fringe over the whole field is unambiguous and
// guides the finer levels. The levels must be ordered by strictly increasing
// Frequency and share the length rows*cols. The returned slice is the absolute
// phase at the highest frequency.
//
// For two levels the beat is wrap(φ_hi − φ_lo) at frequency f_hi−f_lo; it is
// spatially unwrapped and used to resolve the high frequency. For three levels
// the beat between the two lowest frequencies resolves the middle frequency,
// which in turn resolves the highest. It returns an error unless exactly two or
// three valid, increasing levels are supplied.
func HeterodyneUnwrap(levels []FrequencyPhase, rows, cols int, horizontal bool) ([]float64, error) {
	if err := validateLevels(levels, rows, cols); err != nil {
		return nil, err
	}
	switch len(levels) {
	case 2:
		lo, hi := levels[0], levels[1]
		beat := subtractWrap(hi.Wrapped, lo.Wrapped)
		beatFreq := hi.Frequency - lo.Frequency
		beatAbs := UnwrapPhaseMap(beat, rows, cols, horizontal)
		return unwrapWithReference(beatAbs, beatFreq, hi.Wrapped, hi.Frequency), nil
	case 3:
		a, b, c := levels[0], levels[1], levels[2]
		beat := subtractWrap(b.Wrapped, a.Wrapped)
		beatFreq := b.Frequency - a.Frequency
		beatAbs := UnwrapPhaseMap(beat, rows, cols, horizontal)
		absB := unwrapWithReference(beatAbs, beatFreq, b.Wrapped, b.Frequency)
		absC := unwrapWithReference(absB, b.Frequency, c.Wrapped, c.Frequency)
		return absC, nil
	default:
		return nil, fmt.Errorf("structured_light: HeterodyneUnwrap supports 2 or 3 frequencies, got %d", len(levels))
	}
}

// validateLevels checks a level set for the common preconditions of the
// temporal-unwrapping routines.
func validateLevels(levels []FrequencyPhase, rows, cols int) error {
	if len(levels) < 2 {
		return fmt.Errorf("structured_light: need at least 2 frequency levels, got %d", len(levels))
	}
	n := rows * cols
	for i, l := range levels {
		if l.Frequency <= 0 {
			return fmt.Errorf("structured_light: level %d has non-positive frequency %g", i, l.Frequency)
		}
		if len(l.Wrapped) != n {
			return fmt.Errorf("structured_light: level %d map length %d != rows*cols %d", i, len(l.Wrapped), n)
		}
		if i > 0 && l.Frequency <= levels[i-1].Frequency {
			return fmt.Errorf("structured_light: frequencies must strictly increase; level %d (%g) <= level %d (%g)", i, l.Frequency, i-1, levels[i-1].Frequency)
		}
	}
	return nil
}
