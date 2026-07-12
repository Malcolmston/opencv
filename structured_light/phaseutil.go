package structured_light

import (
	"fmt"
	"math"

	cv "github.com/malcolmston/opencv"
)

// WrapPhase brings an arbitrary phase angle into the canonical interval
// (-π, π]. It is the inverse notion of phase unwrapping: whereas unwrapping
// removes 2π jumps, WrapPhase reintroduces the principal-value wrap. It is used
// throughout the multi-frequency and hybrid routines to compare phases modulo
// 2π.
func WrapPhase(a float64) float64 {
	w := math.Mod(a, 2*math.Pi)
	if w <= -math.Pi {
		w += 2 * math.Pi
	}
	if w > math.Pi {
		w -= 2 * math.Pi
	}
	return w
}

// PhaseToCoord converts an absolute (unwrapped) phase map into projector
// coordinates. For a fringe pattern carrying freq full periods across an image
// of the given extent (Width for vertical fringes, Height for horizontal), the
// absolute phase φ at a pixel corresponds to the projector coordinate
// φ·extent/(2π·freq). The result is a new []float64 of the same length. It
// panics if freq is not positive or extent is not positive.
func PhaseToCoord(abs []float64, freq float64, extent int) []float64 {
	if freq <= 0 {
		panic("structured_light: PhaseToCoord requires freq>0")
	}
	if extent <= 0 {
		panic("structured_light: PhaseToCoord requires extent>0")
	}
	scale := float64(extent) / (2 * math.Pi * freq)
	out := make([]float64, len(abs))
	for i, v := range abs {
		out[i] = v * scale
	}
	return out
}

// subtractWrap returns the element-wise wrapped difference WrapPhase(a-b). The
// slices must share a length; it panics otherwise. This is the basic beat
// operation of heterodyne unwrapping.
func subtractWrap(a, b []float64) []float64 {
	if len(a) != len(b) {
		panic("structured_light: subtractWrap length mismatch")
	}
	out := make([]float64, len(a))
	for i := range a {
		out[i] = WrapPhase(a[i] - b[i])
	}
	return out
}

// unwrapWithReference performs one temporal-unwrapping step: it removes the 2π
// ambiguity of a higher-frequency wrapped phase map using an already-absolute
// lower-frequency reference. For each pixel the reference absolute phase is
// scaled by freq/refFreq to predict the absolute phase at the target frequency,
// the integer fringe order is the rounded difference from the wrapped value, and
// the unwrapped phase is wrapped + 2π·order. This is per-pixel and therefore
// immune to the spatial error propagation of line-by-line unwrapping.
func unwrapWithReference(ref []float64, refFreq float64, wrapped []float64, freq float64) []float64 {
	scale := freq / refFreq
	out := make([]float64, len(wrapped))
	for i := range wrapped {
		est := ref[i] * scale
		k := math.Round((est - wrapped[i]) / (2 * math.Pi))
		out[i] = wrapped[i] + 2*math.Pi*k
	}
	return out
}

// NStepWrappedPhase computes the wrapped phase of an N-step phase-shifted stack
// with an arbitrary, caller-supplied uniform phase step, generalizing
// [SinusoidalPattern.ComputeWrappedPhase] to any N≥3 and any step. The estimator
// is the standard least-squares solution
//
//	φ = atan2( -Σ Iᵢ·sin(i·shift), Σ Iᵢ·cos(i·shift) )
//
// If shift is zero the canonical step 2π/N is used. captured must hold at least
// three images of identical size; each may be single- or multi-channel (reduced
// to luma). The result is a row-major []float64 in (-π, π]. It panics on an
// inconsistent stack.
func NStepWrappedPhase(captured []*cv.Mat, shift float64) []float64 {
	n := len(captured)
	if n < 3 {
		panic(fmt.Sprintf("structured_light: NStepWrappedPhase requires >=3 images, got %d", n))
	}
	if captured[0] == nil {
		panic("structured_light: NStepWrappedPhase captured[0] is nil")
	}
	if shift == 0 {
		shift = 2 * math.Pi / float64(n)
	}
	rows, cols := captured[0].Rows, captured[0].Cols
	grays := make([]*cv.Mat, n)
	for i, m := range captured {
		if m == nil {
			panic(fmt.Sprintf("structured_light: NStepWrappedPhase captured[%d] is nil", i))
		}
		if m.Rows != rows || m.Cols != cols {
			panic(fmt.Sprintf("structured_light: NStepWrappedPhase captured[%d] is %dx%d, want %dx%d", i, m.Rows, m.Cols, rows, cols))
		}
		grays[i] = toGray(m)
	}
	sinD := make([]float64, n)
	cosD := make([]float64, n)
	for i := 0; i < n; i++ {
		d := float64(i) * shift
		sinD[i] = math.Sin(d)
		cosD[i] = math.Cos(d)
	}
	np := rows * cols
	phase := make([]float64, np)
	for p := 0; p < np; p++ {
		var num, den float64
		for i := 0; i < n; i++ {
			v := float64(grays[i].Data[p])
			num += v * sinD[i]
			den += v * cosD[i]
		}
		phase[p] = math.Atan2(-num, den)
	}
	return phase
}

// CombineGrayAndPhase fuses a coarse integer fringe order (recovered from a
// Gray-code or binary stack) with a fine wrapped phase (recovered from a
// phase-shift stack) into a single continuous absolute phase, the standard
// Gray-code-plus-phase-shift hybrid. For fringe order m and wrapped phase w the
// absolute phase is w + 2π·k where k = round((2π·m - w)/2π); the rounding makes
// the fusion robust to half-fringe misregistration between the two stacks. The
// slices must share a length; it panics otherwise.
func CombineGrayAndPhase(fringeOrder []int, wrapped []float64) []float64 {
	if len(fringeOrder) != len(wrapped) {
		panic("structured_light: CombineGrayAndPhase length mismatch")
	}
	out := make([]float64, len(wrapped))
	for i := range wrapped {
		coarse := 2 * math.Pi * float64(fringeOrder[i])
		k := math.Round((coarse - wrapped[i]) / (2 * math.Pi))
		out[i] = wrapped[i] + 2*math.Pi*k
	}
	return out
}

// PhaseGradientQuality builds a quality map for a wrapped phase field, suitable
// for [QualityGuidedUnwrap]. Each pixel's quality is 1/(1+g), where g is the
// largest wrapped phase difference to its 4-connected neighbours; smooth regions
// score near 1 and residue-prone discontinuities score lower. The input is a
// row-major wrapped phase of length rows*cols; the output is a new slice of the
// same length. It panics on a size mismatch.
func PhaseGradientQuality(wrapped []float64, rows, cols int) []float64 {
	if len(wrapped) != rows*cols {
		panic("structured_light: PhaseGradientQuality length != rows*cols")
	}
	q := make([]float64, len(wrapped))
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			i := y*cols + x
			g := 0.0
			if x+1 < cols {
				if d := math.Abs(WrapPhase(wrapped[i+1] - wrapped[i])); d > g {
					g = d
				}
			}
			if x > 0 {
				if d := math.Abs(WrapPhase(wrapped[i-1] - wrapped[i])); d > g {
					g = d
				}
			}
			if y+1 < rows {
				if d := math.Abs(WrapPhase(wrapped[i+cols] - wrapped[i])); d > g {
					g = d
				}
			}
			if y > 0 {
				if d := math.Abs(WrapPhase(wrapped[i-cols] - wrapped[i])); d > g {
					g = d
				}
			}
			q[i] = 1.0 / (1.0 + g)
		}
	}
	return q
}
