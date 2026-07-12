package structured_light

import (
	"fmt"
	"math"

	cv "github.com/malcolmston/opencv"
)

// phaseSums accumulates, per pixel, the three quantities every phase-shift
// quality metric is built from: the DC sum Σ Iᵢ, and the in-/out-of-phase
// correlations S = Σ Iᵢ·sin(iδ) and C = Σ Iᵢ·cos(iδ). It returns them along with
// the pixel count and step actually used. shift zero selects the canonical
// 2π/N step. It panics on an inconsistent stack (fewer than three images or a
// size mismatch).
func phaseSums(captured []*cv.Mat, shift float64) (dc, s, c []float64, n int) {
	n = len(captured)
	if n < 3 {
		panic(fmt.Sprintf("structured_light: phase quality requires >=3 images, got %d", n))
	}
	if captured[0] == nil {
		panic("structured_light: phase quality captured[0] is nil")
	}
	if shift == 0 {
		shift = 2 * math.Pi / float64(n)
	}
	rows, cols := captured[0].Rows, captured[0].Cols
	grays := make([]*cv.Mat, n)
	for i, m := range captured {
		if m == nil {
			panic(fmt.Sprintf("structured_light: phase quality captured[%d] is nil", i))
		}
		if m.Rows != rows || m.Cols != cols {
			panic(fmt.Sprintf("structured_light: phase quality captured[%d] is %dx%d, want %dx%d", i, m.Rows, m.Cols, rows, cols))
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
	dc = make([]float64, np)
	s = make([]float64, np)
	c = make([]float64, np)
	for p := 0; p < np; p++ {
		var sumDC, sumS, sumC float64
		for i := 0; i < n; i++ {
			v := float64(grays[i].Data[p])
			sumDC += v
			sumS += v * sinD[i]
			sumC += v * cosD[i]
		}
		dc[p], s[p], c[p] = sumDC, sumS, sumC
	}
	return dc, s, c, n
}

// ComputeBackground returns the per-pixel background (DC / bias) intensity of an
// N-step phase-shifted stack, i.e. the mean of the captured images. shift zero
// selects the canonical 2π/N step (the background does not depend on the step,
// but the argument keeps the quality-metric signatures uniform). The result is a
// row-major []float64. It panics on an inconsistent stack.
func ComputeBackground(captured []*cv.Mat, shift float64) []float64 {
	dc, _, _, n := phaseSums(captured, shift)
	out := make([]float64, len(dc))
	for i, v := range dc {
		out[i] = v / float64(n)
	}
	return out
}

// ComputeAmplitude returns the per-pixel modulation amplitude B = (2/N)·√(S²+C²)
// of an N-step phase-shifted stack, where S and C are the sine/cosine
// correlations of the captured intensities with the projected phase. It is the
// contrast of the recovered fringe and, together with [ComputeBackground],
// characterizes signal strength. shift zero selects the canonical 2π/N step. The
// result is a row-major []float64. It panics on an inconsistent stack.
func ComputeAmplitude(captured []*cv.Mat, shift float64) []float64 {
	_, s, c, n := phaseSums(captured, shift)
	out := make([]float64, len(s))
	k := 2.0 / float64(n)
	for i := range s {
		out[i] = k * math.Hypot(s[i], c[i])
	}
	return out
}

// ComputeDataModulation returns the per-pixel data modulation (fringe
// visibility) γ = B/A of an N-step phase-shifted stack, the ratio of the
// modulation amplitude B to the background A. It lies in [0,1] for a physical
// capture and is the standard confidence measure for phase-shifting
// profilometry: near 1 where fringes are crisp, near 0 in shadow or on
// saturated/textureless surfaces. Pixels with zero background yield 0. shift
// zero selects the canonical 2π/N step. The result is a row-major []float64. It
// panics on an inconsistent stack.
func ComputeDataModulation(captured []*cv.Mat, shift float64) []float64 {
	dc, s, c, n := phaseSums(captured, shift)
	out := make([]float64, len(s))
	nf := float64(n)
	for i := range s {
		bg := dc[i] / nf
		if bg <= 0 {
			continue
		}
		amp := (2.0 / nf) * math.Hypot(s[i], c[i])
		out[i] = amp / bg
	}
	return out
}

// ShadowMask builds a boolean lit/shadow mask from an all-white and an all-black
// reference capture: a pixel is lit (true) when white−black exceeds thresh. This
// is the robust ambient-light rejection used before decoding a Gray-code or
// phase stack. The references must be single- or multi-channel of identical
// size. The result is row-major of length Rows*Cols. It panics on a size
// mismatch.
func ShadowMask(white, black *cv.Mat, thresh int) []bool {
	if white == nil || black == nil {
		panic("structured_light: ShadowMask requires non-nil references")
	}
	if white.Rows != black.Rows || white.Cols != black.Cols {
		panic("structured_light: ShadowMask reference size mismatch")
	}
	wg := toGray(white)
	bg := toGray(black)
	out := make([]bool, len(wg.Data))
	for i := range wg.Data {
		out[i] = int(wg.Data[i])-int(bg.Data[i]) > thresh
	}
	return out
}

// OverexposureMask flags pixels that are saturated (sample ≥ sat) in any image
// of a stack; such pixels clip the sinusoid and corrupt the recovered phase, so
// they are excluded from decoding. Every image must share the size of the first.
// The result is row-major of length Rows*Cols, true where overexposed. It panics
// on an empty stack or a size mismatch.
func OverexposureMask(images []*cv.Mat, sat uint8) []bool {
	if len(images) == 0 || images[0] == nil {
		panic("structured_light: OverexposureMask requires a non-empty stack")
	}
	rows, cols := images[0].Rows, images[0].Cols
	grays := make([]*cv.Mat, len(images))
	for i, m := range images {
		if m == nil || m.Rows != rows || m.Cols != cols {
			panic(fmt.Sprintf("structured_light: OverexposureMask image %d size mismatch", i))
		}
		grays[i] = toGray(m)
	}
	out := make([]bool, rows*cols)
	for i := range out {
		for _, g := range grays {
			if g.Data[i] >= sat {
				out[i] = true
				break
			}
		}
	}
	return out
}

// ModulationMask thresholds a data-modulation (or amplitude) map: a pixel is
// valid (true) when its value is at least minMod. Combine it with [ShadowMask]
// and the negation of [OverexposureMask] via [CombineMasks] to form the final
// decoding mask. The result has the length of mod.
func ModulationMask(mod []float64, minMod float64) []bool {
	out := make([]bool, len(mod))
	for i, v := range mod {
		out[i] = v >= minMod
	}
	return out
}

// CombineMasks returns the element-wise logical AND of one or more boolean masks
// of equal length; a pixel is valid only where every input marks it valid. With
// no arguments it returns nil. It panics if the masks differ in length.
func CombineMasks(masks ...[]bool) []bool {
	if len(masks) == 0 {
		return nil
	}
	n := len(masks[0])
	for _, m := range masks {
		if len(m) != n {
			panic("structured_light: CombineMasks length mismatch")
		}
	}
	out := make([]bool, n)
	for i := range out {
		out[i] = true
		for _, m := range masks {
			if !m[i] {
				out[i] = false
				break
			}
		}
	}
	return out
}
