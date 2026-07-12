package phase_unwrapping_test

import (
	"fmt"
	"math"

	cv "github.com/malcolmston/opencv"
	pu "github.com/malcolmston/opencv/phase_unwrapping"
)

// ExampleHistogramPhaseUnwrapping_UnwrapPhaseMapGrid unwraps a tilted-ramp phase
// surface whose total range exceeds 2*pi. Because the surface is residue-free it
// is recovered exactly, up to a single global 2*pi constant.
func ExampleHistogramPhaseUnwrapping_UnwrapPhaseMapGrid() {
	const rows, cols = 8, 8

	// A continuous surface spanning more than 2*pi.
	original := make([][]float64, rows)
	for i := 0; i < rows; i++ {
		original[i] = make([]float64, cols)
		for j := 0; j < cols; j++ {
			original[i][j] = 0.6*float64(i) + 0.5*float64(j)
		}
	}

	// Wrap it into (-pi, pi], the only thing a real sensor measures.
	wrapped := pu.WrapPhaseMap(original)

	// Unwrap.
	h := pu.NewHistogramPhaseUnwrapping(pu.DefaultParams(cols, rows))
	unwrapped, err := h.UnwrapPhaseMapGrid(wrapped)
	if err != nil {
		panic(err)
	}

	// Remove the arbitrary global offset and compare to the original.
	offset := unwrapped[0][0] - original[0][0]
	maxErr := 0.0
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			e := math.Abs((unwrapped[i][j] - offset) - original[i][j])
			if e > maxErr {
				maxErr = e
			}
		}
	}
	fmt.Printf("residues=%d recovered-exactly=%v\n", pu.CountResidues(wrapped), maxErr < 1e-9)
	// Output: residues=0 recovered-exactly=true
}

// ExampleHistogramPhaseUnwrapping_UnwrapPhaseMap shows the FloatMat-based API.
func ExampleHistogramPhaseUnwrapping_UnwrapPhaseMap() {
	const rows, cols = 6, 6
	wrapped := cv.NewFloatMat(rows, cols)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			wrapped.Data[i*cols+j] = pu.Wrap(0.7*float64(i) + 0.4*float64(j))
		}
	}

	h := pu.NewHistogramPhaseUnwrapping(pu.DefaultParams(cols, rows))
	out, err := h.UnwrapPhaseMap(wrapped)
	if err != nil {
		panic(err)
	}
	fmt.Printf("output %dx%d\n", out.Rows, out.Cols)
	// Output: output 6x6
}

// ExampleResidues detects that a smooth surface is residue-free while a spiral
// phase singularity carries a residue.
func ExampleResidues() {
	const n = 8
	smooth := make([][]float64, n)
	spiral := make([][]float64, n)
	for i := 0; i < n; i++ {
		smooth[i] = make([]float64, n)
		spiral[i] = make([]float64, n)
		for j := 0; j < n; j++ {
			smooth[i][j] = pu.Wrap(0.3 * float64(i+j))
			spiral[i][j] = pu.Wrap(math.Atan2(float64(i)-4, float64(j)-4))
		}
	}
	fmt.Printf("smooth=%d spiral>0=%v\n", pu.CountResidues(smooth), pu.CountResidues(spiral) > 0)
	// Output: smooth=0 spiral>0=true
}
