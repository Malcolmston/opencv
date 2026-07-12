package phase_unwrapping_test

import (
	"fmt"
	"math"

	pu "github.com/malcolmston/opencv/phase_unwrapping"
)

// makeSurface builds a smooth residue-free surface spanning more than 2*pi.
func makeSurface(rows, cols int) [][]float64 {
	g := make([][]float64, rows)
	for i := 0; i < rows; i++ {
		g[i] = make([]float64, cols)
		for j := 0; j < cols; j++ {
			g[i][j] = 0.35*float64(i) + 0.28*float64(j)
		}
	}
	return g
}

// recoveredExactly reports whether unwrapped equals original up to a single
// global constant.
func recoveredExactly(unwrapped, original [][]float64, tol float64) bool {
	offset := unwrapped[0][0] - original[0][0]
	for i := range original {
		for j := range original[i] {
			if math.Abs((unwrapped[i][j]-offset)-original[i][j]) > tol {
				return false
			}
		}
	}
	return true
}

// ExampleLeastSquaresUnwrap unwraps a wrapped ramp with the unweighted DCT
// least-squares solver.
func ExampleLeastSquaresUnwrap() {
	original := makeSurface(12, 16)
	wrapped := pu.WrapPhaseMap(original)
	unwrapped, err := pu.LeastSquaresUnwrap(wrapped)
	if err != nil {
		panic(err)
	}
	fmt.Println(recoveredExactly(unwrapped, original, 1e-6))
	// Output: true
}

// ExampleGoldsteinBranchCut shows that the branch-cut result is always congruent
// to the wrapped input, even for a spiral phase singularity that carries a
// residue.
func ExampleGoldsteinBranchCut() {
	const n = 16
	spiral := make([][]float64, n)
	for i := 0; i < n; i++ {
		spiral[i] = make([]float64, n)
		for j := 0; j < n; j++ {
			spiral[i][j] = pu.Wrap(math.Atan2(float64(i)-8, float64(j)-8))
		}
	}
	unwrapped, err := pu.GoldsteinBranchCut(spiral, 0)
	if err != nil {
		panic(err)
	}
	// Re-wrapping the result reproduces the input.
	rw := pu.Rewrap(unwrapped)
	maxErr := 0.0
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			e := math.Abs(pu.Wrap(rw[i][j] - spiral[i][j]))
			if e > maxErr {
				maxErr = e
			}
		}
	}
	fmt.Printf("residues=%v congruent=%v\n", pu.CountResidues(spiral) > 0, maxErr < 1e-9)
	// Output: residues=true congruent=true
}

// ExampleTemporalUnwrap unwraps a multi-frequency acquisition hierarchically from
// the coarsest (unambiguous) map to the finest.
func ExampleTemporalUnwrap() {
	rows, cols := 8, 8
	scales := []float64{1, 3, 9}
	base := make([][]float64, rows)
	for i := 0; i < rows; i++ {
		base[i] = make([]float64, cols)
		for j := 0; j < cols; j++ {
			base[i][j] = 0.8 * math.Pi * (float64(i)/float64(rows-1) - 0.5)
		}
	}
	maps := make([][][]float64, len(scales))
	for k, s := range scales {
		m := make([][]float64, rows)
		for i := 0; i < rows; i++ {
			m[i] = make([]float64, cols)
			for j := 0; j < cols; j++ {
				m[i][j] = pu.Wrap(s * base[i][j])
			}
		}
		maps[k] = m
	}
	unwrapped, err := pu.TemporalUnwrap(maps, scales)
	if err != nil {
		panic(err)
	}
	expected := make([][]float64, rows)
	for i := 0; i < rows; i++ {
		expected[i] = make([]float64, cols)
		for j := 0; j < cols; j++ {
			expected[i][j] = scales[len(scales)-1] * base[i][j]
		}
	}
	fmt.Println(recoveredExactly(unwrapped, expected, 1e-9))
	// Output: true
}

// ExampleUnwrapper selects an unwrapping method through the common interface.
func ExampleUnwrapper() {
	original := makeSurface(10, 10)
	wrapped := pu.WrapPhaseMap(original)

	var method pu.Unwrapper = pu.QualityGuidedUnwrapper{}
	unwrapped, err := method.Unwrap(wrapped)
	if err != nil {
		panic(err)
	}
	fmt.Println(recoveredExactly(unwrapped, original, 1e-6))
	// Output: true
}

// ExampleMaskedUnwrap unwraps only the pixels inside a mask, leaving masked-out
// pixels as NaN.
func ExampleMaskedUnwrap() {
	rows, cols := 10, 10
	original := makeSurface(rows, cols)
	wrapped := pu.WrapPhaseMap(original)
	mask := make([][]bool, rows)
	for i := 0; i < rows; i++ {
		mask[i] = make([]bool, cols)
		for j := 0; j < cols; j++ {
			mask[i][j] = true
		}
	}
	mask[0][0] = false // exclude one corner

	unwrapped, err := pu.MaskedUnwrap(wrapped, mask, nil)
	if err != nil {
		panic(err)
	}
	fmt.Printf("corner-nan=%v interior-ok=%v\n",
		math.IsNaN(unwrapped[0][0]),
		!math.IsNaN(unwrapped[5][5]))
	// Output: corner-nan=true interior-ok=true
}
