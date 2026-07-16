package cudaoptflow_test

import (
	"fmt"
	"math"

	cv "github.com/malcolmston/opencv"
	"github.com/malcolmston/opencv/cudaoptflow"
)

// Example estimates dense optical flow between two synthetic frames whose scene
// content is translated by a known (dx, dy) = (2, 1) offset. The frames are
// uploaded to device matrices, Farneback flow is computed on the (CPU-backed)
// device, and the interior mean flow is reported rounded to whole pixels.
func Example() {
	prev, next := exampleShifted(64, 64, 2, 1)
	flow := cudaoptflow.NewFarnebackOpticalFlow(4, 4).Calc(
		cudaoptflow.GpuMatFromMat(prev),
		cudaoptflow.GpuMatFromMat(next),
		cudaoptflow.NewStream(),
	)
	u, v := flow.MeanFlow(8)
	fmt.Printf("mean flow: %.0f %.0f\n", math.Round(u), math.Round(v))
	// Output: mean flow: 2 1
}

// exampleShifted returns a prev/next grayscale pair of size h x w whose broadband
// texture is translated by (dx, dy), so the true flow from prev to next is
// (dx, dy) everywhere. A margin keeps every sampled pixel inside the base
// pattern.
func exampleShifted(h, w, dx, dy int) (prev, next *cv.Mat) {
	const margin = 12
	base := examplePattern(h+2*margin, w+2*margin)
	prev = cv.NewMat(h, w, 1)
	next = cv.NewMat(h, w, 1)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			prev.Set(y, x, 0, uint8(base[y+margin][x+margin]+0.5))
			next.Set(y, x, 0, uint8(base[y+margin-dy][x+margin-dx]+0.5))
		}
	}
	return prev, next
}

// examplePattern builds a deterministic, broadband, locally-unique grayscale
// image (hashed value noise, lightly smoothed and rescaled to 0..255) so shifted
// crops give block-matching estimators strong, non-periodic gradients to lock
// onto.
func examplePattern(rows, cols int) [][]float64 {
	b := make([][]float64, rows)
	for r := 0; r < rows; r++ {
		b[r] = make([]float64, cols)
		for c := 0; c < cols; c++ {
			hsh := uint32(r)*73856093 ^ uint32(c)*19349663 ^ 0x9e3779b9
			hsh *= 2654435761
			hsh ^= hsh >> 15
			b[r][c] = float64(hsh & 0xff)
		}
	}
	for pass := 0; pass < 2; pass++ {
		b = exampleBoxBlur3(b)
	}
	lo, hi := b[0][0], b[0][0]
	for r := range b {
		for c := range b[r] {
			if b[r][c] < lo {
				lo = b[r][c]
			}
			if b[r][c] > hi {
				hi = b[r][c]
			}
		}
	}
	if hi > lo {
		for r := range b {
			for c := range b[r] {
				b[r][c] = (b[r][c] - lo) / (hi - lo) * 255
			}
		}
	}
	return b
}

// exampleBoxBlur3 averages each sample with its 3x3 neighbourhood, clamped at the
// borders.
func exampleBoxBlur3(in [][]float64) [][]float64 {
	rows, cols := len(in), len(in[0])
	out := make([][]float64, rows)
	for r := 0; r < rows; r++ {
		out[r] = make([]float64, cols)
		for c := 0; c < cols; c++ {
			var sum float64
			var n int
			for dr := -1; dr <= 1; dr++ {
				for dc := -1; dc <= 1; dc++ {
					rr, cc := r+dr, c+dc
					if rr < 0 || rr >= rows || cc < 0 || cc >= cols {
						continue
					}
					sum += in[rr][cc]
					n++
				}
			}
			out[r][c] = sum / float64(n)
		}
	}
	return out
}
