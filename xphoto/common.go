package xphoto

import (
	"fmt"
	"math"

	cv "github.com/malcolmston/opencv"
)

// clampU8 rounds v to the nearest integer and clamps it into [0,255].
func clampU8(v float64) uint8 {
	v = math.Round(v)
	if v <= 0 {
		return 0
	}
	if v >= 255 {
		return 255
	}
	return uint8(v)
}

// atRep returns the sample at (y, x, c) clamping out-of-range coordinates to the
// nearest edge (OpenCV's BORDER_REPLICATE). It mirrors the root package's
// unexported atReplicate, which is not accessible from this subpackage.
func atRep(m *cv.Mat, y, x, c int) uint8 {
	if y < 0 {
		y = 0
	} else if y >= m.Rows {
		y = m.Rows - 1
	}
	if x < 0 {
		x = 0
	} else if x >= m.Cols {
		x = m.Cols - 1
	}
	return m.At(y, x, c)
}

// requireNonEmpty panics if m is nil or has no samples.
func requireNonEmpty(m *cv.Mat, name string) {
	if m == nil || m.Empty() {
		panic(fmt.Sprintf("xphoto: %s given an empty image", name))
	}
}

// requireChannels panics if src does not have exactly want channels.
func requireChannels(src *cv.Mat, want int, name string) {
	if src.Channels != want {
		panic(fmt.Sprintf("xphoto: %s requires %d channels, got %d", name, want, src.Channels))
	}
}

// requireSameSize panics if a and b differ in rows or columns.
func requireSameSize(a, b *cv.Mat, name string) {
	if a.Rows != b.Rows || a.Cols != b.Cols {
		panic(fmt.Sprintf("xphoto: %s requires matching sizes, got %dx%d and %dx%d",
			name, a.Rows, a.Cols, b.Rows, b.Cols))
	}
}

// luma returns the BT.601 luma of an RGB sample triple.
func luma(r, g, b float64) float64 {
	return 0.299*r + 0.587*g + 0.114*b
}

// median returns the median of vals, which must be non-empty. The input slice
// is reordered.
func median(vals []float64) float64 {
	n := len(vals)
	// Simple insertion sort keeps it deterministic and dependency-free.
	for i := 1; i < n; i++ {
		v := vals[i]
		j := i - 1
		for j >= 0 && vals[j] > v {
			vals[j+1] = vals[j]
			j--
		}
		vals[j+1] = v
	}
	if n%2 == 1 {
		return vals[n/2]
	}
	return 0.5 * (vals[n/2-1] + vals[n/2])
}
