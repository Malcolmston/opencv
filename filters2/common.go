package filters2

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// requireNonEmpty panics unless m is a usable, non-empty matrix.
func requireNonEmpty(m *cv.Mat, op string) {
	if m == nil || m.Empty() {
		panic("filters2: " + op + ": empty source matrix")
	}
}

// requireGray panics unless m is a usable single-channel matrix.
func requireGray(m *cv.Mat, op string) {
	requireNonEmpty(m, op)
	if m.Channels != 1 {
		panic("filters2: " + op + ": operation requires a single-channel matrix")
	}
}

// requireSameShape panics unless a and b have identical dimensions and channel
// counts.
func requireSameShape(a, b *cv.Mat, op string) {
	requireNonEmpty(a, op)
	requireNonEmpty(b, op)
	if a.Rows != b.Rows || a.Cols != b.Cols || a.Channels != b.Channels {
		panic("filters2: " + op + ": matrices must have identical shape")
	}
}

// clampIdx clamps i to [0, n-1] implementing edge replication (BORDER_REPLICATE).
func clampIdx(i, n int) int {
	if i < 0 {
		return 0
	}
	if i >= n {
		return n - 1
	}
	return i
}

// clampU8 rounds v to the nearest integer and clamps it to the [0,255] range.
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

// atReplicate returns the sample of m at (y, x, c) with out-of-range
// coordinates clamped to the nearest edge.
func atReplicate(m *cv.Mat, y, x, c int) uint8 {
	y = clampIdx(y, m.Rows)
	x = clampIdx(x, m.Cols)
	return m.Data[(y*m.Cols+x)*m.Channels+c]
}

// requireOddPositive panics unless k is a positive odd integer.
func requireOddPositive(k int, op string) {
	if k <= 0 || k%2 == 0 {
		panic("filters2: " + op + ": window size must be a positive odd integer")
	}
}

// gaussianRadius returns a sensible truncation radius (ceil(3*sigma)) for a
// Gaussian of the given standard deviation, never smaller than 1.
func gaussianRadius(sigma float64) int {
	r := int(math.Ceil(3 * sigma))
	if r < 1 {
		r = 1
	}
	return r
}

// like allocates a zeroed matrix matching m's dimensions and channel count.
func like(m *cv.Mat) *cv.Mat { return cv.NewMat(m.Rows, m.Cols, m.Channels) }

// maxChannels is the largest channel count supported by the routines that use
// fixed-size per-pixel accumulators.
const maxChannels = 8

// requireChannels panics unless m has between 1 and maxChannels channels.
func requireChannels(m *cv.Mat, op string) {
	if m.Channels < 1 || m.Channels > maxChannels {
		panic("filters2: " + op + ": channel count must be between 1 and 8")
	}
}
