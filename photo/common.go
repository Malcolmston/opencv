package photo

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

// requireChannels panics if src does not have exactly want channels.
func requireChannels(src *cv.Mat, want int, name string) {
	if src.Channels != want {
		panic(fmt.Sprintf("photo: %s requires %d channels, got %d", name, want, src.Channels))
	}
}

// requireSameSize panics if a and b differ in rows or columns.
func requireSameSize(a, b *cv.Mat, name string) {
	if a.Rows != b.Rows || a.Cols != b.Cols {
		panic(fmt.Sprintf("photo: %s requires matching sizes, got %dx%d and %dx%d",
			name, a.Rows, a.Cols, b.Rows, b.Cols))
	}
}

// oddAtLeast returns n forced to be a positive odd integer, using def when n<=0.
func oddAtLeast(n, def int) int {
	if n <= 0 {
		n = def
	}
	if n%2 == 0 {
		n++
	}
	return n
}

// grayOf returns a single-channel Mat: src itself when already single-channel,
// otherwise its BT.601 luma. src must be 1- or 3-channel.
func grayOf(src *cv.Mat) *cv.Mat {
	switch src.Channels {
	case 1:
		return src.Clone()
	case 3:
		return cv.CvtColor(src, cv.ColorRGB2Gray)
	default:
		panic(fmt.Sprintf("photo: grayOf requires 1 or 3 channels, got %d", src.Channels))
	}
}

// gradientMagnitude returns the per-pixel 3x3 Sobel gradient magnitude of a
// single-channel image, indexed as y*cols+x.
func gradientMagnitude(gray *cv.Mat) []float64 {
	rows, cols := gray.Rows, gray.Cols
	out := make([]float64, rows*cols)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			p := func(dy, dx int) float64 { return float64(atRep(gray, y+dy, x+dx, 0)) }
			gx := (p(-1, 1) + 2*p(0, 1) + p(1, 1)) - (p(-1, -1) + 2*p(0, -1) + p(1, -1))
			gy := (p(1, -1) + 2*p(1, 0) + p(1, 1)) - (p(-1, -1) + 2*p(-1, 0) + p(-1, 1))
			out[y*cols+x] = math.Hypot(gx, gy)
		}
	}
	return out
}
