package inpaint

import (
	"fmt"
	"math"

	cv "github.com/malcolmston/opencv"
)

// neighbors4 lists the 4-connected neighbour offsets (dy, dx).
var neighbors4 = [4][2]int{{-1, 0}, {1, 0}, {0, -1}, {0, 1}}

// neighbors8 lists the 8-connected neighbour offsets (dy, dx).
var neighbors8 = [8][2]int{
	{-1, -1}, {-1, 0}, {-1, 1},
	{0, -1}, {0, 1},
	{1, -1}, {1, 0}, {1, 1},
}

// inpaintClampU8 rounds v to the nearest integer and clamps it into [0,255].
func inpaintClampU8(v float64) uint8 {
	v = math.Round(v)
	if v <= 0 {
		return 0
	}
	if v >= 255 {
		return 255
	}
	return uint8(v)
}

// inpaintAtRep returns the sample at (y, x, c) clamping out-of-range
// coordinates to the nearest edge (OpenCV's BORDER_REPLICATE).
func inpaintAtRep(m *cv.Mat, y, x, c int) uint8 {
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

// inpaintLuma returns the BT.601 luma of pixel (y, x). For single-channel
// images the sample itself is returned.
func inpaintLuma(m *cv.Mat, y, x int) float64 {
	if m.Channels == 1 {
		return float64(m.At(y, x, 0))
	}
	r := float64(m.At(y, x, 0))
	g := float64(m.At(y, x, 1))
	b := float64(m.At(y, x, 2))
	return 0.299*r + 0.587*g + 0.114*b
}

// inpaintRequireImage panics if img is nil or empty.
func inpaintRequireImage(img *cv.Mat, name string) {
	if img == nil || img.Empty() {
		panic(fmt.Sprintf("inpaint: %s given an empty image", name))
	}
}

// inpaintRequireChannels panics if src does not have exactly want channels.
func inpaintRequireChannels(src *cv.Mat, want int, name string) {
	if src.Channels != want {
		panic(fmt.Sprintf("inpaint: %s requires %d channels, got %d", name, want, src.Channels))
	}
}

// inpaintRequireMaskMatch panics if the mask does not match the image size.
func inpaintRequireMaskMatch(img *cv.Mat, mask *Mask, name string) {
	if mask == nil {
		panic(fmt.Sprintf("inpaint: %s given a nil mask", name))
	}
	if img.Rows != mask.Rows || img.Cols != mask.Cols {
		panic(fmt.Sprintf("inpaint: %s mask is %dx%d, want %dx%d",
			name, mask.Rows, mask.Cols, img.Rows, img.Cols))
	}
}

// inpaintClampInt clamps v into [lo, hi].
func inpaintClampInt(v, lo, hi int) int {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}
