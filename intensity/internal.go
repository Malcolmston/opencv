package intensity

import (
	"fmt"

	cv "github.com/malcolmston/opencv"
)

// clampToUint8 rounds toward zero (the caller adds any rounding bias) and clamps
// the result into [0, 255], mirroring the root package's helper of the same
// name so that this package's transforms agree with cv's at the boundary.
func clampToUint8(v float64) uint8 {
	if v <= 0 {
		return 0
	}
	if v >= 255 {
		return 255
	}
	return uint8(v)
}

// requireImage panics unless m is a non-empty image.
func requireImage(m *cv.Mat, name string) {
	if m.Empty() {
		panic(fmt.Sprintf("intensity: %s given an empty image", name))
	}
}

// buildLUT constructs a 256-entry lookup table by evaluating fn at each input
// intensity i in [0,255]. The float result of fn is biased by +0.5, truncated
// and clamped into [0,255].
func buildLUT(fn func(i int) float64) []uint8 {
	lut := make([]uint8, 256)
	for i := 0; i < 256; i++ {
		lut[i] = clampToUint8(fn(i) + 0.5)
	}
	return lut
}

// applyLUT maps every sample of every channel of src through the 256-entry
// lookup table lut and returns a new Mat of the same shape. It panics if lut is
// not 256 entries long.
func applyLUT(src *cv.Mat, lut []uint8) *cv.Mat {
	if len(lut) != 256 {
		panic("intensity: applyLUT requires a 256-entry lookup table")
	}
	dst := cv.NewMat(src.Rows, src.Cols, src.Channels)
	for i, s := range src.Data {
		dst.Data[i] = lut[s]
	}
	return dst
}
