package fuzzy

import (
	"fmt"

	cv "github.com/malcolmston/opencv"
)

// FT02DIteration performs a single degree-0 F-transform inpainting iteration,
// mirroring OpenCV's ft::FT02D_iteration. Given the current image and the mask of
// pixels still to reconstruct (unknown: non-zero marks an unknown pixel), it runs
// one masked forward+inverse pass and fills every unknown pixel that at least one
// valid basis function reached.
//
// It returns the updated image (with the newly reconstructed pixels written and
// all known pixels preserved exactly), a new unknown-mask in which the pixels
// filled by this pass have been cleared to zero, and the count of pixels that
// remain unknown. Calling it repeatedly — feeding each returned mask back in —
// grows the reconstruction inward one kernel radius at a time and is exactly the
// loop [Inpaint] runs for the [Iterative] algorithm; exposing one step lets a
// caller drive, inspect, or bound the progression itself. A returned count of
// zero means the hole is completely filled.
//
// unknown must match the image dimensions. kernel must be a square, odd-sided
// [cv.FloatMat] from [CreateKernel]. The inputs are not modified.
func FT02DIteration(img *cv.Mat, kernel *cv.FloatMat, unknown *cv.Mat) (out, remaining *cv.Mat, stillUnknown int) {
	if img == nil || img.Empty() {
		panic("fuzzy: FT02DIteration given an empty image")
	}
	if unknown == nil || unknown.Rows != img.Rows || unknown.Cols != img.Cols {
		panic(fmt.Sprintf("fuzzy: FT02DIteration mask must match image size %dx%d", img.Rows, img.Cols))
	}
	rows, cols, ch := img.Rows, img.Cols, img.Channels

	// Build the validity mask the F-transform consumes (non-zero == valid).
	validity := cv.NewMat(rows, cols, 1)
	for p := 0; p < rows*cols; p++ {
		if unknown.Data[p*unknown.Channels] == 0 {
			validity.Data[p] = 255
		}
	}

	c := FT02DComponents(img, kernel, validity)
	rec, covered := inverseWithCoverage(c)

	out = img.Clone()
	remaining = cv.NewMat(rows, cols, 1)
	for p := 0; p < rows*cols; p++ {
		if unknown.Data[p*unknown.Channels] == 0 {
			continue // known pixel: leave it and its mask at zero.
		}
		if covered[p] {
			copy(out.Data[p*ch:p*ch+ch], rec.Data[p*ch:p*ch+ch])
			// filled: remaining stays 0 for this pixel.
		} else {
			remaining.Data[p] = 255
			stillUnknown++
		}
	}
	return out, remaining, stillUnknown
}
