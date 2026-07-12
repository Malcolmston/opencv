package fuzzy

import (
	"fmt"

	cv "github.com/malcolmston/opencv"
)

// InpaintAlgorithm selects the F-transform inpainting strategy used by [Inpaint].
type InpaintAlgorithm int

const (
	// OneStep reconstructs every unknown pixel in a single F-transform pass,
	// mirroring OpenCV's ft::ONE_STEP. It fills each hole from the valid pixels
	// under the overlapping basis functions that reach it, so it works best when
	// holes are no wider than about the kernel radius. Larger holes may contain
	// pixels no valid basis reaches; those are left unchanged in this mode.
	OneStep InpaintAlgorithm = iota
	// Iterative repeats the one-step reconstruction, promoting each newly filled
	// pixel to "known" before the next pass so coverage grows inward until the
	// whole hole is filled (or no further progress is possible). It mirrors the
	// spirit of OpenCV's ft::ITERATIVE and copes with holes larger than the
	// kernel at the cost of extra passes.
	Iterative
)

// String returns the name of the algorithm.
func (a InpaintAlgorithm) String() string {
	switch a {
	case OneStep:
		return "OneStep"
	case Iterative:
		return "Iterative"
	default:
		return fmt.Sprintf("InpaintAlgorithm(%d)", int(a))
	}
}

// maxInpaintIterations bounds the Iterative algorithm so it always terminates
// even if some pixels can never be reached (e.g. an entirely masked image).
const maxInpaintIterations = 1000

// Inpaint reconstructs the unknown pixels of img using the degree-0 F-transform,
// mirroring OpenCV's ft::inpaint. mask marks the pixels to reconstruct: a pixel
// is treated as unknown wherever mask's first channel is non-zero, and as a
// valid observation everywhere else. (This is the inverse of OpenCV's raw
// ft::inpaint mask polarity, and matches the "mask of unknown pixels" convention
// used elsewhere in this module; the underlying F-transform still consumes a
// validity mask internally.)
//
// mask must have the same width and height as img. img may be grayscale or
// multi-channel. radius sets the kernel radius (and partition spacing) in pixels
// and must be >= 1; function chooses the basis shape. The original img is not
// modified — a filled clone is returned in which known pixels are preserved
// exactly and unknown pixels are replaced by the reconstruction.
//
// With [OneStep] holes wider than the kernel may retain some unreconstructed
// pixels; use [Iterative] to fill them.
func Inpaint(img, mask *cv.Mat, radius int, function BasisFunction, algorithm InpaintAlgorithm) *cv.Mat {
	if img == nil || img.Empty() {
		panic("fuzzy: Inpaint given an empty image")
	}
	if mask == nil || mask.Rows != img.Rows || mask.Cols != img.Cols {
		panic(fmt.Sprintf("fuzzy: Inpaint mask must match image size %dx%d", img.Rows, img.Cols))
	}
	if radius < 1 {
		panic(fmt.Sprintf("fuzzy: Inpaint radius must be >= 1, got %d", radius))
	}

	rows, cols := img.Rows, img.Cols
	kernel := CreateKernel(function, radius)

	// unknown[p] marks pixels still to be reconstructed.
	unknown := make([]bool, rows*cols)
	for p := 0; p < rows*cols; p++ {
		unknown[p] = mask.Data[p*mask.Channels] != 0
	}

	// validity is the mask the F-transform consumes: non-zero == valid pixel.
	validity := cv.NewMat(rows, cols, 1)
	syncValidity := func() {
		for p := 0; p < rows*cols; p++ {
			if unknown[p] {
				validity.Data[p] = 0
			} else {
				validity.Data[p] = 255
			}
		}
	}

	work := img.Clone()
	ch := img.Channels

	fillOnce := func() bool {
		syncValidity()
		c := FT02DComponents(work, kernel, validity)
		c.Function = function
		rec, covered := inverseWithCoverage(c)
		changed := false
		for p := 0; p < rows*cols; p++ {
			if !unknown[p] || !covered[p] {
				continue
			}
			copy(work.Data[p*ch:p*ch+ch], rec.Data[p*ch:p*ch+ch])
			unknown[p] = false
			changed = true
		}
		return changed
	}

	switch algorithm {
	case Iterative:
		for iter := 0; iter < maxInpaintIterations; iter++ {
			if !fillOnce() {
				break
			}
			if !anyUnknown(unknown) {
				break
			}
		}
	default: // OneStep
		fillOnce()
	}
	return work
}

// anyUnknown reports whether any pixel remains unknown.
func anyUnknown(unknown []bool) bool {
	for _, u := range unknown {
		if u {
			return true
		}
	}
	return false
}
