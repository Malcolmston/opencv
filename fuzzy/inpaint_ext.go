package fuzzy

import (
	"fmt"

	cv "github.com/malcolmston/opencv"
)

// Reconstruct is a convenience method equivalent to [FT02DInverse](c); it paints
// the degree-0 components back into a full-size image. It mirrors the
// [Components1.Reconstruct] method of the degree-1 components.
func (c *Components) Reconstruct() *cv.Mat {
	return FT02DInverse(c)
}

// InpaintMultiStep reconstructs the unknown pixels of img with a multi-resolution
// F-transform scheme, mirroring the spirit of OpenCV's ft::MULTI_STEP inpainting.
// Where [Inpaint] with [OneStep] fills only pixels within one kernel radius of
// known data — leaving the centre of a large hole untouched — and [Iterative]
// grows inward one radius at a time at a single scale, MultiStep works
// coarse-to-fine: it first bridges the hole with large kernels, then refines the
// filled region with progressively smaller kernels and a final gradient-aware
// (degree-1) pass. The result fills arbitrarily large holes completely and, on
// structured content such as gradients, with markedly lower error than one-step.
//
// mask marks the pixels to reconstruct (non-zero == unknown), matching [Inpaint]'s
// polarity, and must match the image size. radius is the finest kernel radius and
// must be >= 1; function chooses the basis shape. The original img is not
// modified; known pixels are preserved exactly in the returned image.
func InpaintMultiStep(img, mask *cv.Mat, radius int, function BasisFunction) *cv.Mat {
	if img == nil || img.Empty() {
		panic("fuzzy: InpaintMultiStep given an empty image")
	}
	if mask == nil || mask.Rows != img.Rows || mask.Cols != img.Cols {
		panic(fmt.Sprintf("fuzzy: InpaintMultiStep mask must match image size %dx%d", img.Rows, img.Cols))
	}
	if radius < 1 {
		panic(fmt.Sprintf("fuzzy: InpaintMultiStep radius must be >= 1, got %d", radius))
	}

	rows, cols, ch := img.Rows, img.Cols, img.Channels
	work := img.Clone()

	// unknown[p] marks pixels still to be reconstructed. origUnknown remembers the
	// original hole so the refinement pass knows which pixels it may overwrite.
	unknown := make([]bool, rows*cols)
	origUnknown := make([]bool, rows*cols)
	for p := 0; p < rows*cols; p++ {
		u := mask.Data[p*mask.Channels] != 0
		unknown[p] = u
		origUnknown[p] = u
	}

	// Coarse-to-fine radius schedule: larger kernels bridge wide holes in a few
	// iterations, finer kernels then tighten the fill. Duplicates are skipped.
	schedule := dedupRadii([]int{radius * 4, radius * 2, radius})
	for _, rr := range schedule {
		kernel := CreateKernel(function, rr)
		iterativeFill(work, unknown, kernel, function)
		if !anyUnknown(unknown) {
			break
		}
	}
	// A final safety pass at the finest scale in case a residual pixel remains.
	if anyUnknown(unknown) {
		iterativeFill(work, unknown, CreateKernel(function, radius), function)
	}

	// Gradient-aware refinement: fit degree-1 planes to the ORIGINAL known pixels
	// and overwrite each covered hole pixel with that reconstruction. This replaces
	// the flat coarse fill with a locally linear estimate, cutting error on
	// gradients while leaving any pixel the fit cannot reach at its coarse value.
	validity := cv.NewMat(rows, cols, 1)
	for p := 0; p < rows*cols; p++ {
		if !origUnknown[p] {
			validity.Data[p] = 255
		}
	}
	comps := FT12DComponents(work, CreateKernel(function, radius), validity)
	comps.Function = function
	rec, covered := comps.inverseWithCoverage()
	for p := 0; p < rows*cols; p++ {
		if origUnknown[p] && covered[p] {
			copy(work.Data[p*ch:p*ch+ch], rec.Data[p*ch:p*ch+ch])
		}
	}
	return work
}

// iterativeFill repeatedly runs the masked degree-0 F-transform, promoting each
// newly covered unknown pixel to known, until a pass makes no further progress or
// the hole is filled. It mutates work (writing reconstructed samples) and unknown
// (clearing filled pixels).
func iterativeFill(work *cv.Mat, unknown []bool, kernel *cv.FloatMat, function BasisFunction) {
	rows, cols, ch := work.Rows, work.Cols, work.Channels
	validity := cv.NewMat(rows, cols, 1)
	for iter := 0; iter < maxInpaintIterations; iter++ {
		for p := 0; p < rows*cols; p++ {
			if unknown[p] {
				validity.Data[p] = 0
			} else {
				validity.Data[p] = 255
			}
		}
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
		if !changed || !anyUnknown(unknown) {
			break
		}
	}
}

// dedupRadii returns the input radii with duplicates and non-positive values
// removed, preserving order.
func dedupRadii(radii []int) []int {
	out := make([]int, 0, len(radii))
	seen := make(map[int]bool, len(radii))
	for _, r := range radii {
		if r < 1 || seen[r] {
			continue
		}
		seen[r] = true
		out = append(out, r)
	}
	return out
}
