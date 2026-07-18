package inpaint

import (
	"fmt"

	cv "github.com/malcolmston/opencv"
)

// Method selects the region-filling algorithm used by [Inpaint].
type Method int

const (
	// MethodTelea selects Telea's Fast Marching Method inpainting
	// ([InpaintTelea]): fast, good for thin defects and smooth regions.
	MethodTelea Method = iota
	// MethodNavierStokes selects the fluid-dynamics transport fill
	// ([InpaintNavierStokes]): continues edges into the hole.
	MethodNavierStokes
	// MethodDiffusion selects the harmonic Laplace fill ([InpaintDiffusion]):
	// the smoothest interpolation of the boundary.
	MethodDiffusion
	// MethodCriminisi selects exemplar-based completion ([InpaintCriminisi]):
	// best for textured or structured holes, but the slowest.
	MethodCriminisi
	// MethodPatchMatch selects iterated PatchMatch synthesis
	// ([InpaintPatchMatch]): texture-aware and randomised (deterministically
	// seeded).
	MethodPatchMatch
)

// String returns the method name.
func (m Method) String() string {
	switch m {
	case MethodTelea:
		return "Telea"
	case MethodNavierStokes:
		return "NavierStokes"
	case MethodDiffusion:
		return "Diffusion"
	case MethodCriminisi:
		return "Criminisi"
	case MethodPatchMatch:
		return "PatchMatch"
	default:
		return fmt.Sprintf("Method(%d)", int(m))
	}
}

// Inpaint reconstructs the pixels of img selected by mask using the chosen
// method, returning a filled clone (img is not modified). radius is the
// neighbourhood radius for [MethodTelea] (minimum 1) and is ignored by the other
// methods, which use their default options. mask must match img's size
// (true = fill). img may be single- or three-channel. For finer control over a
// method call its dedicated function directly. It panics on an unknown method.
func Inpaint(img *cv.Mat, mask *Mask, radius int, method Method) *cv.Mat {
	switch method {
	case MethodTelea:
		return InpaintTelea(img, mask, radius)
	case MethodNavierStokes:
		return InpaintNavierStokes(img, mask, 0)
	case MethodDiffusion:
		return InpaintDiffusion(img, mask)
	case MethodCriminisi:
		return InpaintCriminisi(img, mask, DefaultCriminisiOptions())
	case MethodPatchMatch:
		return InpaintPatchMatch(img, mask, DefaultPatchMatchOptions())
	default:
		panic(fmt.Sprintf("inpaint: Inpaint unknown method %d", int(method)))
	}
}

// InpaintMat is a convenience wrapper accepting the mask as a [cv.Mat] rather
// than a [Mask]: any pixel whose first channel is non-zero is filled. It builds
// the mask via [MaskFromMat] and defers to [Inpaint].
func InpaintMat(img, mask *cv.Mat, radius int, method Method) *cv.Mat {
	inpaintRequireImage(img, "InpaintMat")
	return Inpaint(img, MaskFromMat(mask, 0), radius, method)
}
