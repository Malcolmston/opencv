package inpaint

import (
	"image"
	"math"

	cv "github.com/malcolmston/opencv"
)

// CloneMode selects how source guidance gradients are chosen in [SeamlessClone].
type CloneMode int

const (
	// NormalClone imports the source's gradients directly, so the cloned region
	// keeps the source's texture while its overall colour is retargeted to the
	// destination at the seam.
	NormalClone CloneMode = iota
	// MixedClone uses, per edge, whichever of the source and destination
	// gradients is larger in magnitude, preserving salient destination structure
	// under the pasted region (useful when the source has flat areas).
	MixedClone
	// MonochromeTransfer imports only the source's luma gradients (identical
	// across channels), transferring texture without the source's colour.
	MonochromeTransfer
)

// PoissonBlend reconstructs an image equal to dst outside region and, inside
// region, the image whose gradient best matches the guidance vector field
// (guidanceX, guidanceY) while remaining continuous with dst at the region
// boundary. It is the gradient-domain workhorse behind [SeamlessClone]:
// SolvePoisson is applied to the divergence of the guidance field with dst as
// the Dirichlet boundary. guidanceX, guidanceY and dst must share shape and
// channel count; region must match their size. A non-positive iterations uses
// the solver's default. dst is not modified — a new image is returned.
func PoissonBlend(dst *cv.Mat, guidanceX, guidanceY *FloatImage, region *Mask, iterations int) *cv.Mat {
	inpaintRequireImage(dst, "PoissonBlend")
	if guidanceX.Rows != dst.Rows || guidanceX.Cols != dst.Cols || guidanceX.Channels != dst.Channels ||
		guidanceY.Rows != dst.Rows || guidanceY.Cols != dst.Cols || guidanceY.Channels != dst.Channels {
		panic("inpaint: PoissonBlend guidance and dst shape mismatch")
	}
	guidance := Divergence(guidanceX, guidanceY)
	return SolvePoisson(guidance, dst, region, iterations)
}

// SeamlessClone imports the masked part of src into dst using Poisson image
// editing, returning a new image (dst is not modified). The masked region of src
// — where mask is selected — is positioned so the mask's bounding-box centre
// lands at center in dst. Inside the pasted region the result's gradients follow
// a guidance field chosen by mode while the region border is pinned to dst, so
// the seam is invisible. src and dst must be three-channel; mask must match
// src's dimensions. Intended for modest image sizes.
func SeamlessClone(src, dst *cv.Mat, mask *Mask, center image.Point, mode CloneMode) *cv.Mat {
	inpaintRequireImage(src, "SeamlessClone src")
	inpaintRequireImage(dst, "SeamlessClone dst")
	inpaintRequireChannels(src, 3, "SeamlessClone src")
	inpaintRequireChannels(dst, 3, "SeamlessClone dst")
	inpaintRequireMaskMatch(src, mask, "SeamlessClone")

	bbox, ok := mask.BoundingBox()
	if !ok {
		return dst.Clone()
	}
	bcx := (bbox.Min.X + bbox.Max.X - 1) / 2
	bcy := (bbox.Min.Y + bbox.Max.Y - 1) / 2
	offX := center.X - bcx
	offY := center.Y - bcy

	drows, dcols := dst.Rows, dst.Cols

	// region: destination pixels covered by the mapped, in-bounds mask.
	region := NewMask(drows, dcols)
	for sy := bbox.Min.Y; sy < bbox.Max.Y; sy++ {
		for sx := bbox.Min.X; sx < bbox.Max.X; sx++ {
			if !mask.At(sy, sx) {
				continue
			}
			dy, dx := sy+offY, sx+offX
			if dy < 0 || dy >= drows || dx < 0 || dx >= dcols {
				continue
			}
			region.Set(dy, dx, true)
		}
	}
	if region.Empty() {
		return dst.Clone()
	}

	// Build the guidance vector field over dst by mapping each dst pixel back to
	// source coordinates.
	gx := NewFloatImage(drows, dcols, 3)
	gy := NewFloatImage(drows, dcols, 3)
	for dy := 0; dy < drows; dy++ {
		for dx := 0; dx < dcols; dx++ {
			sy, sx := dy-offY, dx-offX
			for c := 0; c < 3; c++ {
				var sgx, sgy float64
				switch mode {
				case MonochromeTransfer:
					sgx = inpaintLumaRep(src, sy, sx+1) - inpaintLumaRep(src, sy, sx)
					sgy = inpaintLumaRep(src, sy+1, sx) - inpaintLumaRep(src, sy, sx)
				default:
					sgx = float64(inpaintAtRep(src, sy, sx+1, c)) - float64(inpaintAtRep(src, sy, sx, c))
					sgy = float64(inpaintAtRep(src, sy+1, sx, c)) - float64(inpaintAtRep(src, sy, sx, c))
				}
				if mode == MixedClone {
					dgx := float64(inpaintAtRep(dst, dy, dx+1, c)) - float64(inpaintAtRep(dst, dy, dx, c))
					dgy := float64(inpaintAtRep(dst, dy+1, dx, c)) - float64(inpaintAtRep(dst, dy, dx, c))
					if math.Abs(dgx) > math.Abs(sgx) {
						sgx = dgx
					}
					if math.Abs(dgy) > math.Abs(sgy) {
						sgy = dgy
					}
				}
				gx.Set(dy, dx, c, sgx)
				gy.Set(dy, dx, c, sgy)
			}
		}
	}
	return PoissonBlend(dst, gx, gy, region, 0)
}

// inpaintLumaRep returns the BT.601 luma of src at (y, x) with edge replication.
func inpaintLumaRep(src *cv.Mat, y, x int) float64 {
	if src.Channels == 1 {
		return float64(inpaintAtRep(src, y, x, 0))
	}
	r := float64(inpaintAtRep(src, y, x, 0))
	g := float64(inpaintAtRep(src, y, x, 1))
	b := float64(inpaintAtRep(src, y, x, 2))
	return 0.299*r + 0.587*g + 0.114*b
}
