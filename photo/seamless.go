package photo

import (
	"image"
	"math"

	cv "github.com/malcolmston/opencv"
)

// SeamlessCloneMode selects how the source guidance gradients are chosen in
// [SeamlessClone].
type SeamlessCloneMode int

const (
	// NormalClone imports the source's gradients directly, so the cloned region
	// takes on the source's texture while its overall colour is retargeted to the
	// destination at the seam.
	NormalClone SeamlessCloneMode = iota
	// MixedClone uses, per neighbour, whichever of the source and destination
	// gradients is larger in magnitude. This preserves salient destination
	// structure under the pasted region (useful when the source has flat areas).
	MixedClone
)

// SeamlessClone blends the masked part of src into dst using Poisson image
// editing, returning a new image (dst is not modified). The masked region of
// src — where mask's first channel is non-zero — is positioned so that the
// mask's bounding-box centre lands at center in dst. Inside the region the
// result's Laplacian is driven to match a guidance field (see
// [SeamlessCloneMode]) while its border is pinned to the destination, which is
// solved with Gauss–Seidel iteration. Because the border is pinned to dst, the
// seam is continuous with the destination.
//
// src, dst and the result are three-channel; mask must match src's dimensions.
// Intended for small images.
func SeamlessClone(src, dst, mask *cv.Mat, center image.Point, mode SeamlessCloneMode) *cv.Mat {
	if src == nil || dst == nil || mask == nil {
		panic("photo: SeamlessClone given a nil argument")
	}
	requireChannels(src, 3, "SeamlessClone src")
	requireChannels(dst, 3, "SeamlessClone dst")
	requireSameSize(src, mask, "SeamlessClone mask")

	// Bounding box of the mask in source coordinates.
	minX, minY := mask.Cols, mask.Rows
	maxX, maxY := -1, -1
	for y := 0; y < mask.Rows; y++ {
		for x := 0; x < mask.Cols; x++ {
			if mask.At(y, x, 0) != 0 {
				if x < minX {
					minX = x
				}
				if x > maxX {
					maxX = x
				}
				if y < minY {
					minY = y
				}
				if y > maxY {
					maxY = y
				}
			}
		}
	}
	out := dst.Clone()
	if maxX < 0 {
		return out // empty mask: nothing to clone
	}

	// Offset mapping source coords -> destination coords so the mask bbox centre
	// aligns with center.
	bcx := (minX + maxX) / 2
	bcy := (minY + maxY) / 2
	offX := center.X - bcx
	offY := center.Y - bcy

	// Collect interior (unknown) pixels: masked source pixels whose destination
	// image is in bounds. interior[dstIdx] marks them.
	interior := make([]bool, dst.Rows*dst.Cols)
	var idxs []int
	for sy := minY; sy <= maxY; sy++ {
		for sx := minX; sx <= maxX; sx++ {
			if mask.At(sy, sx, 0) == 0 {
				continue
			}
			dy, dx := sy+offY, sx+offX
			if dy < 0 || dy >= dst.Rows || dx < 0 || dx >= dst.Cols {
				continue
			}
			interior[dy*dst.Cols+dx] = true
			idxs = append(idxs, dy*dst.Cols+dx)
		}
	}
	if len(idxs) == 0 {
		return out
	}

	// Solve each channel independently.
	f := make([]float64, dst.Rows*dst.Cols) // working solution for one channel
	for c := 0; c < 3; c++ {
		// Initialise interior with the source values (good starting guess).
		for _, i := range idxs {
			dy, dx := i/dst.Cols, i%dst.Cols
			sy, sx := dy-offY, dx-offX
			f[i] = float64(atRep(src, sy, sx, c))
		}
		const maxIter = 5000
		const tol = 0.3
		for iter := 0; iter < maxIter; iter++ {
			var maxDelta float64
			for _, i := range idxs {
				dy, dx := i/dst.Cols, i%dst.Cols
				sy, sx := dy-offY, dx-offX
				var neighborSum, guidance float64
				for _, d := range neighbors4 {
					ndy, ndx := dy+d[0], dx+d[1]
					nsy, nsx := sy+d[0], sx+d[1]
					// Neighbour value: interior uses the current solution, else the
					// pinned destination (Dirichlet boundary).
					if ndy >= 0 && ndy < dst.Rows && ndx >= 0 && ndx < dst.Cols && interior[ndy*dst.Cols+ndx] {
						neighborSum += f[ndy*dst.Cols+ndx]
					} else {
						neighborSum += float64(atRep(out, ndy, ndx, c))
					}
					// Guidance gradient for this neighbour.
					gsrc := float64(atRep(src, sy, sx, c)) - float64(atRep(src, nsy, nsx, c))
					if mode == MixedClone {
						gdst := float64(atRep(out, dy, dx, c)) - float64(atRep(out, ndy, ndx, c))
						if math.Abs(gdst) > math.Abs(gsrc) {
							gsrc = gdst
						}
					}
					guidance += gsrc
				}
				v := (neighborSum + guidance) / 4
				if delta := math.Abs(v - f[i]); delta > maxDelta {
					maxDelta = delta
				}
				f[i] = v
			}
			if maxDelta < tol {
				break
			}
		}
		for _, i := range idxs {
			dy, dx := i/dst.Cols, i%dst.Cols
			out.Set(dy, dx, c, clampU8(f[i]))
		}
	}
	return out
}
