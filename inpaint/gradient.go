package inpaint

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// SolvePoisson reconstructs an image whose discrete Laplacian equals the
// guidance field inside the selected region while matching boundary exactly
// outside it (a Dirichlet problem ∇²f = guidance on region, f = boundary on the
// complement). It is solved by Gauss-Seidel relaxation and is the core of every
// gradient-domain edit in this package. guidance and boundary must share shape
// and channel count; region must match their size. A non-positive iterations
// uses an internal cap with a convergence test. With a zero guidance field the
// result is the harmonic (smoothest) extension of the boundary — for a uniform
// boundary that is the same uniform value.
func SolvePoisson(guidance *FloatImage, boundary *cv.Mat, region *Mask, iterations int) *cv.Mat {
	inpaintRequireImage(boundary, "SolvePoisson")
	inpaintRequireMaskMatch(boundary, region, "SolvePoisson")
	if guidance.Rows != boundary.Rows || guidance.Cols != boundary.Cols || guidance.Channels != boundary.Channels {
		panic("inpaint: SolvePoisson guidance and boundary shape mismatch")
	}
	rows, cols, ch := boundary.Rows, boundary.Cols, boundary.Channels
	out := boundary.Clone()

	var idxs []int
	interior := region.Data
	for i, v := range interior {
		if v {
			idxs = append(idxs, i)
		}
	}
	if len(idxs) == 0 {
		return out
	}

	maxIter := iterations
	if maxIter <= 0 {
		maxIter = 5000
	}
	const tol = 0.05

	f := make([]float64, rows*cols)
	for c := 0; c < ch; c++ {
		for _, i := range idxs {
			y, x := i/cols, i%cols
			f[i] = float64(out.At(y, x, c))
		}
		for iter := 0; iter < maxIter; iter++ {
			var maxDelta float64
			for _, i := range idxs {
				y, x := i/cols, i%cols
				var nsum float64
				for _, d := range neighbors4 {
					ny, nx := y+d[0], x+d[1]
					if ny >= 0 && ny < rows && nx >= 0 && nx < cols && interior[ny*cols+nx] {
						nsum += f[ny*cols+nx]
					} else {
						nsum += float64(inpaintAtRep(out, ny, nx, c))
					}
				}
				v := (nsum - guidance.At(y, x, c)) / 4
				if d := math.Abs(v - f[i]); d > maxDelta {
					maxDelta = d
				}
				f[i] = v
			}
			if maxDelta < tol {
				break
			}
		}
		for _, i := range idxs {
			y, x := i/cols, i%cols
			out.Set(y, x, c, inpaintClampU8(f[i]))
		}
	}
	return out
}

// ColorChange multiplies the colour of the region selected by mask in src by the
// per-channel factors (rMul, gMul, bMul) in the gradient domain: the source
// gradients are scaled and reintegrated against the unchanged surroundings, so
// the recoloured region joins the rest seamlessly. src must be three-channel;
// mask must match its size. src is not modified — a new image is returned.
func ColorChange(src *cv.Mat, mask *Mask, rMul, gMul, bMul float64) *cv.Mat {
	inpaintRequireImage(src, "ColorChange")
	inpaintRequireChannels(src, 3, "ColorChange")
	inpaintRequireMaskMatch(src, mask, "ColorChange")
	muls := [3]float64{rMul, gMul, bMul}
	gx := GradientX(src)
	gy := GradientY(src)
	for i := 0; i < len(gx.Data); i++ {
		c := i % 3
		gx.Data[i] *= muls[c]
		gy.Data[i] *= muls[c]
	}
	guidance := Divergence(gx, gy)
	return SolvePoisson(guidance, src, mask, 0)
}

// IlluminationChange locally relights the region selected by mask in src by
// compressing its gradient field with the Fattal-style operator
// g' = alpha^beta * |g|^(-beta) * g and reintegrating it, which suppresses harsh
// lighting while keeping texture. Typical alpha is around 0.2 and beta around
// 0.4; larger beta flattens illumination more. src must be three-channel; mask
// must match its size. src is not modified — a new image is returned.
func IlluminationChange(src *cv.Mat, mask *Mask, alpha, beta float64) *cv.Mat {
	inpaintRequireImage(src, "IlluminationChange")
	inpaintRequireChannels(src, 3, "IlluminationChange")
	inpaintRequireMaskMatch(src, mask, "IlluminationChange")
	gx := GradientX(src)
	gy := GradientY(src)
	for i := 0; i < len(gx.Data); i++ {
		mag := math.Hypot(gx.Data[i], gy.Data[i])
		if mag < 1e-4 {
			continue
		}
		scale := math.Pow(alpha, beta) * math.Pow(mag, -beta)
		gx.Data[i] *= scale
		gy.Data[i] *= scale
	}
	guidance := Divergence(gx, gy)
	return SolvePoisson(guidance, src, mask, 0)
}

// TextureFlattening washes out fine texture inside the region selected by mask
// in src while preserving strong edges: gradients whose magnitude falls outside
// the Canny-style band [lowThreshold, highThreshold] (measured on the luma
// gradient magnitude) are zeroed before reintegration, so only salient contours
// survive. src must be three-channel; mask must match its size. src is not
// modified — a new image is returned.
func TextureFlattening(src *cv.Mat, mask *Mask, lowThreshold, highThreshold float64) *cv.Mat {
	inpaintRequireImage(src, "TextureFlattening")
	inpaintRequireChannels(src, 3, "TextureFlattening")
	inpaintRequireMaskMatch(src, mask, "TextureFlattening")
	gx := GradientX(src)
	gy := GradientY(src)
	rows, cols := src.Rows, src.Cols
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			// Luma gradient magnitude gates all three channels together.
			lgx := inpaintLuma(src, y, inpaintClampInt(x+1, 0, cols-1)) - inpaintLuma(src, y, x)
			lgy := inpaintLuma(src, inpaintClampInt(y+1, 0, rows-1), x) - inpaintLuma(src, y, x)
			mag := math.Hypot(lgx, lgy)
			if mag < lowThreshold || mag > highThreshold {
				i := (y*cols + x) * 3
				for c := 0; c < 3; c++ {
					gx.Data[i+c] = 0
					gy.Data[i+c] = 0
				}
			}
		}
	}
	guidance := Divergence(gx, gy)
	return SolvePoisson(guidance, src, mask, 0)
}
