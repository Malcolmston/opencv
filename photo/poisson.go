package photo

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// ColorChange multiplies the colour of the masked region of src by per-channel
// factors while keeping its texture, using Poisson image editing. Rather than
// scaling pixel values directly (which would create a visible seam), it scales
// the source's colour gradients inside the mask and reintegrates them subject to
// the unmasked border, so the recoloured region blends seamlessly. Setting a
// factor above one intensifies a channel and below one attenuates it.
//
// src must be three-channel; mask has the same width and height as src and marks
// the region to change where its first channel is non-zero. The mask should not
// touch the image border (the surrounding pixels form the Dirichlet boundary).
// The original src is not modified.
func ColorChange(src, mask *cv.Mat, redMul, greenMul, blueMul float64) *cv.Mat {
	requireChannels(src, 3, "ColorChange")
	requireSameSize(src, mask, "ColorChange")
	mul := [3]float64{redMul, greenMul, blueMul}
	srcF := floatChannels(src)
	interior := interiorMask(mask)
	guidance := func(c, py, px, ny, nx int) float64 {
		return mul[c] * (srcF[c][py*src.Cols+px] - srcF[c][ny*src.Cols+nx])
	}
	return poissonEdit(src, interior, guidance)
}

// IlluminationChange locally relights the masked region of src by compressing
// its gradient field, following the Poisson local-illumination-change edit of
// Pérez et al. The gradient magnitudes inside the mask are remapped by the power
// law alpha^beta * |grad|^(-beta), which flattens strong gradients and lifts
// weak ones, then reintegrated against the unmasked border. It is useful for
// removing highlights or evening out shadows within a region.
//
// src must be three-channel; mask marks the region where its first channel is
// non-zero and should not touch the image border. alpha (default 0.2) sets the
// overall strength and beta (default 0.4) the compression exponent; both default
// when non-positive. The original src is not modified.
func IlluminationChange(src, mask *cv.Mat, alpha, beta float64) *cv.Mat {
	requireChannels(src, 3, "IlluminationChange")
	requireSameSize(src, mask, "IlluminationChange")
	if alpha <= 0 {
		alpha = 0.2
	}
	if beta <= 0 {
		beta = 0.4
	}
	srcF := floatChannels(src)
	interior := interiorMask(mask)
	scale := math.Pow(alpha, beta)
	guidance := func(c, py, px, ny, nx int) float64 {
		g := srcF[c][py*src.Cols+px] - srcF[c][ny*src.Cols+nx]
		mag := math.Abs(g)
		if mag < 1e-4 {
			return 0
		}
		return scale * math.Pow(mag, -beta) * g
	}
	return poissonEdit(src, interior, guidance)
}

// TextureFlattening washes out fine texture inside the masked region of src
// while preserving its major edges, by importing only the source gradients that
// coincide with an edge and setting all other guidance gradients to zero. When
// the flattened gradient field is reintegrated with Poisson editing, smooth
// regions become nearly flat while strong contours survive. Edges are detected
// from the gradient magnitude of the source luma.
//
// src must be three-channel; mask marks the region where its first channel is
// non-zero and should not touch the image border. A gradient is kept as an edge
// where its magnitude lies at or above lowThreshold; highThreshold and
// kernelSize are accepted for OpenCV API parity (highThreshold additionally
// forces very strong gradients to be kept). The original src is not modified.
func TextureFlattening(src, mask *cv.Mat, lowThreshold, highThreshold float64, kernelSize int) *cv.Mat {
	requireChannels(src, 3, "TextureFlattening")
	requireSameSize(src, mask, "TextureFlattening")
	_ = kernelSize
	if lowThreshold <= 0 {
		lowThreshold = 30
	}
	if highThreshold <= lowThreshold {
		highThreshold = 3 * lowThreshold
	}
	srcF := floatChannels(src)
	interior := interiorMask(mask)
	mag := gradientMagnitude(grayOf(src))
	rows, cols := src.Rows, src.Cols

	// Hysteresis edge map: a pixel is an edge if its gradient is strong
	// (>= highThreshold), or weak (>= lowThreshold) but adjacent to a strong one.
	edge := make([]bool, rows*cols)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			i := y*cols + x
			if mag[i] >= highThreshold {
				edge[i] = true
				continue
			}
			if mag[i] >= lowThreshold {
				for _, d := range neighbors4 {
					ny, nx := y+d[0], x+d[1]
					if ny >= 0 && ny < rows && nx >= 0 && nx < cols && mag[ny*cols+nx] >= highThreshold {
						edge[i] = true
						break
					}
				}
			}
		}
	}
	guidance := func(c, py, px, ny, nx int) float64 {
		if edge[py*cols+px] || edge[ny*cols+nx] {
			return srcF[c][py*cols+px] - srcF[c][ny*cols+nx]
		}
		return 0
	}
	return poissonEdit(src, interior, guidance)
}

// floatChannels returns the per-channel float planes of m, indexed [c][y*cols+x].
func floatChannels(m *cv.Mat) [][]float64 {
	ch := m.Channels
	out := make([][]float64, ch)
	for c := 0; c < ch; c++ {
		p := make([]float64, m.Rows*m.Cols)
		for i := range p {
			p[i] = float64(m.Data[i*ch+c])
		}
		out[c] = p
	}
	return out
}

// interiorMask marks the pixels whose first mask channel is non-zero.
func interiorMask(mask *cv.Mat) []bool {
	interior := make([]bool, mask.Rows*mask.Cols)
	for i := range interior {
		interior[i] = mask.Data[i*mask.Channels] != 0
	}
	return interior
}

// poissonEdit solves, for every interior pixel and channel, the Poisson
// equation whose Laplacian matches a guidance divergence, with the unmodified
// img pinned outside the interior as the Dirichlet boundary. guidance(c, py, px,
// ny, nx) returns the guidance gradient contribution from interior pixel
// (px,py) toward neighbour (nx,ny) on channel c. Solved with Gauss–Seidel.
func poissonEdit(img *cv.Mat, interior []bool, guidance func(c, py, px, ny, nx int) float64) *cv.Mat {
	rows, cols, ch := img.Rows, img.Cols, img.Channels
	out := img.Clone()

	var idxs []int
	for i, in := range interior {
		if in {
			idxs = append(idxs, i)
		}
	}
	if len(idxs) == 0 {
		return out
	}

	f := make([]float64, rows*cols)
	for c := 0; c < ch; c++ {
		for _, i := range idxs {
			f[i] = float64(img.Data[i*ch+c])
		}
		const maxIter = 5000
		const tol = 0.3
		for iter := 0; iter < maxIter; iter++ {
			var maxDelta float64
			for _, i := range idxs {
				py, px := i/cols, i%cols
				var neighborSum, guide float64
				for _, d := range neighbors4 {
					ny, nx := py+d[0], px+d[1]
					if ny < 0 || ny >= rows || nx < 0 || nx >= cols {
						// Out-of-image neighbour: replicate the interior value so the
						// border stays free (Neumann-like) instead of pulling to 0.
						neighborSum += f[i]
						continue
					}
					j := ny*cols + nx
					if interior[j] {
						neighborSum += f[j]
					} else {
						neighborSum += float64(out.Data[j*ch+c])
					}
					guide += guidance(c, py, px, ny, nx)
				}
				v := (neighborSum + guide) / 4
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
			out.Data[i*ch+c] = clampU8(f[i])
		}
	}
	return out
}
