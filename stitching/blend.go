package stitching

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// Layer is one image warped into the panorama canvas, ready for blending. Color
// holds the resampled pixels over the full canvas (zero outside the image), and
// Weight is a per-pixel blending weight over the same canvas: zero means the
// pixel is not covered by this layer, and larger positive values indicate pixels
// that should dominate the blend (the feather ramp is highest at the image
// centre and lowest at its border).
type Layer struct {
	Color  *cv.Mat
	Weight *cv.FloatMat
}

// Blender combines the overlapping [Layer]s of a panorama into a single image of
// the given size. Implementations include [Feather] and [MultiBandBlend].
type Blender interface {
	// Blend merges layers into a rows×cols image with the given channel count.
	Blend(layers []Layer, rows, cols, channels int) (*cv.Mat, error)
}

// Feather is a distance-weighted (alpha) blender. Each covered pixel is the
// weighted average of the contributing layers, using the feather ramp stored in
// [Layer].Weight. Because the ramp varies smoothly from the image centre to its
// border, overlapping regions cross-dissolve without a visible seam.
type Feather struct{}

// Blend implements [Blender] with per-pixel weighted averaging.
func (Feather) Blend(layers []Layer, rows, cols, channels int) (*cv.Mat, error) {
	out := cv.NewMat(rows, cols, channels)
	acc := make([]float64, rows*cols*channels)
	wsum := make([]float64, rows*cols)
	for _, l := range layers {
		for p := 0; p < rows*cols; p++ {
			w := l.Weight.Data[p]
			if w <= 0 {
				continue
			}
			wsum[p] += w
			base := p * channels
			for c := 0; c < channels; c++ {
				acc[base+c] += w * float64(l.Color.Data[base+c])
			}
		}
	}
	for p := 0; p < rows*cols; p++ {
		if wsum[p] <= 0 {
			continue
		}
		base := p * channels
		inv := 1 / wsum[p]
		for c := 0; c < channels; c++ {
			out.Data[base+c] = clampUint8(acc[base+c]*inv + 0.5)
		}
	}
	return out, nil
}

// MultiBandBlend is a Laplacian-pyramid (multi-band) blender. It decomposes each
// layer into a Laplacian pyramid and blends band by band using Gaussian-smoothed
// weights, so low-frequency content is blended over a wide transition while
// high-frequency detail is blended sharply. This suppresses both visible seams
// and ghosting better than [Feather] at the cost of more computation. Bands sets
// the number of pyramid levels (clamped to what the canvas size allows); a value
// of zero or less selects a sensible default.
type MultiBandBlend struct {
	Bands int
}

// Blend implements [Blender] with multi-band pyramid blending.
func (mb MultiBandBlend) Blend(layers []Layer, rows, cols, channels int) (*cv.Mat, error) {
	levels := pyramidLevels(rows, cols, mb.Bands)
	dims := levelDims(rows, cols, levels)

	// Per-pixel normalised weights at full resolution.
	wsum := make([]float64, rows*cols)
	for _, l := range layers {
		for p := 0; p < rows*cols; p++ {
			if l.Weight.Data[p] > 0 {
				wsum[p] += l.Weight.Data[p]
			}
		}
	}

	// Accumulate the blended Laplacian pyramid, one entry per level, holding
	// interleaved channels.
	blended := make([][]float64, levels)
	for l := range blended {
		blended[l] = make([]float64, dims[l].rows*dims[l].cols*channels)
	}

	for _, layer := range layers {
		// Normalised weight for this layer at full resolution.
		w := make([]float64, rows*cols)
		for p := 0; p < rows*cols; p++ {
			if wsum[p] > 0 && layer.Weight.Data[p] > 0 {
				w[p] = layer.Weight.Data[p] / wsum[p]
			}
		}
		gw := gaussianPyramid(w, dims) // weight Gaussian pyramid
		col := make([]float64, rows*cols*channels)
		for i := range col {
			col[i] = float64(layer.Color.Data[i])
		}
		lp := laplacianPyramid(col, dims, channels) // colour Laplacian pyramid
		for lv := 0; lv < levels; lv++ {
			npx := dims[lv].rows * dims[lv].cols
			for p := 0; p < npx; p++ {
				weight := gw[lv][p]
				if weight == 0 {
					continue
				}
				base := p * channels
				for c := 0; c < channels; c++ {
					blended[lv][base+c] += weight * lp[lv][base+c]
				}
			}
		}
	}

	final := collapsePyramid(blended, dims, channels)
	out := cv.NewMat(rows, cols, channels)
	// Zero the blend outside the panorama footprint so uncovered pixels stay
	// black despite pyramid bleeding.
	for p := 0; p < rows*cols; p++ {
		if wsum[p] <= 0 {
			continue
		}
		base := p * channels
		for c := 0; c < channels; c++ {
			out.Data[base+c] = clampUint8(final[base+c] + 0.5)
		}
	}
	return out, nil
}

// dim is the size of one pyramid level.
type dim struct{ rows, cols int }

// pyramidLevels chooses the number of pyramid levels, honouring the requested
// band count but never reducing a dimension below four pixels.
func pyramidLevels(rows, cols, bands int) int {
	if bands <= 0 {
		bands = 4
	}
	maxLv := 1
	r, c := rows, cols
	for maxLv < bands && r > 4 && c > 4 {
		r = (r + 1) / 2
		c = (c + 1) / 2
		maxLv++
	}
	return maxLv
}

// levelDims returns the size of each pyramid level, halving (rounding up) at
// every step.
func levelDims(rows, cols, levels int) []dim {
	dims := make([]dim, levels)
	dims[0] = dim{rows, cols}
	for l := 1; l < levels; l++ {
		dims[l] = dim{(dims[l-1].rows + 1) / 2, (dims[l-1].cols + 1) / 2}
	}
	return dims
}

// gaussianPyramid builds the Gaussian pyramid of a single-channel image over the
// given level sizes.
func gaussianPyramid(img []float64, dims []dim) [][]float64 {
	out := make([][]float64, len(dims))
	out[0] = img
	for l := 1; l < len(dims); l++ {
		out[l] = reduce(out[l-1], dims[l-1].rows, dims[l-1].cols, 1)
	}
	return out
}

// laplacianPyramid builds the Laplacian pyramid of an interleaved multi-channel
// image over the given level sizes. The coarsest level holds the residual
// Gaussian.
func laplacianPyramid(img []float64, dims []dim, channels int) [][]float64 {
	levels := len(dims)
	g := make([][]float64, levels)
	g[0] = img
	for l := 1; l < levels; l++ {
		g[l] = reduce(g[l-1], dims[l-1].rows, dims[l-1].cols, channels)
	}
	lp := make([][]float64, levels)
	for l := 0; l < levels-1; l++ {
		up := expand(g[l+1], dims[l+1].rows, dims[l+1].cols, dims[l].rows, dims[l].cols, channels)
		diff := make([]float64, len(g[l]))
		for i := range diff {
			diff[i] = g[l][i] - up[i]
		}
		lp[l] = diff
	}
	lp[levels-1] = g[levels-1]
	return lp
}

// collapsePyramid reconstructs an image from its blended Laplacian pyramid.
func collapsePyramid(pyr [][]float64, dims []dim, channels int) []float64 {
	levels := len(dims)
	cur := pyr[levels-1]
	for l := levels - 2; l >= 0; l-- {
		up := expand(cur, dims[l+1].rows, dims[l+1].cols, dims[l].rows, dims[l].cols, channels)
		next := make([]float64, len(pyr[l]))
		for i := range next {
			next[i] = pyr[l][i] + up[i]
		}
		cur = next
	}
	return cur
}

// binomial is the normalised 5-tap [1 4 6 4 1]/16 smoothing kernel shared by the
// pyramid reduce and expand steps.
var binomial = [5]float64{1.0 / 16, 4.0 / 16, 6.0 / 16, 4.0 / 16, 1.0 / 16}

// blur5 convolves an interleaved image with the separable 5-tap binomial kernel,
// replicating the border.
func blur5(img []float64, rows, cols, channels int) []float64 {
	tmp := make([]float64, len(img))
	clamp := func(v, hi int) int {
		if v < 0 {
			return 0
		}
		if v >= hi {
			return hi - 1
		}
		return v
	}
	// Horizontal pass.
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			for c := 0; c < channels; c++ {
				var s float64
				for k := -2; k <= 2; k++ {
					xx := clamp(x+k, cols)
					s += binomial[k+2] * img[(y*cols+xx)*channels+c]
				}
				tmp[(y*cols+x)*channels+c] = s
			}
		}
	}
	// Vertical pass.
	out := make([]float64, len(img))
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			for c := 0; c < channels; c++ {
				var s float64
				for k := -2; k <= 2; k++ {
					yy := clamp(y+k, rows)
					s += binomial[k+2] * tmp[(yy*cols+x)*channels+c]
				}
				out[(y*cols+x)*channels+c] = s
			}
		}
	}
	return out
}

// reduce blurs an interleaved image and subsamples by two, producing a level of
// size ((rows+1)/2)×((cols+1)/2).
func reduce(img []float64, rows, cols, channels int) []float64 {
	blurred := blur5(img, rows, cols, channels)
	dr := (rows + 1) / 2
	dc := (cols + 1) / 2
	out := make([]float64, dr*dc*channels)
	for y := 0; y < dr; y++ {
		for x := 0; x < dc; x++ {
			sy := y * 2
			sx := x * 2
			for c := 0; c < channels; c++ {
				out[(y*dc+x)*channels+c] = blurred[(sy*cols+sx)*channels+c]
			}
		}
	}
	return out
}

// expand upsamples an interleaved image to the target size by injecting zeros
// and smoothing, scaling by four so mean brightness is preserved.
func expand(img []float64, srows, scols, drows, dcols, channels int) []float64 {
	up := make([]float64, drows*dcols*channels)
	for y := 0; y < srows; y++ {
		dy := y * 2
		if dy >= drows {
			continue
		}
		for x := 0; x < scols; x++ {
			dx := x * 2
			if dx >= dcols {
				continue
			}
			for c := 0; c < channels; c++ {
				up[(dy*dcols+dx)*channels+c] = img[(y*scols+x)*channels+c]
			}
		}
	}
	blurred := blur5(up, drows, dcols, channels)
	for i := range blurred {
		blurred[i] *= 4
	}
	return blurred
}

// clampUint8 rounds and clamps a float sample into the [0,255] range.
func clampUint8(v float64) uint8 {
	if v <= 0 {
		return 0
	}
	if v >= 255 {
		return 255
	}
	return uint8(v)
}

// featherWeight builds the single-channel feather ramp for an image of the given
// size: each pixel's weight is its Chebyshev-style distance to the nearest
// border, linearly rescaled into the byte range [1,255] so every interior pixel
// has a positive, warpable weight while the ramp still falls toward the edges.
func featherWeight(rows, cols int) *cv.Mat {
	w := cv.NewMat(rows, cols, 1)
	maxd := (min(rows, cols) + 1) / 2
	if maxd < 1 {
		maxd = 1
	}
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			d := min(min(x+1, cols-x), min(y+1, rows-y))
			var q float64
			if maxd > 1 {
				q = 1 + 254*float64(d-1)/float64(maxd-1)
			} else {
				q = 255
			}
			w.Data[y*cols+x] = clampUint8(math.Floor(q + 0.5))
		}
	}
	return w
}
