package stitch

import (
	"errors"
	"math"

	cv "github.com/malcolmston/opencv"
)

// Layer is one source image warped onto the mosaic canvas together with a
// per-pixel blending weight over the same canvas. Image holds the resampled
// colour (zero outside the covered region) and Weight the coverage/feather
// weight: zero means the pixel is not covered by this layer and larger values
// mean the layer should dominate the blend there. All layers passed to a single
// blend must share the same dimensions.
type Layer struct {
	// Image is the canvas-sized colour of this layer.
	Image *cv.Mat
	// Weight is the canvas-sized blending weight, matching Image's rows and cols.
	Weight *cv.FloatMat
}

// Blender combines the overlapping layers of a mosaic into one image.
// Implementations include [FeatherBlender] and [MultiBandBlender].
type Blender interface {
	// Blend merges the layers into a single canvas-sized image.
	Blend(layers []Layer) (*cv.Mat, error)
}

var errNoLayers = errors.New("stitch: no layers to blend")

// checkLayers validates that layers is non-empty and every layer matches the
// first in size and channel count, returning those dimensions.
func checkLayers(layers []Layer) (rows, cols, ch int, err error) {
	if len(layers) == 0 {
		return 0, 0, 0, errNoLayers
	}
	rows = layers[0].Image.Rows
	cols = layers[0].Image.Cols
	ch = layers[0].Image.Channels
	for _, l := range layers {
		if l.Image.Rows != rows || l.Image.Cols != cols || l.Image.Channels != ch {
			return 0, 0, 0, errors.New("stitch: blend layers differ in size or channels")
		}
		if l.Weight.Rows != rows || l.Weight.Cols != cols {
			return 0, 0, 0, errors.New("stitch: blend layer weight size mismatch")
		}
	}
	return rows, cols, ch, nil
}

// FeatherWeightMap returns a width×height feather weight map that is largest at
// the image centre and tapers toward the border, so overlapping images
// cross-dissolve. Weights are the product of a horizontal and a vertical hat
// ramp, each raised to the sharpness power (sharpness of 1 is a plain linear
// ramp; larger values concentrate weight toward the centre). Every weight is
// strictly positive.
func FeatherWeightMap(width, height int, sharpness float64) *cv.FloatMat {
	m := cv.NewFloatMat(height, width)
	halfW := float64(width+1) / 2
	halfH := float64(height+1) / 2
	for y := 0; y < height; y++ {
		hy := float64(min(y+1, height-y)) / halfH
		for x := 0; x < width; x++ {
			hx := float64(min(x+1, width-x)) / halfW
			w := hx * hy
			if sharpness != 1 {
				w = math.Pow(w, sharpness)
			}
			m.Data[y*width+x] = w
		}
	}
	return m
}

// FeatherBlender averages the layers pixel by pixel, weighting each contribution
// by its [Layer].Weight. Because the weights vary smoothly from image centre to
// border, overlaps cross-dissolve without a hard seam. It is fast but can ghost
// when images are misaligned; [MultiBandBlender] handles that better.
type FeatherBlender struct{}

// NewFeatherBlender returns a ready-to-use feather blender.
func NewFeatherBlender() *FeatherBlender {
	return &FeatherBlender{}
}

// Blend implements [Blender] with per-pixel weighted averaging.
func (FeatherBlender) Blend(layers []Layer) (*cv.Mat, error) {
	rows, cols, ch, err := checkLayers(layers)
	if err != nil {
		return nil, err
	}
	out := cv.NewMat(rows, cols, ch)
	acc := make([]float64, rows*cols*ch)
	wsum := make([]float64, rows*cols)
	for _, l := range layers {
		for p := 0; p < rows*cols; p++ {
			w := l.Weight.Data[p]
			if w <= 0 {
				continue
			}
			wsum[p] += w
			base := p * ch
			for c := 0; c < ch; c++ {
				acc[base+c] += w * float64(l.Image.Data[base+c])
			}
		}
	}
	for p := 0; p < rows*cols; p++ {
		if wsum[p] <= 0 {
			continue
		}
		base := p * ch
		inv := 1 / wsum[p]
		for c := 0; c < ch; c++ {
			out.Data[base+c] = clampByte(acc[base+c] * inv)
		}
	}
	return out, nil
}

var pyrKernel = [5]float64{1.0 / 16, 4.0 / 16, 6.0 / 16, 4.0 / 16, 1.0 / 16}

// PyrDownFloat blurs src with a 5-tap Gaussian and subsamples it by two, halving
// each dimension (rounding up). It is the reduce step of a Gaussian pyramid.
// Border samples replicate the edge.
func PyrDownFloat(src *cv.FloatMat) *cv.FloatMat {
	rows, cols := src.Rows, src.Cols
	dstC := (cols + 1) / 2
	dstR := (rows + 1) / 2
	if dstC < 1 {
		dstC = 1
	}
	if dstR < 1 {
		dstR = 1
	}
	clampX := func(x int) int {
		if x < 0 {
			return 0
		}
		if x >= cols {
			return cols - 1
		}
		return x
	}
	clampY := func(y int) int {
		if y < 0 {
			return 0
		}
		if y >= rows {
			return rows - 1
		}
		return y
	}
	// Horizontal reduce into temp (rows × dstC).
	temp := make([]float64, rows*dstC)
	for y := 0; y < rows; y++ {
		for dx := 0; dx < dstC; dx++ {
			center := 2 * dx
			var s float64
			for k := 0; k < 5; k++ {
				s += pyrKernel[k] * src.Data[y*cols+clampX(center+k-2)]
			}
			temp[y*dstC+dx] = s
		}
	}
	// Vertical reduce into dst (dstR × dstC).
	dst := cv.NewFloatMat(dstR, dstC)
	for dy := 0; dy < dstR; dy++ {
		center := 2 * dy
		for x := 0; x < dstC; x++ {
			var s float64
			for k := 0; k < 5; k++ {
				s += pyrKernel[k] * temp[clampY(center+k-2)*dstC+x]
			}
			dst.Data[dy*dstC+x] = s
		}
	}
	return dst
}

// PyrUpFloat upsamples src to the given target dimensions by zero-insertion
// followed by a 5-tap Gaussian smoothing (scaled to preserve brightness). It is
// the expand step of a Laplacian pyramid; dstRows and dstCols are normally the
// dimensions of the next-finer pyramid level. Border samples replicate the edge.
func PyrUpFloat(src *cv.FloatMat, dstRows, dstCols int) *cv.FloatMat {
	up := make([]float64, dstRows*dstCols)
	for y := 0; y < src.Rows; y++ {
		yy := 2 * y
		if yy >= dstRows {
			continue
		}
		for x := 0; x < src.Cols; x++ {
			xx := 2 * x
			if xx >= dstCols {
				continue
			}
			up[yy*dstCols+xx] = src.Data[y*src.Cols+x]
		}
	}
	clampX := func(x int) int {
		if x < 0 {
			return 0
		}
		if x >= dstCols {
			return dstCols - 1
		}
		return x
	}
	clampY := func(y int) int {
		if y < 0 {
			return 0
		}
		if y >= dstRows {
			return dstRows - 1
		}
		return y
	}
	// Horizontal smooth (scale by 2 to compensate for zero insertion).
	temp := make([]float64, dstRows*dstCols)
	for y := 0; y < dstRows; y++ {
		for x := 0; x < dstCols; x++ {
			var s float64
			for k := 0; k < 5; k++ {
				s += pyrKernel[k] * up[y*dstCols+clampX(x+k-2)]
			}
			temp[y*dstCols+x] = 2 * s
		}
	}
	// Vertical smooth.
	dst := cv.NewFloatMat(dstRows, dstCols)
	for y := 0; y < dstRows; y++ {
		for x := 0; x < dstCols; x++ {
			var s float64
			for k := 0; k < 5; k++ {
				s += pyrKernel[k] * temp[clampY(y+k-2)*dstCols+x]
			}
			dst.Data[y*dstCols+x] = 2 * s
		}
	}
	return dst
}

// BuildGaussianPyramid returns the Gaussian pyramid of img with levels reduce
// steps: the result has length levels+1, element 0 being img itself and each
// subsequent element a [PyrDownFloat] of the one before. Reduction stops early if
// a dimension would fall below one pixel.
func BuildGaussianPyramid(img *cv.FloatMat, levels int) []*cv.FloatMat {
	pyr := make([]*cv.FloatMat, 0, levels+1)
	cur := img
	pyr = append(pyr, cur)
	for i := 0; i < levels; i++ {
		if cur.Rows <= 1 || cur.Cols <= 1 {
			break
		}
		cur = PyrDownFloat(cur)
		pyr = append(pyr, cur)
	}
	return pyr
}

// BuildLaplacianPyramid returns the Laplacian pyramid of img with levels bands.
// Each band k is G_k − expand(G_{k+1}) of the Gaussian pyramid, and the final
// element is the coarsest Gaussian residual, so [CollapseLaplacianPyramid]
// reconstructs img exactly (up to floating-point rounding). The result length is
// the Gaussian pyramid length.
func BuildLaplacianPyramid(img *cv.FloatMat, levels int) []*cv.FloatMat {
	g := BuildGaussianPyramid(img, levels)
	lap := make([]*cv.FloatMat, len(g))
	for k := 0; k < len(g)-1; k++ {
		up := PyrUpFloat(g[k+1], g[k].Rows, g[k].Cols)
		d := cv.NewFloatMat(g[k].Rows, g[k].Cols)
		for i := range d.Data {
			d.Data[i] = g[k].Data[i] - up.Data[i]
		}
		lap[k] = d
	}
	lap[len(g)-1] = g[len(g)-1]
	return lap
}

// CollapseLaplacianPyramid reconstructs the image from its Laplacian pyramid,
// the inverse of [BuildLaplacianPyramid]. It expands the coarsest level and adds
// each finer band in turn.
func CollapseLaplacianPyramid(pyr []*cv.FloatMat) *cv.FloatMat {
	if len(pyr) == 0 {
		return nil
	}
	cur := pyr[len(pyr)-1]
	for k := len(pyr) - 2; k >= 0; k-- {
		up := PyrUpFloat(cur, pyr[k].Rows, pyr[k].Cols)
		d := cv.NewFloatMat(pyr[k].Rows, pyr[k].Cols)
		for i := range d.Data {
			d.Data[i] = pyr[k].Data[i] + up.Data[i]
		}
		cur = d
	}
	return cur
}

// MultiBandBlender implements Burt–Adelson multi-band blending: each layer is
// decomposed into a Laplacian pyramid, the bands are merged using
// Gaussian-blurred weights at matching scales, and the result is collapsed back
// to an image. Blending across frequency bands avoids both the visible seams of
// hard cuts and the ghosting of plain feathering.
type MultiBandBlender struct {
	// NumBands is the number of pyramid levels used for blending. Values below 1
	// select an automatic depth based on the canvas size.
	NumBands int
}

// NewMultiBandBlender returns a multi-band blender using numBands pyramid levels.
// Pass a value below 1 to choose the depth automatically from the canvas size.
func NewMultiBandBlender(numBands int) *MultiBandBlender {
	return &MultiBandBlender{NumBands: numBands}
}

// Blend implements [Blender] with multi-band (pyramid) blending. Each channel is
// blended independently and the result is clamped to the 8-bit range.
func (b MultiBandBlender) Blend(layers []Layer) (*cv.Mat, error) {
	rows, cols, ch, err := checkLayers(layers)
	if err != nil {
		return nil, err
	}
	levels := b.NumBands
	maxLevels := 0
	for s := min(rows, cols); s > 1; s /= 2 {
		maxLevels++
	}
	if levels < 1 {
		levels = maxLevels
		if levels > 5 {
			levels = 5
		}
	}
	if levels > maxLevels {
		levels = maxLevels
	}
	if levels < 1 {
		levels = 1
	}

	// Gaussian weight pyramids per layer (shared across channels).
	weightPyr := make([][]*cv.FloatMat, len(layers))
	for i, l := range layers {
		weightPyr[i] = BuildGaussianPyramid(l.Weight, levels)
	}
	nLevels := len(weightPyr[0])

	out := cv.NewMat(rows, cols, ch)
	channel := cv.NewFloatMat(rows, cols)
	for c := 0; c < ch; c++ {
		// Accumulators per pyramid level.
		accum := make([]*cv.FloatMat, nLevels)
		wAccum := make([]*cv.FloatMat, nLevels)
		for i, l := range layers {
			// Extract channel c into a FloatMat and build its Laplacian pyramid.
			for p := 0; p < rows*cols; p++ {
				channel.Data[p] = float64(l.Image.Data[p*ch+c])
			}
			lap := BuildLaplacianPyramid(channel, levels)
			wp := weightPyr[i]
			for lv := 0; lv < nLevels; lv++ {
				if accum[lv] == nil {
					accum[lv] = cv.NewFloatMat(lap[lv].Rows, lap[lv].Cols)
					wAccum[lv] = cv.NewFloatMat(lap[lv].Rows, lap[lv].Cols)
				}
				w := wp[lv]
				for p := range lap[lv].Data {
					accum[lv].Data[p] += lap[lv].Data[p] * w.Data[p]
					wAccum[lv].Data[p] += w.Data[p]
				}
			}
		}
		// Normalise each level by its accumulated weight.
		blended := make([]*cv.FloatMat, nLevels)
		for lv := 0; lv < nLevels; lv++ {
			d := cv.NewFloatMat(accum[lv].Rows, accum[lv].Cols)
			for p := range d.Data {
				if wAccum[lv].Data[p] > 1e-12 {
					d.Data[p] = accum[lv].Data[p] / wAccum[lv].Data[p]
				}
			}
			blended[lv] = d
		}
		result := CollapseLaplacianPyramid(blended)
		for p := 0; p < rows*cols; p++ {
			out.Data[p*ch+c] = clampByte(result.Data[p])
		}
	}
	return out, nil
}
