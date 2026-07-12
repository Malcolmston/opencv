package dnn_superres

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// floatImage is a lightweight planar-interleaved float buffer used by the
// iterative refiners, where intermediate values legitimately go negative
// (back-projected residuals) and must not be clamped between passes.
type floatImage struct {
	rows, cols, ch int
	data           []float64
}

// newFloatImage allocates a zeroed float buffer of the given shape.
func newFloatImage(rows, cols, ch int) *floatImage {
	return &floatImage{rows: rows, cols: cols, ch: ch, data: make([]float64, rows*cols*ch)}
}

// floatFromMat lifts a uint8 Mat into a float buffer of identical shape.
func floatFromMat(m *cv.Mat) *floatImage {
	f := newFloatImage(m.Rows, m.Cols, m.Channels)
	for i, v := range m.Data {
		f.data[i] = float64(v)
	}
	return f
}

// toMat rounds and clamps a float buffer back to a uint8 Mat.
func (f *floatImage) toMat() *cv.Mat {
	m := cv.NewMat(f.rows, f.cols, f.ch)
	for i, v := range f.data {
		m.Data[i] = clampByte(v)
	}
	return m
}

// resampleFloat resizes a float buffer to (dstH, dstW) with the separable
// kernel k of the given support radius, using the same normalized,
// border-replicated scheme as [resampleSeparable] but keeping full float
// precision (no per-pass clamping). It underlies the iterative back-projection
// refiner, whose residuals are signed.
func resampleFloat(src *floatImage, dstW, dstH int, k kernelFunc, radius int) *floatImage {
	ch := src.ch
	scaleX := float64(src.cols) / float64(dstW)
	scaleY := float64(src.rows) / float64(dstH)

	xTaps := make([][]int, dstW)
	xW := make([][]float64, dstW)
	for x := 0; x < dstW; x++ {
		center := (float64(x)+0.5)*scaleX - 0.5
		base := int(math.Floor(center))
		var taps []int
		var ws []float64
		var sum float64
		for t := base - radius + 1; t <= base+radius; t++ {
			w := k(center - float64(t))
			if w == 0 {
				continue
			}
			taps = append(taps, clampInt(t, 0, src.cols-1))
			ws = append(ws, w)
			sum += w
		}
		if sum != 0 {
			for i := range ws {
				ws[i] /= sum
			}
		}
		xTaps[x] = taps
		xW[x] = ws
	}

	inter := make([]float64, src.rows*dstW*ch)
	for y := 0; y < src.rows; y++ {
		for x := 0; x < dstW; x++ {
			taps := xTaps[x]
			ws := xW[x]
			for c := 0; c < ch; c++ {
				var acc float64
				for i, tx := range taps {
					acc += ws[i] * src.data[(y*src.cols+tx)*ch+c]
				}
				inter[(y*dstW+x)*ch+c] = acc
			}
		}
	}

	dst := newFloatImage(dstH, dstW, ch)
	for y := 0; y < dstH; y++ {
		center := (float64(y)+0.5)*scaleY - 0.5
		base := int(math.Floor(center))
		var taps []int
		var ws []float64
		var sum float64
		for t := base - radius + 1; t <= base+radius; t++ {
			w := k(center - float64(t))
			if w == 0 {
				continue
			}
			taps = append(taps, clampInt(t, 0, src.rows-1))
			ws = append(ws, w)
			sum += w
		}
		if sum != 0 {
			for i := range ws {
				ws[i] /= sum
			}
		}
		for x := 0; x < dstW; x++ {
			for c := 0; c < ch; c++ {
				var acc float64
				for i, ty := range taps {
					acc += ws[i] * inter[(ty*dstW+x)*ch+c]
				}
				dst.data[(y*dstW+x)*ch+c] = acc
			}
		}
	}
	return dst
}

// downsampleAverage reduces a float buffer by an integer factor with
// non-overlapping box (area) averaging — the adjoint of nearest-block
// upsampling and the forward degradation model assumed by the back-projection
// refiner. The source dimensions must be exact multiples of factor.
func downsampleAverage(src *floatImage, factor int) *floatImage {
	lw, lh := src.cols/factor, src.rows/factor
	out := newFloatImage(lh, lw, src.ch)
	inv := 1.0 / float64(factor*factor)
	for y := 0; y < lh; y++ {
		for x := 0; x < lw; x++ {
			for c := 0; c < src.ch; c++ {
				var sum float64
				for dy := 0; dy < factor; dy++ {
					for dx := 0; dx < factor; dx++ {
						sum += src.data[((y*factor+dy)*src.cols+(x*factor+dx))*src.ch+c]
					}
				}
				out.data[(y*lw+x)*out.ch+c] = sum * inv
			}
		}
	}
	return out
}

// UpsampleScale enlarges src by an arbitrary integer scale (>= 2) with bicubic
// (Keys/Catmull-Rom) convolution. It is the general-purpose arbitrary-factor
// entry point — [UpsampleBicubic] is limited to the trained-model scales 2, 3
// and 4, whereas this accepts any factor. It returns an error for an empty
// image or a scale below 2.
func UpsampleScale(src *cv.Mat, scale int) (*cv.Mat, error) {
	if err := validateAnyScale(src, scale); err != nil {
		return nil, err
	}
	return resampleSeparable(src, src.Cols*scale, src.Rows*scale, keysCubic, 2), nil
}

// IterativeBackProjection enlarges src by an arbitrary integer scale (>= 2) and
// then refines the result with iterative back-projection (IBP), the classic
// reconstruction-based super-resolution loop. Starting from a bicubic estimate
// H, each iteration simulates the low-resolution image by area-downsampling H,
// measures the residual against the true low-resolution src, upsamples that
// residual and adds a fraction (beta) of it back into H. The estimate therefore
// converges toward an image that reproduces src exactly under the downsampling
// model, sharpening edges beyond plain interpolation.
//
// iterations controls how many refinement passes run; values of 5–20 are
// typical and it must be at least 1. It returns an error for an empty image, a
// scale below 2, or iterations below 1.
func IterativeBackProjection(src *cv.Mat, scale, iterations int) (*cv.Mat, error) {
	if err := validateAnyScale(src, scale); err != nil {
		return nil, err
	}
	if iterations < 1 {
		return nil, errIterations
	}
	const beta = 0.75
	lr := floatFromMat(src)
	hrW, hrH := src.Cols*scale, src.Rows*scale
	hr := resampleFloat(lr, hrW, hrH, keysCubic, 2)
	for it := 0; it < iterations; it++ {
		simLR := downsampleAverage(hr, scale)
		// Residual in low-resolution space.
		res := newFloatImage(lr.rows, lr.cols, lr.ch)
		for i := range res.data {
			res.data[i] = lr.data[i] - simLR.data[i]
		}
		// Back-project the residual into high-resolution space and correct.
		resHR := resampleFloat(res, hrW, hrH, keysCubic, 2)
		for i := range hr.data {
			hr.data[i] += beta * resHR.data[i]
		}
	}
	return hr.toMat(), nil
}

// UpsampleIBP is a convenience wrapper over [IterativeBackProjection] using a
// fixed, well-behaved iteration count (10). It enlarges src by an arbitrary
// integer scale (>= 2), returning an error for an empty image or a scale below
// 2.
func UpsampleIBP(src *cv.Mat, scale int) (*cv.Mat, error) {
	return IterativeBackProjection(src, scale, 10)
}

// lapDouble performs one LapSRN-style ×2 upscaling step: a bicubic base is
// doubled and a high-frequency residual — an unsharp/Laplacian detail band
// gated by local edge strength — is added on top, the weight-free analogue of a
// LapSRN pyramid level's learned residual prediction.
func lapDouble(src *cv.Mat) *cv.Mat {
	base := resampleSeparable(src, src.Cols*2, src.Rows*2, keysCubic, 2)
	// Detail band: base minus a Gaussian-blurred base, added back to steepen
	// edges. Amount is modest so repeated levels stay stable.
	return unsharpSeparable(base, 0.6, 3, 1.0)
}

// UpsampleLapSRN enlarges src by an arbitrary integer scale (>= 2) with a
// LapSRN-style progressive (Laplacian-pyramid) upscaler. The image is doubled
// repeatedly (×2 → ×4 → ×8 …), and at every level a bicubic base is combined
// with an edge-gated high-frequency residual, mirroring how the trained LapSRN
// network predicts residuals level by level — but using fixed kernels and no
// learned weights. When scale is not a power of two, the residual doublings run
// as far as they fit and a final bicubic resample lands the image on the exact
// target size.
//
// This is NOT the trained LapSRN network; it contains no weights. It returns an
// error for an empty image or a scale below 2.
func UpsampleLapSRN(src *cv.Mat, scale int) (*cv.Mat, error) {
	if err := validateAnyScale(src, scale); err != nil {
		return nil, err
	}
	cur := src.Clone()
	curScale := 1
	for curScale*2 <= scale {
		cur = lapDouble(cur)
		curScale *= 2
	}
	if curScale != scale {
		cur = resampleSeparable(cur, src.Cols*scale, src.Rows*scale, keysCubic, 2)
	}
	return cur, nil
}

// UpsampleESPCN enlarges src by an arbitrary integer scale (>= 2) with an
// ESPCN-style sub-pixel (pixel-shuffle) upscaler. For scale factor r the method
// builds r fixed one-dimensional polyphase filters — the r phase decompositions
// of a Keys cubic kernel — and produces the r² feature planes that an ESPCN's
// final sub-pixel convolution layer would, then interleaves them with the
// pixel-shuffle (depth-to-space) rearrangement to form the high-resolution grid.
// Each output sub-pixel position is thus reconstructed by its own dedicated
// phase filter, exactly as in ESPCN, but the filters are fixed rather than
// learned.
//
// This is NOT the trained ESPCN network; it contains no weights. It returns an
// error for an empty image or a scale below 2.
func UpsampleESPCN(src *cv.Mat, scale int) (*cv.Mat, error) {
	if err := validateAnyScale(src, scale); err != nil {
		return nil, err
	}
	r := scale
	ch := src.Channels
	// Build the r polyphase tap sets of the Keys cubic. Phase p corresponds to
	// fractional sample offset dp = (p+0.5)/r - 0.5 within a source cell; the
	// support is the 4 taps at relative source offsets -1,0,1,2.
	const radius = 2
	type tap struct {
		off int
		w   float64
	}
	phase := make([][]tap, r)
	for p := 0; p < r; p++ {
		dp := (float64(p)+0.5)/float64(r) - 0.5
		var taps []tap
		var sum float64
		for m := -radius + 1; m <= radius; m++ {
			w := keysCubic(dp - float64(m))
			if w == 0 {
				continue
			}
			taps = append(taps, tap{off: m, w: w})
			sum += w
		}
		for i := range taps {
			taps[i].w /= sum
		}
		phase[p] = taps
	}

	sw, sh := src.Cols, src.Rows
	dstW, dstH := sw*r, sh*r
	// Horizontal sub-pixel convolution: src.Rows × dstW feature buffer.
	inter := make([]float64, sh*dstW*ch)
	for y := 0; y < sh; y++ {
		for x := 0; x < sw; x++ {
			for px := 0; px < r; px++ {
				taps := phase[px]
				ox := x*r + px
				for c := 0; c < ch; c++ {
					var acc float64
					for _, t := range taps {
						sx := clampInt(x+t.off, 0, sw-1)
						acc += t.w * float64(src.Data[(y*sw+sx)*ch+c])
					}
					inter[(y*dstW+ox)*ch+c] = acc
				}
			}
		}
	}
	// Vertical sub-pixel convolution + pixel shuffle into the destination.
	dst := cv.NewMat(dstH, dstW, ch)
	for y := 0; y < sh; y++ {
		for py := 0; py < r; py++ {
			taps := phase[py]
			oy := y*r + py
			for x := 0; x < dstW; x++ {
				for c := 0; c < ch; c++ {
					var acc float64
					for _, t := range taps {
						sy := clampInt(y+t.off, 0, sh-1)
						acc += t.w * inter[(sy*dstW+x)*ch+c]
					}
					dst.Data[(oy*dstW+x)*ch+c] = clampByte(acc)
				}
			}
		}
	}
	return dst, nil
}
