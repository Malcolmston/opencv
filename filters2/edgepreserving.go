package filters2

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// BilateralFilter applies an edge-preserving bilateral filter. Each output
// sample is a weighted average of its neighbourhood in which the weight is the
// product of a spatial Gaussian (standard deviation sigmaSpace) and a range
// Gaussian on colour difference (standard deviation sigmaColor). The diameter d
// sets the neighbourhood size; if d <= 0 it is derived from sigmaSpace. For
// multi-channel images the range term uses the Euclidean distance across all
// channels so colours are smoothed jointly. It panics on empty input.
func BilateralFilter(src *cv.Mat, d int, sigmaColor, sigmaSpace float64) *cv.Mat {
	requireNonEmpty(src, "BilateralFilter")
	requireChannels(src, "BilateralFilter")
	if sigmaColor <= 0 {
		sigmaColor = 1
	}
	if sigmaSpace <= 0 {
		sigmaSpace = 1
	}
	if d <= 0 {
		d = 2*int(math.Round(1.5*sigmaSpace)) + 1
	}
	if d%2 == 0 {
		d++
	}
	radius := d / 2
	return bilateralCore(src, src, radius, sigmaColor, sigmaSpace)
}

// JointBilateralFilter applies a cross (joint) bilateral filter: the spatial
// term and the averaging act on src, but the range weight is computed from the
// separate guide image. This smooths src while preserving the edges present in
// guide, and is the standard tool for upsampling or denoising one channel using
// a cleaner reference. src and guide must have identical dimensions and channel
// counts. If d <= 0 it is derived from sigmaSpace. It panics on empty input or
// a shape mismatch.
func JointBilateralFilter(src, guide *cv.Mat, d int, sigmaColor, sigmaSpace float64) *cv.Mat {
	requireSameShape(src, guide, "JointBilateralFilter")
	requireChannels(src, "JointBilateralFilter")
	if sigmaColor <= 0 {
		sigmaColor = 1
	}
	if sigmaSpace <= 0 {
		sigmaSpace = 1
	}
	if d <= 0 {
		d = 2*int(math.Round(1.5*sigmaSpace)) + 1
	}
	if d%2 == 0 {
		d++
	}
	radius := d / 2
	return bilateralCore(src, guide, radius, sigmaColor, sigmaSpace)
}

// bilateralCore averages src using spatial weights and range weights taken from
// guide (guide == src for the plain bilateral filter).
func bilateralCore(src, guide *cv.Mat, radius int, sigmaColor, sigmaSpace float64) *cv.Mat {
	dst := like(src)
	gs2 := 2 * sigmaSpace * sigmaSpace
	gc2 := 2 * sigmaColor * sigmaColor
	// Precompute spatial weights.
	span := 2*radius + 1
	spatial := make([]float64, span*span)
	si := 0
	for dy := -radius; dy <= radius; dy++ {
		for dx := -radius; dx <= radius; dx++ {
			spatial[si] = math.Exp(-float64(dx*dx+dy*dy) / gs2)
			si++
		}
	}
	ch := src.Channels
	for y := 0; y < src.Rows; y++ {
		for x := 0; x < src.Cols; x++ {
			var sum [8]float64
			var wsum float64
			si := 0
			for dy := -radius; dy <= radius; dy++ {
				for dx := -radius; dx <= radius; dx++ {
					// Range distance from the guide image.
					var cd2 float64
					for c := 0; c < ch; c++ {
						gc := float64(atReplicate(guide, y, x, c))
						gn := float64(atReplicate(guide, y+dy, x+dx, c))
						diff := gn - gc
						cd2 += diff * diff
					}
					w := spatial[si] * math.Exp(-cd2/gc2)
					for c := 0; c < ch; c++ {
						sum[c] += w * float64(atReplicate(src, y+dy, x+dx, c))
					}
					wsum += w
					si++
				}
			}
			base := (y*src.Cols + x) * ch
			for c := 0; c < ch; c++ {
				dst.Data[base+c] = clampU8(sum[c] / wsum)
			}
		}
	}
	return dst
}

// GuidedFilter applies the guided filter of He, Sun and Tang: an
// edge-preserving smoothing of src whose edges follow the single-channel guide
// image. radius sets the square window half-width and eps (in squared
// normalised-intensity units, i.e. samples scaled to [0,1]) controls the degree
// of smoothing — larger eps smooths more. src may have any channel count and is
// filtered per channel against the shared guide. With guide == src and eps == 0
// the filter is an identity on the interior. It panics on empty input, a
// multi-channel guide, or mismatched dimensions.
func GuidedFilter(src, guide *cv.Mat, radius int, eps float64) *cv.Mat {
	requireNonEmpty(src, "GuidedFilter")
	requireGray(guide, "GuidedFilter")
	if src.Rows != guide.Rows || src.Cols != guide.Cols {
		panic("filters2: GuidedFilter: src and guide dimensions must match")
	}
	if radius < 1 {
		panic("filters2: GuidedFilter requires radius >= 1")
	}
	rows, cols := src.Rows, src.Cols
	// Guide in [0,1].
	I := NewFloatImage(rows, cols)
	for i, v := range guide.Data {
		I.Data[i] = float64(v) / 255
	}
	meanI := boxMean(I, radius)
	corrI := boxMean(mulFloat(I, I), radius)
	varI := NewFloatImage(rows, cols)
	for i := range varI.Data {
		varI.Data[i] = corrI.Data[i] - meanI.Data[i]*meanI.Data[i]
	}

	dst := like(src)
	ch := src.Channels
	for c := 0; c < ch; c++ {
		p := NewFloatImage(rows, cols)
		for i := 0; i < rows*cols; i++ {
			p.Data[i] = float64(src.Data[i*ch+c]) / 255
		}
		meanP := boxMean(p, radius)
		corrIp := boxMean(mulFloat(I, p), radius)
		a := NewFloatImage(rows, cols)
		b := NewFloatImage(rows, cols)
		for i := range a.Data {
			covIp := corrIp.Data[i] - meanI.Data[i]*meanP.Data[i]
			ai := covIp / (varI.Data[i] + eps)
			a.Data[i] = ai
			b.Data[i] = meanP.Data[i] - ai*meanI.Data[i]
		}
		meanA := boxMean(a, radius)
		meanB := boxMean(b, radius)
		for i := 0; i < rows*cols; i++ {
			q := meanA.Data[i]*I.Data[i] + meanB.Data[i]
			dst.Data[i*ch+c] = clampU8(q * 255)
		}
	}
	return dst
}

// mulFloat returns the pointwise product of two equally sized float images.
func mulFloat(a, b *FloatImage) *FloatImage {
	out := NewFloatImage(a.Rows, a.Cols)
	for i := range a.Data {
		out.Data[i] = a.Data[i] * b.Data[i]
	}
	return out
}

// boxMean returns the mean of f over a (2r+1)×(2r+1) window with edge
// replication, computed via a separable running sum.
func boxMean(f *FloatImage, r int) *FloatImage {
	rows, cols := f.Rows, f.Cols
	tmp := NewFloatImage(rows, cols)
	win := float64(2*r + 1)
	// Horizontal pass.
	for y := 0; y < rows; y++ {
		base := y * cols
		var sum float64
		for k := -r; k <= r; k++ {
			sum += f.Data[base+clampIdx(k, cols)]
		}
		tmp.Data[base] = sum / win
		for x := 1; x < cols; x++ {
			sum += f.Data[base+clampIdx(x+r, cols)] - f.Data[base+clampIdx(x-r-1, cols)]
			tmp.Data[base+x] = sum / win
		}
	}
	// Vertical pass.
	out := NewFloatImage(rows, cols)
	for x := 0; x < cols; x++ {
		var sum float64
		for k := -r; k <= r; k++ {
			sum += tmp.Data[clampIdx(k, rows)*cols+x]
		}
		out.Data[x] = sum / win
		for y := 1; y < rows; y++ {
			sum += tmp.Data[clampIdx(y+r, rows)*cols+x] - tmp.Data[clampIdx(y-r-1, rows)*cols+x]
			out.Data[y*cols+x] = sum / win
		}
	}
	return out
}

// NonLocalMeans denoises src with the non-local means algorithm of Buades,
// Coll and Morel. For every pixel it forms a weighted average of pixels within
// a (2*searchRadius+1) search window, weighting each by the Gaussian similarity
// of a (2*templateRadius+1) patch around it to the reference patch. The filter
// strength h controls how fast weights decay with patch distance (larger h
// removes more noise and detail). For multi-channel images patch distance sums
// over all channels. It panics on empty input or non-positive radii.
func NonLocalMeans(src *cv.Mat, h float64, templateRadius, searchRadius int) *cv.Mat {
	requireNonEmpty(src, "NonLocalMeans")
	requireChannels(src, "NonLocalMeans")
	if templateRadius < 1 || searchRadius < 1 {
		panic("filters2: NonLocalMeans requires positive templateRadius and searchRadius")
	}
	if h <= 0 {
		h = 1
	}
	rows, cols, ch := src.Rows, src.Cols, src.Channels
	dst := like(src)
	patchCount := float64((2*templateRadius + 1) * (2*templateRadius + 1) * ch)
	h2 := h * h
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			var sum [8]float64
			var wsum float64
			var maxW float64
			for sy := -searchRadius; sy <= searchRadius; sy++ {
				for sx := -searchRadius; sx <= searchRadius; sx++ {
					if sy == 0 && sx == 0 {
						continue
					}
					// Sum of squared patch differences.
					var d2 float64
					for py := -templateRadius; py <= templateRadius; py++ {
						for px := -templateRadius; px <= templateRadius; px++ {
							for c := 0; c < ch; c++ {
								a := float64(atReplicate(src, y+py, x+px, c))
								b := float64(atReplicate(src, y+sy+py, x+sx+px, c))
								diff := a - b
								d2 += diff * diff
							}
						}
					}
					w := math.Exp(-d2 / (patchCount * h2))
					if w > maxW {
						maxW = w
					}
					for c := 0; c < ch; c++ {
						sum[c] += w * float64(atReplicate(src, y+sy, x+sx, c))
					}
					wsum += w
				}
			}
			// The reference pixel itself is weighted by the maximum neighbour
			// weight (standard NLM practice) to avoid self-domination.
			if maxW == 0 {
				maxW = 1
			}
			for c := 0; c < ch; c++ {
				sum[c] += maxW * float64(atReplicate(src, y, x, c))
			}
			wsum += maxW
			base := (y*cols + x) * ch
			for c := 0; c < ch; c++ {
				dst.Data[base+c] = clampU8(sum[c] / wsum)
			}
		}
	}
	return dst
}

// DiffusionOption selects the conduction (edge-stopping) function used by
// [AnisotropicDiffusion].
type DiffusionOption int

const (
	// DiffusionExponential uses g(x) = exp(-(x/kappa)^2), which privileges
	// high-contrast edges over low-contrast ones (Perona-Malik option 1).
	DiffusionExponential DiffusionOption = iota
	// DiffusionQuadratic uses g(x) = 1/(1+(x/kappa)^2), which privileges wide
	// regions over smaller ones (Perona-Malik option 2).
	DiffusionQuadratic
)

// AnisotropicDiffusion applies iterations of Perona-Malik anisotropic
// diffusion, an edge-preserving smoothing that damps intra-region gradients
// while conserving edges. kappa is the gradient contrast threshold, lambda the
// integration step (stable for 0 < lambda <= 0.25) and option selects the
// conduction function. Multi-channel images are diffused per channel. It panics
// on empty input or a non-positive iteration count.
func AnisotropicDiffusion(src *cv.Mat, iterations int, kappa, lambda float64, option DiffusionOption) *cv.Mat {
	requireNonEmpty(src, "AnisotropicDiffusion")
	if iterations < 1 {
		panic("filters2: AnisotropicDiffusion requires iterations >= 1")
	}
	if kappa <= 0 {
		kappa = 1
	}
	if lambda <= 0 || lambda > 0.25 {
		lambda = 0.25
	}
	rows, cols, ch := src.Rows, src.Cols, src.Channels
	dst := like(src)
	g := func(grad float64) float64 {
		q := grad / kappa
		if option == DiffusionQuadratic {
			return 1 / (1 + q*q)
		}
		return math.Exp(-q * q)
	}
	for c := 0; c < ch; c++ {
		// Working buffer for this channel.
		buf := make([]float64, rows*cols)
		for i := 0; i < rows*cols; i++ {
			buf[i] = float64(src.Data[i*ch+c])
		}
		next := make([]float64, rows*cols)
		at := func(b []float64, y, x int) float64 {
			return b[clampIdx(y, rows)*cols+clampIdx(x, cols)]
		}
		for it := 0; it < iterations; it++ {
			for y := 0; y < rows; y++ {
				for x := 0; x < cols; x++ {
					cur := buf[y*cols+x]
					dn := at(buf, y-1, x) - cur
					ds := at(buf, y+1, x) - cur
					de := at(buf, y, x+1) - cur
					dw := at(buf, y, x-1) - cur
					upd := g(math.Abs(dn))*dn + g(math.Abs(ds))*ds +
						g(math.Abs(de))*de + g(math.Abs(dw))*dw
					next[y*cols+x] = cur + lambda*upd
				}
			}
			buf, next = next, buf
		}
		for i := 0; i < rows*cols; i++ {
			dst.Data[i*ch+c] = clampU8(buf[i])
		}
	}
	return dst
}

// KuwaharaFilter applies the Kuwahara edge-preserving smoothing filter. For a
// window of the given odd size it splits the neighbourhood into four
// overlapping quadrants, and replaces the centre sample with the mean of the
// quadrant having the lowest intensity variance, thereby smoothing flat areas
// while keeping edges sharp. Multi-channel images choose the quadrant by total
// variance across channels and output that quadrant's per-channel mean. It
// panics on empty input or a non-positive even size.
func KuwaharaFilter(src *cv.Mat, size int) *cv.Mat {
	requireNonEmpty(src, "KuwaharaFilter")
	requireOddPositive(size, "KuwaharaFilter")
	requireChannels(src, "KuwaharaFilter")
	rows, cols, ch := src.Rows, src.Cols, src.Channels
	dst := like(src)
	a := size / 2 // quadrant half-size; each quadrant is (a+1)x(a+1)
	// Quadrant offset origins relative to the centre.
	quadrants := [4][2]int{{-a, -a}, {-a, 0}, {0, -a}, {0, 0}}
	q := a + 1
	n := float64(q * q)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			bestVar := math.Inf(1)
			var bestMean [8]float64
			for _, off := range quadrants {
				var mean [8]float64
				var sumSq [8]float64
				for dy := 0; dy < q; dy++ {
					for dx := 0; dx < q; dx++ {
						yy := y + off[0] + dy
						xx := x + off[1] + dx
						for c := 0; c < ch; c++ {
							v := float64(atReplicate(src, yy, xx, c))
							mean[c] += v
							sumSq[c] += v * v
						}
					}
				}
				var totalVar float64
				for c := 0; c < ch; c++ {
					mean[c] /= n
					totalVar += sumSq[c]/n - mean[c]*mean[c]
				}
				if totalVar < bestVar {
					bestVar = totalVar
					bestMean = mean
				}
			}
			base := (y*cols + x) * ch
			for c := 0; c < ch; c++ {
				dst.Data[base+c] = clampU8(bestMean[c])
			}
		}
	}
	return dst
}
