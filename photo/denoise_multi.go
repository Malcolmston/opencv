package photo

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// FastNlMeansDenoisingMulti denoises the frame imgs[imgToDenoiseIndex] using a
// temporal window of temporalWindowSize consecutive frames centred on it. This
// is the multi-frame form of non-local means: for each output pixel the search
// scans the spatial window in every frame of the temporal window, so matching
// patches found in neighbouring frames (as arise when the same scene is
// captured repeatedly) contribute to the average. It suppresses noise far more
// effectively than single-frame denoising while preserving detail, because
// genuine structure is consistent across frames whereas noise is not.
//
// All frames must share the same dimensions and channel count. temporalWindowSize
// is forced to a positive odd integer (default 3); the window is clamped to the
// available frames at the sequence ends. h, templateWin and searchWin behave as
// in [FastNlMeansDenoising]. The function is channel-agnostic.
func FastNlMeansDenoisingMulti(imgs []*cv.Mat, imgToDenoiseIndex, temporalWindowSize int, h float64, templateWin, searchWin int) *cv.Mat {
	frames := validateSequence(imgs, imgToDenoiseIndex, "FastNlMeansDenoisingMulti")
	return nlMeansMulti(frames, imgToDenoiseIndex, temporalWindowSize, h, templateWin, searchWin)
}

// FastNlMeansDenoisingColoredMulti is the colour, multi-frame denoiser: it
// converts every frame to Y'CrCb, denoises the target frame's luma plane across
// the temporal window with strength h and its two chroma planes with strength
// hColor, then converts back to RGB. See [FastNlMeansDenoisingMulti] and
// [FastNlMeansDenoisingColored] for the parameter meanings. All frames must be
// three-channel and equally sized.
func FastNlMeansDenoisingColoredMulti(imgs []*cv.Mat, imgToDenoiseIndex, temporalWindowSize int, h, hColor float64, templateWin, searchWin int) *cv.Mat {
	frames := validateSequence(imgs, imgToDenoiseIndex, "FastNlMeansDenoisingColoredMulti")
	for _, f := range frames {
		requireChannels(f, 3, "FastNlMeansDenoisingColoredMulti")
	}
	ycc := make([]*cv.Mat, len(frames))
	for i, f := range frames {
		ycc[i] = cv.CvtColor(f, cv.ColorRGB2YCrCb)
	}
	// Build luma and chroma sequences.
	luma := make([]*cv.Mat, len(frames))
	chroma := make([]*cv.Mat, len(frames))
	for i, f := range ycc {
		pl := f.Split()
		luma[i] = pl[0]
		chroma[i] = cv.Merge([]*cv.Mat{pl[1], pl[2]})
	}
	yDen := nlMeansMulti(luma, imgToDenoiseIndex, temporalWindowSize, h, templateWin, searchWin)
	cDen := nlMeansMulti(chroma, imgToDenoiseIndex, temporalWindowSize, hColor, templateWin, searchWin)
	cd := cDen.Split()
	merged := cv.Merge([]*cv.Mat{yDen, cd[0], cd[1]})
	return cv.CvtColor(merged, cv.ColorYCrCb2RGB)
}

// validateSequence checks a frame sequence and returns it unchanged.
func validateSequence(imgs []*cv.Mat, idx int, name string) []*cv.Mat {
	if len(imgs) == 0 {
		panic("photo: " + name + " given no frames")
	}
	if idx < 0 || idx >= len(imgs) {
		panic("photo: " + name + " imgToDenoiseIndex out of range")
	}
	r, c, ch := imgs[0].Rows, imgs[0].Cols, imgs[0].Channels
	for _, f := range imgs {
		if f == nil || f.Empty() {
			panic("photo: " + name + " given an empty frame")
		}
		if f.Rows != r || f.Cols != c || f.Channels != ch {
			panic("photo: " + name + " requires uniform frame dimensions")
		}
	}
	return imgs
}

// nlMeansMulti is the shared multi-frame non-local means core.
func nlMeansMulti(frames []*cv.Mat, idx, temporalWindowSize int, h float64, templateWin, searchWin int) *cv.Mat {
	if h <= 0 {
		h = 1
	}
	temporalWindowSize = oddAtLeast(temporalWindowSize, 3)
	templateWin = oddAtLeast(templateWin, 3)
	searchWin = oddAtLeast(searchWin, 7)
	tr := templateWin / 2
	sr := searchWin / 2
	twr := temporalWindowSize / 2

	target := frames[idx]
	rows, cols, ch := target.Rows, target.Cols, target.Channels
	out := cv.NewMat(rows, cols, ch)

	// Temporal frame range, clamped to available frames.
	fLo := idx - twr
	if fLo < 0 {
		fLo = 0
	}
	fHi := idx + twr
	if fHi >= len(frames) {
		fHi = len(frames) - 1
	}

	patchSamples := float64((2*tr + 1) * (2*tr + 1) * ch)
	h2 := h * h
	acc := make([]float64, ch)

	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			for c := range acc {
				acc[c] = 0
			}
			var wsum float64
			for fi := fLo; fi <= fHi; fi++ {
				cand := frames[fi]
				for sy := y - sr; sy <= y+sr; sy++ {
					for sx := x - sr; sx <= x+sr; sx++ {
						var d float64
						for ty := -tr; ty <= tr; ty++ {
							for tx := -tr; tx <= tr; tx++ {
								for c := 0; c < ch; c++ {
									diff := float64(atRep(target, y+ty, x+tx, c)) - float64(atRep(cand, sy+ty, sx+tx, c))
									d += diff * diff
								}
							}
						}
						w := math.Exp(-(d / patchSamples) / h2)
						for c := 0; c < ch; c++ {
							acc[c] += w * float64(atRep(cand, sy, sx, c))
						}
						wsum += w
					}
				}
			}
			for c := 0; c < ch; c++ {
				out.Set(y, x, c, clampU8(acc[c]/wsum))
			}
		}
	}
	return out
}

// DenoiseTVL1 denoises a set of noisy observations of the same scene using the
// total-variation L1 (TV-L1) model, minimising the total variation of the
// result subject to an L1 fidelity to the observations. TV regularisation
// removes noise while, unlike quadratic smoothing, preserving sharp edges; the
// L1 data term makes it robust to outliers such as salt-and-pepper noise. The
// problem is solved with a Chambolle–Pock primal–dual iteration over niters
// steps. This mirrors OpenCV's denoise_TVL1.
//
// observations must all share the same dimensions and channel count (each
// channel is denoised independently); at least one observation is required.
// lambda is the fidelity weight — larger keeps the result closer to the data
// (less smoothing); it defaults to 1.0 when non-positive. niters defaults to 30
// when non-positive. The returned image has the observations' shape.
func DenoiseTVL1(observations []*cv.Mat, lambda float64, niters int) *cv.Mat {
	if len(observations) == 0 {
		panic("photo: DenoiseTVL1 given no observations")
	}
	r, c, ch := observations[0].Rows, observations[0].Cols, observations[0].Channels
	for _, o := range observations {
		if o == nil || o.Empty() {
			panic("photo: DenoiseTVL1 given an empty observation")
		}
		if o.Rows != r || o.Cols != c || o.Channels != ch {
			panic("photo: DenoiseTVL1 requires uniform observation dimensions")
		}
	}
	if lambda <= 0 {
		lambda = 1.0
	}
	if niters <= 0 {
		niters = 30
	}

	out := cv.NewMat(r, c, ch)
	// Data target: mean of the observations (per channel), in float.
	f := make([]float64, r*c)
	for chn := 0; chn < ch; chn++ {
		for i := range f {
			var s float64
			for _, o := range observations {
				s += float64(o.Data[i*ch+chn])
			}
			f[i] = s / float64(len(observations))
		}
		res := tvl1Channel(f, r, c, lambda, niters)
		for i := 0; i < r*c; i++ {
			out.Data[i*ch+chn] = clampU8(res[i])
		}
	}
	return out
}

// tvl1Channel runs the Chambolle–Pock TV-L1 solver on a single channel plane f
// and returns the denoised plane.
func tvl1Channel(f []float64, rows, cols int, lambda float64, niters int) []float64 {
	n := rows * cols
	u := make([]float64, n)
	uBar := make([]float64, n)
	copy(u, f)
	copy(uBar, f)
	px := make([]float64, n)
	py := make([]float64, n)

	// Step sizes: sigma*tau*L^2 <= 1 with L^2 = 8 (the discrete gradient norm).
	const sigma = 0.35
	const tau = 0.35

	at := func(a []float64, y, x int) float64 { return a[y*cols+x] }

	for it := 0; it < niters; it++ {
		// Dual update: p = proj_{|p|<=1}(p + sigma * grad(uBar)).
		for y := 0; y < rows; y++ {
			for x := 0; x < cols; x++ {
				var gx, gy float64
				if x < cols-1 {
					gx = at(uBar, y, x+1) - at(uBar, y, x)
				}
				if y < rows-1 {
					gy = at(uBar, y+1, x) - at(uBar, y, x)
				}
				nx := px[y*cols+x] + sigma*gx
				ny := py[y*cols+x] + sigma*gy
				norm := math.Hypot(nx, ny)
				if norm > 1 {
					nx /= norm
					ny /= norm
				}
				px[y*cols+x] = nx
				py[y*cols+x] = ny
			}
		}
		// Primal update: u = prox_{tau*lambda*|.-f|}(u + tau * div(p)).
		for y := 0; y < rows; y++ {
			for x := 0; x < cols; x++ {
				// Divergence of p (adjoint of the forward gradient).
				var div float64
				if x < cols-1 {
					div += px[y*cols+x]
				}
				if x > 0 {
					div -= px[y*cols+x-1]
				}
				if y < rows-1 {
					div += py[y*cols+x]
				}
				if y > 0 {
					div -= py[(y-1)*cols+x]
				}
				v := u[y*cols+x] + tau*div
				// Soft-shrink toward f (proximal operator of L1 fidelity).
				thr := tau * lambda
				d := v - f[y*cols+x]
				switch {
				case d > thr:
					v -= thr
				case d < -thr:
					v += thr
				default:
					v = f[y*cols+x]
				}
				uBar[y*cols+x] = 2*v - u[y*cols+x]
				u[y*cols+x] = v
			}
		}
	}
	return u
}
