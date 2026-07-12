package photo

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// FastNlMeansDenoising removes noise with the non-local means algorithm: each
// output sample is a weighted average of the samples in a searchWin×searchWin
// window around it, where the weight of a candidate falls off exponentially
// with the sum-of-squared difference between the templateWin×templateWin patch
// centred on the target and the patch centred on the candidate. Similar patches
// (as arise repeatedly in natural images and, trivially, in flat regions)
// dominate the average, so noise is suppressed while structure is preserved.
//
// h controls the decay: larger h averages more aggressively (stronger
// smoothing), smaller h preserves more detail. templateWin and searchWin are
// forced to positive odd integers (defaults 3 and 7 when non-positive). Border
// samples are edge-replicated. The function is channel-agnostic: for
// multi-channel input the patch distance is summed jointly across channels and
// every channel is averaged with the shared weights. This is the exact
// (unaccelerated) estimator, so keep both windows small — its cost grows as
// searchWin²·templateWin² per pixel.
func FastNlMeansDenoising(img *cv.Mat, h float64, templateWin, searchWin int) *cv.Mat {
	if img == nil || img.Empty() {
		panic("photo: FastNlMeansDenoising given an empty image")
	}
	return nlMeans(img, h, templateWin, searchWin)
}

// FastNlMeansDenoisingColored denoises a three-channel RGB image, decoupling
// luminance from colour. The image is converted to Y'CrCb; the luma plane is
// denoised with strength h and the two chroma planes (jointly) with strength
// hColor, then the result is converted back to RGB. Chroma noise is usually
// smoothed harder than luma (hColor >= h) because the eye tolerates colour
// blurring better than luminance blurring. templateWin and searchWin behave as
// in [FastNlMeansDenoising].
func FastNlMeansDenoisingColored(img *cv.Mat, h, hColor float64, templateWin, searchWin int) *cv.Mat {
	if img == nil || img.Empty() {
		panic("photo: FastNlMeansDenoisingColored given an empty image")
	}
	requireChannels(img, 3, "FastNlMeansDenoisingColored")
	ycc := cv.CvtColor(img, cv.ColorRGB2YCrCb)
	planes := ycc.Split() // Y, Cr, Cb
	yDen := nlMeans(planes[0], h, templateWin, searchWin)
	crcb := cv.Merge([]*cv.Mat{planes[1], planes[2]})
	crcbDen := nlMeans(crcb, hColor, templateWin, searchWin)
	cd := crcbDen.Split()
	merged := cv.Merge([]*cv.Mat{yDen, cd[0], cd[1]})
	return cv.CvtColor(merged, cv.ColorYCrCb2RGB)
}

// nlMeans is the shared non-local means core, working on any channel count.
func nlMeans(img *cv.Mat, h float64, templateWin, searchWin int) *cv.Mat {
	if h <= 0 {
		h = 1
	}
	templateWin = oddAtLeast(templateWin, 3)
	searchWin = oddAtLeast(searchWin, 7)
	tr := templateWin / 2
	sr := searchWin / 2
	rows, cols, ch := img.Rows, img.Cols, img.Channels
	out := cv.NewMat(rows, cols, ch)

	// Patch distance is normalised by the number of samples in a patch so that h
	// has a channel-count-independent meaning.
	patchSamples := float64((2*tr + 1) * (2*tr + 1) * ch)
	h2 := h * h
	acc := make([]float64, ch)

	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			for c := range acc {
				acc[c] = 0
			}
			var wsum float64
			for sy := y - sr; sy <= y+sr; sy++ {
				for sx := x - sr; sx <= x+sr; sx++ {
					// Squared L2 distance between the two patches.
					var d float64
					for ty := -tr; ty <= tr; ty++ {
						for tx := -tr; tx <= tr; tx++ {
							for c := 0; c < ch; c++ {
								diff := float64(atRep(img, y+ty, x+tx, c)) - float64(atRep(img, sy+ty, sx+tx, c))
								d += diff * diff
							}
						}
					}
					w := math.Exp(-(d / patchSamples) / h2)
					for c := 0; c < ch; c++ {
						acc[c] += w * float64(atRep(img, sy, sx, c))
					}
					wsum += w
				}
			}
			for c := 0; c < ch; c++ {
				out.Set(y, x, c, clampU8(acc[c]/wsum))
			}
		}
	}
	return out
}
