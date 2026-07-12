package intensity

import (
	"fmt"
	"math"

	cv "github.com/malcolmston/opencv"
)

// UnsharpMask sharpens img by the classic unsharp-masking rule: it subtracts a
// Gaussian-blurred copy to form a high-pass "mask" and adds a multiple of it
// back,
//
//	out = in + amount · (in − G_sigma * in),
//
// so edges gain local contrast (with a slight overshoot) while flat regions are
// untouched. sigma sets the blur radius (larger sigma sharpens coarser
// structure), amount ≥ 0 sets the strength (0 is the identity, 1 doubles the
// high-frequency content), and threshold ≥ 0 suppresses sharpening where the
// local difference is smaller than it, which avoids amplifying noise and smooth
// gradients. Each channel is processed independently. It panics unless sigma > 0,
// amount ≥ 0 and threshold ≥ 0.
func UnsharpMask(img *cv.Mat, sigma, amount, threshold float64) *cv.Mat {
	requireImage(img, "UnsharpMask")
	if !(sigma > 0) || math.IsInf(sigma, 0) {
		panic(fmt.Sprintf("intensity: UnsharpMask requires sigma > 0, got %v", sigma))
	}
	if !(amount >= 0) {
		panic(fmt.Sprintf("intensity: UnsharpMask requires amount >= 0, got %v", amount))
	}
	if !(threshold >= 0) {
		panic(fmt.Sprintf("intensity: UnsharpMask requires threshold >= 0, got %v", threshold))
	}
	rows, cols, ch := img.Rows, img.Cols, img.Channels
	dst := cv.NewMat(rows, cols, ch)
	for c := 0; c < ch; c++ {
		plane := channelFloat(img, c)
		blur := blurPlaneFloat(plane, rows, cols, sigma)
		for p := range plane {
			diff := plane[p] - blur[p]
			v := plane[p]
			if math.Abs(diff) >= threshold {
				v += amount * diff
			}
			dst.Data[p*ch+c] = clampToUint8(v + 0.5)
		}
	}
	return dst
}

// DodgeAndBurn evens out an image's illumination the way a darkroom printer
// dodges shadows and burns highlights. It estimates the local brightness with a
// Gaussian blur of the luminance and multiplies each pixel by
//
//	gain = (neutral / localBrightness)^amount,
//
// where neutral is the mean luminance: pixels sitting in a darker-than-average
// surround are lightened (dodged) and those in a brighter-than-average surround
// are darkened (burned), which compresses the tonal range and reveals detail
// while leaving fine local contrast intact. sigma sets the surround radius and
// amount ≥ 0 sets the strength (0 is the identity). The same per-pixel gain is
// applied to every channel so colour is preserved. It panics unless sigma > 0
// and amount ≥ 0.
func DodgeAndBurn(img *cv.Mat, amount, sigma float64) *cv.Mat {
	requireImage(img, "DodgeAndBurn")
	if !(sigma > 0) || math.IsInf(sigma, 0) {
		panic(fmt.Sprintf("intensity: DodgeAndBurn requires sigma > 0, got %v", sigma))
	}
	if !(amount >= 0) {
		panic(fmt.Sprintf("intensity: DodgeAndBurn requires amount >= 0, got %v", amount))
	}
	rows, cols, ch := img.Rows, img.Cols, img.Channels
	luma := lumaFloat(img)
	neutral, _ := meanStd(luma)
	if neutral < 1 {
		neutral = 1
	}
	local := blurPlaneFloat(luma, rows, cols, sigma)

	dst := cv.NewMat(rows, cols, ch)
	n := img.Total()
	for p := 0; p < n; p++ {
		lb := local[p]
		if lb < 1 {
			lb = 1
		}
		gain := math.Pow(neutral/lb, amount)
		base := p * ch
		for c := 0; c < ch; c++ {
			dst.Data[base+c] = clampToUint8(float64(img.Data[base+c])*gain + 0.5)
		}
	}
	return dst
}

// CLAHEColor extends Contrast-Limited Adaptive Histogram Equalisation to colour
// images. A single-channel image is passed straight to [cv.CLAHE]. A
// three-channel image has CLAHE applied to its luminance only; every channel is
// then scaled by the ratio of the equalised to the original luminance, so local
// contrast is enhanced while hue and saturation are preserved (equalising the
// channels independently, which this deliberately avoids, would shift colours).
// Any other channel count is equalised channel by channel. clipLimit and
// tileGridSize are forwarded to [cv.CLAHE]; it panics (via that call) if
// tileGridSize < 1, or here on an empty image.
func CLAHEColor(img *cv.Mat, clipLimit float64, tileGridSize int) *cv.Mat {
	requireImage(img, "CLAHEColor")
	rows, cols, ch := img.Rows, img.Cols, img.Channels
	if ch == 1 {
		return cv.CLAHE(img, clipLimit, tileGridSize)
	}
	if ch != 3 {
		// Fall back to independent per-channel equalisation.
		planes := img.Split()
		for i, pl := range planes {
			planes[i] = cv.CLAHE(pl, clipLimit, tileGridSize)
		}
		return cv.Merge(planes)
	}

	luma := lumaFloat(img)
	y := cv.NewMat(rows, cols, 1)
	for p := range luma {
		y.Data[p] = clampToUint8(luma[p] + 0.5)
	}
	enh := cv.CLAHE(y, clipLimit, tileGridSize)

	dst := cv.NewMat(rows, cols, ch)
	n := img.Total()
	for p := 0; p < n; p++ {
		old := luma[p]
		var ratio float64
		if old < 1 {
			// Achromatic near-black pixel: add the luminance gain directly.
			ratio = 1
			base := p * ch
			delta := float64(enh.Data[p]) - old
			for c := 0; c < ch; c++ {
				dst.Data[base+c] = clampToUint8(float64(img.Data[base+c]) + delta + 0.5)
			}
			continue
		}
		ratio = float64(enh.Data[p]) / old
		base := p * ch
		for c := 0; c < ch; c++ {
			dst.Data[base+c] = clampToUint8(float64(img.Data[base+c])*ratio + 0.5)
		}
	}
	return dst
}
