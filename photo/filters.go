package photo

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// EdgePreservingFlag selects the edge-preserving filter variant. Both variants
// are realised here through [cv.BilateralFilter]; the flag is retained for API
// familiarity and documents the intended smoothing character.
type EdgePreservingFlag int

const (
	// RecursFilter requests OpenCV's recursive-filter variant. Here it maps to a
	// bilateral filter.
	RecursFilter EdgePreservingFlag = 1
	// NormconvFilter requests OpenCV's normalized-convolution variant. Here it
	// also maps to a bilateral filter.
	NormconvFilter EdgePreservingFlag = 2
)

// EdgePreservingFilter smooths img while keeping strong edges crisp. sigmaS is
// the spatial extent of the smoothing (in OpenCV's convention, roughly 0..200);
// it is scaled down internally to a bilateral spatial sigma so the kernel stays
// small and fast. sigmaR is the range (intensity) sensitivity in [0,1]: larger
// values blur across bigger intensity gaps. The implementation delegates to
// [cv.BilateralFilter], so it is channel-agnostic.
func EdgePreservingFilter(img *cv.Mat, flags EdgePreservingFlag, sigmaS, sigmaR float64) *cv.Mat {
	if img == nil || img.Empty() {
		panic("photo: EdgePreservingFilter given an empty image")
	}
	_ = flags
	if sigmaS <= 0 {
		sigmaS = 60
	}
	if sigmaR <= 0 {
		sigmaR = 0.4
	}
	// Scale OpenCV's large sigmaS down to a modest bilateral spatial sigma.
	sigmaSpace := sigmaS / 10
	if sigmaSpace < 1 {
		sigmaSpace = 1
	}
	d := 2*int(math.Round(sigmaSpace)) + 1
	return cv.BilateralFilter(img, d, sigmaR*255, sigmaSpace)
}

// DetailEnhance increases local contrast and texture while preserving edges. It
// forms an edge-preserving base layer with [EdgePreservingFilter], takes the
// detail layer as img minus the base, and adds an amplified copy of the detail
// back to the base. sigmaS and sigmaR are passed through to the base filter.
func DetailEnhance(img *cv.Mat, sigmaS, sigmaR float64) *cv.Mat {
	if img == nil || img.Empty() {
		panic("photo: DetailEnhance given an empty image")
	}
	base := EdgePreservingFilter(img, RecursFilter, sigmaS, sigmaR)
	const amp = 3.0
	out := cv.NewMat(img.Rows, img.Cols, img.Channels)
	for y := 0; y < img.Rows; y++ {
		for x := 0; x < img.Cols; x++ {
			for c := 0; c < img.Channels; c++ {
				b := float64(base.At(y, x, c))
				o := float64(img.At(y, x, c))
				out.Set(y, x, c, clampU8(b+amp*(o-b)))
			}
		}
	}
	return out
}

// Stylization gives img a smoothed, cartoon-like look: it edge-preservingly
// smooths the image and then darkens it in proportion to the local gradient
// magnitude, so region interiors flatten while edges are outlined. sigmaS and
// sigmaR control the underlying [EdgePreservingFilter]. Input may be single- or
// three-channel; the output has the same channel count.
func Stylization(img *cv.Mat, sigmaS, sigmaR float64) *cv.Mat {
	if img == nil || img.Empty() {
		panic("photo: Stylization given an empty image")
	}
	smooth := EdgePreservingFilter(img, RecursFilter, sigmaS, sigmaR)
	mag := gradientMagnitude(grayOf(smooth))
	const k = 40.0 // gradient scale: larger => edges outlined more softly
	out := cv.NewMat(img.Rows, img.Cols, img.Channels)
	for y := 0; y < img.Rows; y++ {
		for x := 0; x < img.Cols; x++ {
			w := math.Exp(-mag[y*img.Cols+x] / k)
			for c := 0; c < img.Channels; c++ {
				out.Set(y, x, c, clampU8(w*float64(smooth.At(y, x, c))))
			}
		}
	}
	return out
}
