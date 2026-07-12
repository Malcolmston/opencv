package intensity

import (
	"fmt"
	"math"

	cv "github.com/malcolmston/opencv"
)

// LogAdaptiveTonemap applies the adaptive logarithmic tone-mapping operator of
// Drago et al. (2003), "Adaptive Logarithmic Mapping For Displaying High
// Contrast Scenes". Working on luminance normalised by the image maximum
// Lw_max, each pixel is mapped by
//
//	Ld = log(Lw + 1) / ( log10(Lw_max + 1) · log(2 + 8·(Lw/Lw_max)^p) ),
//	p  = log(bias) / log(0.5),
//
// a logarithm whose base is interpolated by the bias parameter: the map lifts
// shadows and gently compresses highlights, so a dark scene is brightened while
// bright detail is retained rather than clipped. bias in (0,1] controls that
// interpolation — the paper's default is 0.85; smaller values darken and add
// contrast, larger values brighten. It panics unless 0 < bias ≤ 1.
//
// For a colour image the per-pixel luminance gain Ld/Lw is applied to every
// channel, preserving chromatic ratios; a single-channel image is mapped
// directly. A fully black image is returned unchanged. The output is
// deterministic.
func LogAdaptiveTonemap(img *cv.Mat, bias float64) *cv.Mat {
	requireImage(img, "LogAdaptiveTonemap")
	if !(bias > 0 && bias <= 1) {
		panic(fmt.Sprintf("intensity: LogAdaptiveTonemap requires 0 < bias <= 1, got %v", bias))
	}
	rows, cols, ch := img.Rows, img.Cols, img.Channels
	luma := lumaFloat(img)

	// Maximum luminance normalised to [0,1].
	var maxL float64
	for _, l := range luma {
		if l > maxL {
			maxL = l
		}
	}
	if maxL <= 0 {
		return img.Clone()
	}
	lwMax := maxL / 255
	p := math.Log(bias) / math.Log(0.5)
	denomScale := math.Log10(lwMax + 1)

	dst := cv.NewMat(rows, cols, ch)
	n := img.Total()
	for i := 0; i < n; i++ {
		lw := luma[i] / 255
		if lw <= 0 {
			continue // stays black
		}
		ratioL := lw / lwMax
		ld := math.Log(lw+1) / (denomScale * math.Log(2+8*math.Pow(ratioL, p)))
		gain := ld / lw // maps luminance lw -> ld
		base := i * ch
		for c := 0; c < ch; c++ {
			dst.Data[base+c] = clampToUint8(float64(img.Data[base+c])*gain + 0.5)
		}
	}
	return dst
}
