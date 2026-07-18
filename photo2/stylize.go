package photo2

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// DetailEnhance sharpens the fine texture of an image while preserving its
// large-scale structure and edges. The image is smoothed with the recursive
// edge-preserving filter to obtain a base layer; the detail (input minus base)
// is amplified by factor and added back. sigmaS and sigmaR control the
// edge-preserving smoothing; factor > 1 boosts detail (a typical value is 3).
func DetailEnhance(img *cv.Mat, sigmaS, sigmaR, factor float64) *cv.Mat {
	photo2RequireImage(img, "DetailEnhance")
	if factor <= 0 {
		factor = 1
	}
	base := EdgePreservingFilter(img, sigmaS, sigmaR)
	out := cv.NewMat(img.Rows, img.Cols, img.Channels)
	for i := range img.Data {
		b := float64(base.Data[i])
		detail := float64(img.Data[i]) - b
		out.Data[i] = photo2Clamp8(b + factor*detail)
	}
	return out
}

// Stylization produces a watercolour-like rendering: an edge-preserving filter
// flattens colour into smooth regions while edges are retained, giving a
// non-photorealistic, cartoon-adjacent look. sigmaS and sigmaR control the
// smoothing (larger values flatten more). The input must be three-channel.
func Stylization(img *cv.Mat, sigmaS, sigmaR float64) *cv.Mat {
	photo2RequireRGB(img, "Stylization")
	smooth := EdgePreservingFilter(img, sigmaS, sigmaR)
	// Attenuate colour toward regions of low gradient magnitude, emphasising
	// the flat painterly areas (edges are darkened by the mask).
	gray := Grayscale(smooth)
	mag := photo2GradientMagnitude(gray)
	out := cv.NewMat(img.Rows, img.Cols, 3)
	total := img.Rows * img.Cols
	for i := 0; i < total; i++ {
		e := 1 - photo2Clamp01(mag.Data[i]/64)
		for c := 0; c < 3; c++ {
			out.Data[i*3+c] = photo2Clamp8(float64(smooth.Data[i*3+c]) * e)
		}
	}
	return out
}

// PencilSketch renders an image as a pencil drawing. It returns a grayscale
// sketch and a colour-pencil version. The technique divides the grayscale image
// by a blurred, inverted copy of itself (colour dodge), which traces edges as
// dark strokes on a light ground. sigmaS sets the blur radius, sigmaR is
// accepted for API symmetry, and shadeFactor (typically 0.02–0.1) scales the
// overall darkness. The input must be three-channel.
func PencilSketch(img *cv.Mat, sigmaS, sigmaR, shadeFactor float64) (gray *cv.Mat, color *cv.Mat) {
	photo2RequireRGB(img, "PencilSketch")
	if sigmaS <= 0 {
		sigmaS = 1
	}
	if shadeFactor <= 0 {
		shadeFactor = 0.02
	}
	g := Luminance(img) // [0,1]
	blurInv := cv.NewFloatMat(g.Rows, g.Cols)
	blur := GaussianBlurFloat(g, sigmaS)
	for i := range blurInv.Data {
		blurInv.Data[i] = 1 - blur.Data[i]
	}
	sketch := cv.NewFloatMat(g.Rows, g.Cols)
	for i := range sketch.Data {
		denom := blurInv.Data[i]
		if denom < 1e-6 {
			denom = 1e-6
		}
		v := g.Data[i] / denom
		// Apply the shading factor as a mild darkening of the bright ground.
		v *= (1 - shadeFactor)
		sketch.Data[i] = photo2Clamp01(v)
	}
	gray = cv.NewMat(g.Rows, g.Cols, 1)
	for i := range sketch.Data {
		gray.Data[i] = photo2Clamp8(sketch.Data[i] * 255)
	}
	// Colour version: modulate the original colour by the sketch intensity.
	color = cv.NewMat(img.Rows, img.Cols, 3)
	for i := 0; i < g.Rows*g.Cols; i++ {
		s := sketch.Data[i]
		for c := 0; c < 3; c++ {
			color.Data[i*3+c] = photo2Clamp8(float64(img.Data[i*3+c]) * s)
		}
	}
	return gray, color
}

// Cartoon renders an image in a cartoon style by flattening colour with an
// edge-preserving filter and overlaying bold black outlines detected from the
// gradient magnitude. sigmaS and sigmaR control the colour flattening. The input
// must be three-channel.
func Cartoon(img *cv.Mat, sigmaS, sigmaR float64) *cv.Mat {
	photo2RequireRGB(img, "Cartoon")
	flat := EdgePreservingFilter(img, sigmaS, sigmaR)
	gray := Grayscale(img)
	mag := photo2GradientMagnitude(gray)
	out := cv.NewMat(img.Rows, img.Cols, 3)
	total := img.Rows * img.Cols
	for i := 0; i < total; i++ {
		edge := mag.Data[i] > 48 // strong gradient -> outline
		for c := 0; c < 3; c++ {
			if edge {
				out.Data[i*3+c] = 0
			} else {
				out.Data[i*3+c] = flat.Data[i*3+c]
			}
		}
	}
	return out
}

// photo2GradientMagnitude returns the Sobel gradient magnitude of a
// single-channel 8-bit image as a float plane, with reflected borders.
func photo2GradientMagnitude(gray *cv.Mat) *cv.FloatMat {
	rows, cols := gray.Rows, gray.Cols
	at := func(y, x int) float64 {
		return float64(gray.Data[photo2Reflect(y, rows)*cols+photo2Reflect(x, cols)])
	}
	out := cv.NewFloatMat(rows, cols)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			gx := (at(y-1, x+1) + 2*at(y, x+1) + at(y+1, x+1)) -
				(at(y-1, x-1) + 2*at(y, x-1) + at(y+1, x-1))
			gy := (at(y+1, x-1) + 2*at(y+1, x) + at(y+1, x+1)) -
				(at(y-1, x-1) + 2*at(y-1, x) + at(y-1, x+1))
			out.Data[y*cols+x] = math.Hypot(gx, gy)
		}
	}
	return out
}
