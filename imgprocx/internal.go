package imgprocx

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// pixelSum returns the intensity of pixel (x, y) of img, defined as the sum of
// its channel samples. For a single-channel image this is simply the sample
// value; for a colour image it is the sum over channels. It is the intensity
// used by [IntegralImage].
func pixelSum(img *cv.Mat, y, x int) float64 {
	var s float64
	for c := 0; c < img.Channels; c++ {
		s += float64(img.At(y, x, c))
	}
	return s
}

// grayValue returns the luminance of pixel (x, y): the sample itself for a
// single-channel image, otherwise the Rec. 601 weighted sum of the first three
// channels. It underlies the gradient-based routines (phase correlation and
// sub-pixel corner refinement) that need a scalar image.
func grayValue(img *cv.Mat, y, x int) float64 {
	if img.Channels == 1 {
		return float64(img.At(y, x, 0))
	}
	r := float64(img.At(y, x, 0))
	g := float64(img.At(y, x, 1))
	b := float64(img.At(y, x, 2))
	return 0.299*r + 0.587*g + 0.114*b
}

// toGrayPlane converts img to a single-channel float plane of luminance values
// stored row-major, returning it with the image dimensions.
func toGrayPlane(img *cv.Mat) (data []float64, rows, cols int) {
	rows, cols = img.Rows, img.Cols
	data = make([]float64, rows*cols)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			data[y*cols+x] = grayValue(img, y, x)
		}
	}
	return data, rows, cols
}

// bilinearMat samples channel c of src at the fractional coordinate (fx, fy)
// using bilinear interpolation, treating samples outside the image as zero.
func bilinearMat(src *cv.Mat, fx, fy float64, c int) float64 {
	x0 := int(math.Floor(fx))
	y0 := int(math.Floor(fy))
	dx := fx - float64(x0)
	dy := fy - float64(y0)
	v00 := sampleZero(src, y0, x0, c)
	v01 := sampleZero(src, y0, x0+1, c)
	v10 := sampleZero(src, y0+1, x0, c)
	v11 := sampleZero(src, y0+1, x0+1, c)
	top := v00*(1-dx) + v01*dx
	bot := v10*(1-dx) + v11*dx
	return top*(1-dy) + bot*dy
}

// sampleZero returns channel c of src at integer (y, x), or 0 when the
// coordinate lies outside the image.
func sampleZero(src *cv.Mat, y, x, c int) float64 {
	if y < 0 || y >= src.Rows || x < 0 || x >= src.Cols {
		return 0
	}
	return float64(src.At(y, x, c))
}

// clampUint8 rounds v to the nearest integer and clamps it to [0,255].
func clampUint8(v float64) uint8 {
	v = math.Round(v)
	if v < 0 {
		return 0
	}
	if v > 255 {
		return 255
	}
	return uint8(v)
}
