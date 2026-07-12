package ximgproc

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// toGray returns a single-channel float image of img on the [0,255] scale.
// A 1-channel Mat is copied directly; a 3-channel Mat is converted with the
// BT.601 luma weights via cv.CvtColor. It panics for other channel counts.
func toGrayFloat(img *cv.Mat) []float64 {
	g := toGray(img)
	out := make([]float64, g.Total())
	for i := range out {
		out[i] = float64(g.Data[i])
	}
	return out
}

// toGray returns a single-channel Mat view of img: a 1-channel input is cloned,
// a 3-channel input is converted to grayscale. It panics for other counts.
func toGray(img *cv.Mat) *cv.Mat {
	switch img.Channels {
	case 1:
		return img.Clone()
	case 3:
		return cv.CvtColor(img, cv.ColorRGB2Gray)
	default:
		panic("ximgproc: expected a 1- or 3-channel image")
	}
}

// clampU8 rounds v to the nearest integer and clamps it into [0,255].
func clampU8(v float64) uint8 {
	v = math.Round(v)
	if v <= 0 {
		return 0
	}
	if v >= 255 {
		return 255
	}
	return uint8(v)
}

// integral builds a summed-area table of data (rows×cols, row-major). The
// returned slice has (rows+1)×(cols+1) entries so that the inclusive sum over
// the rectangle [y0,y1]×[x0,x1] is
//
//	ii[(y1+1)*(cols+1)+(x1+1)] - ii[y0*(cols+1)+(x1+1)]
//	  - ii[(y1+1)*(cols+1)+x0] + ii[y0*(cols+1)+x0].
func integral(data []float64, rows, cols int) []float64 {
	w := cols + 1
	ii := make([]float64, (rows+1)*w)
	for y := 0; y < rows; y++ {
		var rowSum float64
		for x := 0; x < cols; x++ {
			rowSum += data[y*cols+x]
			ii[(y+1)*w+(x+1)] = ii[y*w+(x+1)] + rowSum
		}
	}
	return ii
}

// rectSum returns the inclusive sum over the clamped rectangle centred window
// [y0,y1]×[x0,x1] from a summed-area table produced by integral.
func rectSum(ii []float64, cols, y0, x0, y1, x1 int) float64 {
	w := cols + 1
	return ii[(y1+1)*w+(x1+1)] - ii[y0*w+(x1+1)] - ii[(y1+1)*w+x0] + ii[y0*w+x0]
}

// boxMean computes, for every pixel, the mean of data over the (2r+1)×(2r+1)
// window centred on it, with the window clamped to the image so that border
// pixels are normalised by the number of valid samples (matching OpenCV's
// guided-filter edge handling). data is row-major rows×cols.
func boxMean(data []float64, rows, cols, r int) []float64 {
	ii := integral(data, rows, cols)
	out := make([]float64, rows*cols)
	for y := 0; y < rows; y++ {
		y0 := y - r
		if y0 < 0 {
			y0 = 0
		}
		y1 := y + r
		if y1 > rows-1 {
			y1 = rows - 1
		}
		for x := 0; x < cols; x++ {
			x0 := x - r
			if x0 < 0 {
				x0 = 0
			}
			x1 := x + r
			if x1 > cols-1 {
				x1 = cols - 1
			}
			n := float64((y1 - y0 + 1) * (x1 - x0 + 1))
			out[y*cols+x] = rectSum(ii, cols, y0, x0, y1, x1) / n
		}
	}
	return out
}

// mul returns the element-wise product of two equally sized slices.
func mul(a, b []float64) []float64 {
	out := make([]float64, len(a))
	for i := range a {
		out[i] = a[i] * b[i]
	}
	return out
}
