package edges2

import (
	"math"
	"sort"

	cv "github.com/malcolmston/opencv"
)

// NonMaximumSuppression thins the gradient magnitude of f by keeping only the
// pixels that are local maxima along the gradient direction, returning the
// suppressed magnitude as a [FloatGrid]. Non-maximal and border pixels are set
// to zero. This is the third stage of the Canny pipeline.
func NonMaximumSuppression(f *GradientField) *FloatGrid {
	rows, cols := f.Rows, f.Cols
	mag := f.Magnitude()
	out := NewFloatGrid(rows, cols)
	for y := 1; y < rows-1; y++ {
		for x := 1; x < cols-1; x++ {
			i := y*cols + x
			angle := math.Atan2(f.Gy.Data[i], f.Gx.Data[i]) * 180 / math.Pi
			if angle < 0 {
				angle += 180
			}
			var a, b float64
			switch {
			case angle < 22.5 || angle >= 157.5:
				a, b = mag.Data[i-1], mag.Data[i+1]
			case angle < 67.5:
				a, b = mag.Data[i-cols+1], mag.Data[i+cols-1]
			case angle < 112.5:
				a, b = mag.Data[i-cols], mag.Data[i+cols]
			default:
				a, b = mag.Data[i-cols-1], mag.Data[i+cols+1]
			}
			if mag.Data[i] >= a && mag.Data[i] >= b {
				out.Data[i] = mag.Data[i]
			}
		}
	}
	return out
}

// DoubleThreshold classifies the samples of a (typically non-maximum
// suppressed) magnitude grid into three levels and returns them as an 8-bit
// [cv.Mat]: samples at or above highThresh become strong edges (255), samples
// in [lowThresh, highThresh) become weak edges (128) and the rest become
// background (0). The thresholds are swapped if given out of order.
func DoubleThreshold(mag *FloatGrid, lowThresh, highThresh float64) *cv.Mat {
	if lowThresh > highThresh {
		lowThresh, highThresh = highThresh, lowThresh
	}
	m := cv.NewMat(mag.Rows, mag.Cols, 1)
	for i, v := range mag.Data {
		switch {
		case v >= highThresh:
			m.Data[i] = 255
		case v >= lowThresh:
			m.Data[i] = 128
		}
	}
	return m
}

// Hysteresis performs Canny double-threshold edge tracking on a magnitude grid
// and returns a binary edge map (255 for edges, 0 otherwise). Pixels at or
// above highThresh are strong seeds; pixels in [lowThresh, highThresh) are kept
// only when 8-connected, directly or transitively, to a strong seed. The
// thresholds are swapped if given out of order.
func Hysteresis(mag *FloatGrid, lowThresh, highThresh float64) *cv.Mat {
	if lowThresh > highThresh {
		lowThresh, highThresh = highThresh, lowThresh
	}
	rows, cols := mag.Rows, mag.Cols
	const (
		weak   = 1
		strong = 2
	)
	label := make([]uint8, rows*cols)
	for i, v := range mag.Data {
		switch {
		case v >= highThresh:
			label[i] = strong
		case v >= lowThresh:
			label[i] = weak
		}
	}
	dst := cv.NewMat(rows, cols, 1)
	stack := make([]int, 0, rows*cols)
	for i, l := range label {
		if l == strong {
			dst.Data[i] = 255
			stack = append(stack, i)
		}
	}
	for len(stack) > 0 {
		i := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		y := i / cols
		x := i % cols
		for dy := -1; dy <= 1; dy++ {
			ny := y + dy
			if ny < 0 || ny >= rows {
				continue
			}
			for dx := -1; dx <= 1; dx++ {
				nx := x + dx
				if nx < 0 || nx >= cols {
					continue
				}
				ni := ny*cols + nx
				if label[ni] == weak && dst.Data[ni] == 0 {
					dst.Data[ni] = 255
					stack = append(stack, ni)
				}
			}
		}
	}
	return dst
}

// CannyField runs the last two Canny stages — non-maximum suppression and
// double-threshold hysteresis — on a precomputed [GradientField] and returns a
// binary edge map (edges 255, background 0). Use it when the gradient has
// already been computed (for example to share it with [HoughCircles]).
func CannyField(f *GradientField, lowThresh, highThresh float64) *cv.Mat {
	return Hysteresis(NonMaximumSuppression(f), lowThresh, highThresh)
}

// Canny runs the complete Canny edge-detection pipeline on a single-channel
// image and returns a binary edge map (edges 255, background 0).
//
// The stages are: Gaussian smoothing with the given sigma, 3×3 Sobel
// gradients, non-maximum suppression along the gradient direction, and
// double-threshold hysteresis that keeps weak edges only when connected to a
// strong one. lowThresh and highThresh are gradient-magnitude thresholds; they
// are swapped if given out of order. It panics on multi-channel input.
func Canny(src *cv.Mat, lowThresh, highThresh, sigma float64) *cv.Mat {
	edges2RequireGray(src, "Canny")
	if sigma <= 0 {
		sigma = 1.0
	}
	blurred := edges2Blur(src, sigma)
	return CannyField(Sobel(blurred), lowThresh, highThresh)
}

// CannyAuto runs [Canny] with thresholds derived automatically from the median
// gradient magnitude of the image: low = (1-scale)*median and
// high = (1+scale)*median, with scale a spread factor around 0.33. It is a
// convenient parameter-free entry point and panics on multi-channel input.
func CannyAuto(src *cv.Mat, scale, sigma float64) *cv.Mat {
	edges2RequireGray(src, "CannyAuto")
	if sigma <= 0 {
		sigma = 1.0
	}
	if scale <= 0 {
		scale = 0.33
	}
	blurred := edges2Blur(src, sigma)
	mag := Sobel(blurred).Magnitude()
	vals := make([]float64, len(mag.Data))
	copy(vals, mag.Data)
	sort.Float64s(vals)
	median := vals[len(vals)/2]
	low := (1 - scale) * median
	high := (1 + scale) * median
	if low < 0 {
		low = 0
	}
	return CannyField(Sobel(blurred), low, high)
}
