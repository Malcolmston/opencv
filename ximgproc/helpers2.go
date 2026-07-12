package ximgproc

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// channelPlane extracts channel c of img as a row-major float slice.
func channelPlane(img *cv.Mat, c int) []float64 {
	n := img.Total()
	out := make([]float64, n)
	ch := img.Channels
	for i := 0; i < n; i++ {
		out[i] = float64(img.Data[i*ch+c])
	}
	return out
}

// planesFromMat splits img into one float plane per channel.
func planesFromMat(img *cv.Mat) [][]float64 {
	planes := make([][]float64, img.Channels)
	for c := range planes {
		planes[c] = channelPlane(img, c)
	}
	return planes
}

// matFromPlanes reassembles clamped 8-bit channels from float planes.
func matFromPlanes(planes [][]float64, rows, cols int) *cv.Mat {
	ch := len(planes)
	out := cv.NewMat(rows, cols, ch)
	for c := 0; c < ch; c++ {
		p := planes[c]
		for i := 0; i < rows*cols; i++ {
			out.Data[i*ch+c] = clampU8(p[i])
		}
	}
	return out
}

// gaussKernel1D returns a normalised 1-D Gaussian kernel of the given sigma,
// truncated at three standard deviations (radius = ceil(3·sigma), min 1).
func gaussKernel1D(sigma float64) []float64 {
	if sigma < 0.01 {
		sigma = 0.01
	}
	r := int(math.Ceil(3 * sigma))
	if r < 1 {
		r = 1
	}
	k := make([]float64, 2*r+1)
	inv := 1.0 / (2 * sigma * sigma)
	var sum float64
	for i := -r; i <= r; i++ {
		v := math.Exp(-float64(i*i) * inv)
		k[i+r] = v
		sum += v
	}
	for i := range k {
		k[i] /= sum
	}
	return k
}

// reflect maps an out-of-range index into [0,n) using reflect-101 border
// handling (…2,1,|0,1,2,…,n-1|,n-2,n-3,…), matching cv.BORDER_REFLECT_101.
func reflect(i, n int) int {
	if n == 1 {
		return 0
	}
	for i < 0 || i >= n {
		if i < 0 {
			i = -i
		}
		if i >= n {
			i = 2*(n-1) - i
		}
	}
	return i
}

// gaussianBlurFloat convolves a row-major rows×cols plane with a separable
// Gaussian of the given sigma, using reflect-101 borders. It never mutates src.
func gaussianBlurFloat(src []float64, rows, cols int, sigma float64) []float64 {
	k := gaussKernel1D(sigma)
	r := len(k) / 2
	tmp := make([]float64, rows*cols)
	// Horizontal pass.
	for y := 0; y < rows; y++ {
		base := y * cols
		for x := 0; x < cols; x++ {
			var acc float64
			for t := -r; t <= r; t++ {
				acc += k[t+r] * src[base+reflect(x+t, cols)]
			}
			tmp[base+x] = acc
		}
	}
	// Vertical pass.
	out := make([]float64, rows*cols)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			var acc float64
			for t := -r; t <= r; t++ {
				acc += k[t+r] * tmp[reflect(y+t, rows)*cols+x]
			}
			out[y*cols+x] = acc
		}
	}
	return out
}
