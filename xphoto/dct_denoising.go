package xphoto

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// DctDenoisingDefaultPatch is the default sliding-window edge length used by
// [DctDenoising] when psize <= 0. It matches OpenCV's default of 16.
const DctDenoisingDefaultPatch = 16

// DctDenoising denoises src with sliding-window DCT hard-thresholding, porting
// cv::xphoto::dctDenoising. Every psize x psize window (stepped by one pixel, so
// windows fully overlap) is transformed with a separable 2D-DCT; coefficients
// whose magnitude is below 3*sigma are zeroed while the DC term is preserved;
// the window is inverse-transformed and accumulated into an output buffer. Each
// output pixel is the average of every window that covered it, which is the
// aggregation that gives DCT denoising its edge-preserving, low-artefact
// behaviour.
//
// sigma is the assumed noise standard deviation; larger sigma removes more
// noise. psize is the window size (psize <= 0 defaults to
// [DctDenoisingDefaultPatch]); it is clamped so it never exceeds the image
// dimensions. src may be single- or three-channel and each channel is denoised
// independently. The input is not modified.
func DctDenoising(src *cv.Mat, sigma float64, psize int) *cv.Mat {
	requireNonEmpty(src, "DctDenoising")
	if sigma <= 0 {
		sigma = 1
	}
	if psize <= 0 {
		psize = DctDenoisingDefaultPatch
	}
	rows, cols := src.Rows, src.Cols
	if psize > rows {
		psize = rows
	}
	if psize > cols {
		psize = cols
	}
	dst := cv.NewMat(rows, cols, src.Channels)
	cos := dctBasis(psize)
	thr := 3.0 * sigma
	for c := 0; c < src.Channels; c++ {
		dctDenoiseChannel(src, dst, c, psize, cos, thr)
	}
	return dst
}

// dctDenoiseChannel runs sliding-window DCT hard-thresholding on channel c.
func dctDenoiseChannel(src, dst *cv.Mat, c, P int, cos [][]float64, thr float64) {
	rows, cols := src.Rows, src.Cols
	plane := make([]float64, rows*cols)
	for i := 0; i < rows*cols; i++ {
		plane[i] = float64(src.Data[i*src.Channels+c])
	}
	num := make([]float64, rows*cols)
	den := make([]float64, rows*cols)

	block := make([]float64, P*P)
	for wy := 0; wy <= rows-P; wy++ {
		for wx := 0; wx <= cols-P; wx++ {
			for i := 0; i < P; i++ {
				row := (wy + i) * cols
				for j := 0; j < P; j++ {
					block[i*P+j] = plane[row+wx+j]
				}
			}
			coef := dct2d(block, P, cos)
			for k := 1; k < P*P; k++ { // k == 0 is the DC term, always kept
				if math.Abs(coef[k]) < thr {
					coef[k] = 0
				}
			}
			rec := idct2d(coef, P, cos)
			for i := 0; i < P; i++ {
				row := (wy + i) * cols
				for j := 0; j < P; j++ {
					idx := row + wx + j
					num[idx] += rec[i*P+j]
					den[idx]++
				}
			}
		}
	}
	for i := 0; i < rows*cols; i++ {
		v := plane[i]
		if den[i] > 0 {
			v = num[i] / den[i]
		}
		dst.Data[i*dst.Channels+c] = clampU8(v)
	}
}
