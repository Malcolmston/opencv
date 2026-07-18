package filters2

import (
	cv "github.com/malcolmston/opencv"
)

// gaussianBlurChannel returns a Gaussian-blurred copy of channel c of src as a
// float slice, using a separable kernel and edge replication.
func gaussianBlurChannel(src *cv.Mat, c int, sigma float64) []float64 {
	radius := gaussianRadius(sigma)
	k := GaussianKernel1D(radius, sigma)
	rows, cols := src.Rows, src.Cols
	tmp := make([]float64, rows*cols)
	// Horizontal pass.
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			var s float64
			for i := -radius; i <= radius; i++ {
				s += k[i+radius] * float64(atReplicate(src, y, x+i, c))
			}
			tmp[y*cols+x] = s
		}
	}
	// Vertical pass.
	out := make([]float64, rows*cols)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			var s float64
			for i := -radius; i <= radius; i++ {
				s += k[i+radius] * tmp[clampIdx(y+i, rows)*cols+x]
			}
			out[y*cols+x] = s
		}
	}
	return out
}

// GaussianBlur returns a Gaussian-blurred copy of src using a separable kernel
// of standard deviation sigma (the truncation radius is chosen from sigma) and
// edge replication. Each channel is processed independently. It is the low-pass
// building block shared by the sharpening filters, exposed for convenience. It
// panics on empty input or a non-positive sigma.
func GaussianBlur(src *cv.Mat, sigma float64) *cv.Mat {
	requireNonEmpty(src, "GaussianBlur")
	if sigma <= 0 {
		panic("filters2: GaussianBlur requires a positive sigma")
	}
	rows, cols, ch := src.Rows, src.Cols, src.Channels
	dst := like(src)
	for c := 0; c < ch; c++ {
		blur := gaussianBlurChannel(src, c, sigma)
		for i := 0; i < rows*cols; i++ {
			dst.Data[i*ch+c] = clampU8(blur[i])
		}
	}
	return dst
}

// UnsharpMask sharpens src by adding a fraction of the difference between the
// image and a Gaussian-blurred version of it: out = src + amount*(src-blur).
// sigma sets the blur radius, amount the sharpening strength, and only
// differences whose magnitude exceeds threshold (in intensity units) are
// applied, which avoids amplifying noise in flat regions. Each channel is
// processed independently. It panics on empty input or a non-positive sigma.
func UnsharpMask(src *cv.Mat, sigma, amount, threshold float64) *cv.Mat {
	requireNonEmpty(src, "UnsharpMask")
	if sigma <= 0 {
		panic("filters2: UnsharpMask requires a positive sigma")
	}
	rows, cols, ch := src.Rows, src.Cols, src.Channels
	dst := like(src)
	for c := 0; c < ch; c++ {
		blur := gaussianBlurChannel(src, c, sigma)
		for i := 0; i < rows*cols; i++ {
			orig := float64(src.Data[i*ch+c])
			diff := orig - blur[i]
			if diff < 0 {
				if -diff < threshold {
					diff = 0
				}
			} else if diff < threshold {
				diff = 0
			}
			dst.Data[i*ch+c] = clampU8(orig + amount*diff)
		}
	}
	return dst
}

// HighBoostFilter applies high-boost (high-frequency-emphasis) filtering:
// out = boost*src - blur, where blur is a Gaussian low-pass of standard
// deviation sigma. With boost == 1 the result is the pure high-pass residual;
// with boost > 1 the original image is progressively re-added, sharpening it
// while retaining overall brightness. Each channel is processed independently.
// It panics on empty input or a non-positive sigma.
func HighBoostFilter(src *cv.Mat, sigma, boost float64) *cv.Mat {
	requireNonEmpty(src, "HighBoostFilter")
	if sigma <= 0 {
		panic("filters2: HighBoostFilter requires a positive sigma")
	}
	rows, cols, ch := src.Rows, src.Cols, src.Channels
	dst := like(src)
	for c := 0; c < ch; c++ {
		blur := gaussianBlurChannel(src, c, sigma)
		for i := 0; i < rows*cols; i++ {
			orig := float64(src.Data[i*ch+c])
			dst.Data[i*ch+c] = clampU8(boost*orig - blur[i])
		}
	}
	return dst
}

// LaplacianSharpen sharpens src by subtracting a scaled discrete Laplacian,
// out = src - amount*Laplacian(src), which emphasises edges and fine detail.
// The 4-neighbour Laplacian is used with edge replication and each channel is
// processed independently. It panics on empty input.
func LaplacianSharpen(src *cv.Mat, amount float64) *cv.Mat {
	requireNonEmpty(src, "LaplacianSharpen")
	rows, cols, ch := src.Rows, src.Cols, src.Channels
	dst := like(src)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			for c := 0; c < ch; c++ {
				center := float64(atReplicate(src, y, x, c))
				lap := float64(atReplicate(src, y-1, x, c)) +
					float64(atReplicate(src, y+1, x, c)) +
					float64(atReplicate(src, y, x-1, c)) +
					float64(atReplicate(src, y, x+1, c)) - 4*center
				dst.Data[(y*cols+x)*ch+c] = clampU8(center - amount*lap)
			}
		}
	}
	return dst
}
