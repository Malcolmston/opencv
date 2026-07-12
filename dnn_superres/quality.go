package dnn_superres

import (
	"fmt"
	"math"

	cv "github.com/malcolmston/opencv"
)

// gaussian1D returns a normalised 1-D Gaussian kernel of the given odd size and
// sigma, used to build the separable window for [SSIM].
func gaussian1D(size int, sigma float64) []float64 {
	k := make([]float64, size)
	c := (size - 1) / 2
	var sum float64
	for i := 0; i < size; i++ {
		d := float64(i - c)
		v := math.Exp(-(d * d) / (2 * sigma * sigma))
		k[i] = v
		sum += v
	}
	for i := range k {
		k[i] /= sum
	}
	return k
}

// blurPlaneFloat convolves a single-channel float plane with the separable
// Gaussian kernel k (border replication), returning a new plane of the same
// size. It keeps full precision so the products needed by SSIM (a·a, a·b) are
// not truncated to bytes.
func blurPlaneFloat(src []float64, rows, cols int, k []float64) []float64 {
	rad := (len(k) - 1) / 2
	tmp := make([]float64, rows*cols)
	// Horizontal.
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			var acc float64
			for t := -rad; t <= rad; t++ {
				xx := clampInt(x+t, 0, cols-1)
				acc += k[t+rad] * src[y*cols+xx]
			}
			tmp[y*cols+x] = acc
		}
	}
	// Vertical.
	out := make([]float64, rows*cols)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			var acc float64
			for t := -rad; t <= rad; t++ {
				yy := clampInt(y+t, 0, rows-1)
				acc += k[t+rad] * tmp[yy*cols+x]
			}
			out[y*cols+x] = acc
		}
	}
	return out
}

// SSIM returns the Mean Structural Similarity Index between two images of
// identical shape, the perceptual companion to [PSNR]. It compares local
// luminance, contrast and structure inside an 11×11 Gaussian window (sigma 1.5)
// with the standard stabilising constants C1 = (0.01·255)² and C2 = (0.03·255)²,
// averaging the per-window scores over the image and then over the channels.
//
// The result lies in [-1, 1]; 1.0 means the images are identical, and higher is
// better. It returns an error if either image is empty or their dimensions or
// channel counts differ.
func SSIM(a, b *cv.Mat) (float64, error) {
	if a == nil || b == nil || a.Empty() || b.Empty() {
		return 0, fmt.Errorf("dnn_superres: SSIM given an empty image")
	}
	if a.Rows != b.Rows || a.Cols != b.Cols || a.Channels != b.Channels {
		return 0, fmt.Errorf("dnn_superres: SSIM shape mismatch %dx%dx%d vs %dx%dx%d",
			a.Rows, a.Cols, a.Channels, b.Rows, b.Cols, b.Channels)
	}
	const c1 = (0.01 * 255) * (0.01 * 255)
	const c2 = (0.03 * 255) * (0.03 * 255)
	size := 11
	if a.Rows < size || a.Cols < size {
		// Shrink the window to fit tiny images (must stay odd and >= 3).
		m := a.Rows
		if a.Cols < m {
			m = a.Cols
		}
		if m%2 == 0 {
			m--
		}
		if m < 3 {
			m = 3
		}
		size = m
	}
	k := gaussian1D(size, 1.5)
	rows, cols, ch := a.Rows, a.Cols, a.Channels
	n := rows * cols
	var total float64
	for c := 0; c < ch; c++ {
		fa := make([]float64, n)
		fb := make([]float64, n)
		faa := make([]float64, n)
		fbb := make([]float64, n)
		fab := make([]float64, n)
		for i := 0; i < n; i++ {
			av := float64(a.Data[i*ch+c])
			bv := float64(b.Data[i*ch+c])
			fa[i], fb[i] = av, bv
			faa[i], fbb[i], fab[i] = av*av, bv*bv, av*bv
		}
		muA := blurPlaneFloat(fa, rows, cols, k)
		muB := blurPlaneFloat(fb, rows, cols, k)
		mAA := blurPlaneFloat(faa, rows, cols, k)
		mBB := blurPlaneFloat(fbb, rows, cols, k)
		mAB := blurPlaneFloat(fab, rows, cols, k)
		var acc float64
		for i := 0; i < n; i++ {
			ma, mb := muA[i], muB[i]
			va := mAA[i] - ma*ma
			vb := mBB[i] - mb*mb
			cov := mAB[i] - ma*mb
			num := (2*ma*mb + c1) * (2*cov + c2)
			den := (ma*ma + mb*mb + c1) * (va + vb + c2)
			acc += num / den
		}
		total += acc / float64(n)
	}
	return total / float64(ch), nil
}
