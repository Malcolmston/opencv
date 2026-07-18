package superres

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// superresCheckSame panics unless a and b have identical dimensions and
// channel counts.
func superresCheckSame(a, b *cv.Mat) {
	if a.Rows != b.Rows || a.Cols != b.Cols || a.Channels != b.Channels {
		panic("superres: images must have identical dimensions and channels")
	}
}

// MSE returns the mean squared error between two images of identical shape,
// averaged over all samples (rows·cols·channels). It panics if the shapes
// differ.
func MSE(a, b *cv.Mat) float64 {
	superresCheckSame(a, b)
	var sum float64
	for i := range a.Data {
		d := float64(a.Data[i]) - float64(b.Data[i])
		sum += d * d
	}
	return sum / float64(len(a.Data))
}

// MAE returns the mean absolute error between two images of identical shape,
// averaged over all samples. It panics if the shapes differ.
func MAE(a, b *cv.Mat) float64 {
	superresCheckSame(a, b)
	var sum float64
	for i := range a.Data {
		sum += math.Abs(float64(a.Data[i]) - float64(b.Data[i]))
	}
	return sum / float64(len(a.Data))
}

// PSNR returns the peak signal-to-noise ratio in decibels between two 8-bit
// images of identical shape (peak value 255). Identical images have no error,
// for which PSNR returns math.Inf(1). It panics if the shapes differ.
func PSNR(a, b *cv.Mat) float64 {
	mse := MSE(a, b)
	if mse == 0 {
		return math.Inf(1)
	}
	return 10 * math.Log10(255*255/mse)
}

// SSIM returns the mean structural similarity index between two grayscale or
// multi-channel images of identical shape. The index is computed over
// non-overlapping window×window blocks (channels handled independently and
// averaged) using the standard constants C1=(0.01·255)² and C2=(0.03·255)².
// The result lies in [-1, 1]; identical images score 1. window must be
// positive; it is clamped to the image size. It panics if the shapes differ.
func SSIM(a, b *cv.Mat, window int) float64 {
	superresCheckSame(a, b)
	if window <= 0 {
		panic("superres: SSIM requires a positive window")
	}
	if window > a.Rows {
		window = a.Rows
	}
	if window > a.Cols {
		window = a.Cols
	}
	const c1 = (0.01 * 255) * (0.01 * 255)
	const c2 = (0.03 * 255) * (0.03 * 255)
	ch := a.Channels
	var total float64
	var count int
	for by := 0; by+window <= a.Rows; by += window {
		for bx := 0; bx+window <= a.Cols; bx += window {
			for c := 0; c < ch; c++ {
				var sa, sb, saa, sbb, sab float64
				n := float64(window * window)
				for y := by; y < by+window; y++ {
					for x := bx; x < bx+window; x++ {
						va := float64(a.Data[(y*a.Cols+x)*ch+c])
						vb := float64(b.Data[(y*b.Cols+x)*ch+c])
						sa += va
						sb += vb
						saa += va * va
						sbb += vb * vb
						sab += va * vb
					}
				}
				ma := sa / n
				mb := sb / n
				va := saa/n - ma*ma
				vb := sbb/n - mb*mb
				cov := sab/n - ma*mb
				s := ((2*ma*mb + c1) * (2*cov + c2)) / ((ma*ma + mb*mb + c1) * (va + vb + c2))
				total += s
				count++
			}
		}
	}
	if count == 0 {
		return 1
	}
	return total / float64(count)
}
