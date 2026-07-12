package dnn_superres

import (
	"fmt"
	"math"

	cv "github.com/malcolmston/opencv"
)

// PSNR returns the Peak Signal-to-Noise Ratio in decibels between two images of
// identical shape, the standard reconstruction-quality metric for
// super-resolution. It is computed over all channels as
//
//	PSNR = 10 * log10(MAX^2 / MSE), MAX = 255
//
// Higher is better; typical upscaling results land in the 25–40 dB range. When
// the images are byte-for-byte identical the MSE is zero and PSNR reports
// +Inf. It returns an error if either image is empty or their dimensions or
// channel counts differ.
func PSNR(a, b *cv.Mat) (float64, error) {
	if a == nil || b == nil || a.Empty() || b.Empty() {
		return 0, fmt.Errorf("dnn_superres: PSNR given an empty image")
	}
	if a.Rows != b.Rows || a.Cols != b.Cols || a.Channels != b.Channels {
		return 0, fmt.Errorf("dnn_superres: PSNR shape mismatch %dx%dx%d vs %dx%dx%d",
			a.Rows, a.Cols, a.Channels, b.Rows, b.Cols, b.Channels)
	}
	mse, err := MSE(a, b)
	if err != nil {
		return 0, err
	}
	if mse == 0 {
		return math.Inf(1), nil
	}
	return 10 * math.Log10(255*255/mse), nil
}

// MSE returns the mean squared error between two images of identical shape,
// averaged over every sample of every channel. It returns an error if either
// image is empty or their shapes differ.
func MSE(a, b *cv.Mat) (float64, error) {
	if a == nil || b == nil || a.Empty() || b.Empty() {
		return 0, fmt.Errorf("dnn_superres: MSE given an empty image")
	}
	if a.Rows != b.Rows || a.Cols != b.Cols || a.Channels != b.Channels {
		return 0, fmt.Errorf("dnn_superres: MSE shape mismatch %dx%dx%d vs %dx%dx%d",
			a.Rows, a.Cols, a.Channels, b.Rows, b.Cols, b.Channels)
	}
	var acc float64
	for i := range a.Data {
		d := float64(a.Data[i]) - float64(b.Data[i])
		acc += d * d
	}
	return acc / float64(len(a.Data)), nil
}
