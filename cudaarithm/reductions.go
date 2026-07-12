package cudaarithm

import "math"

// Sum returns the per-channel sum of all samples, mirroring cv::cuda::sum. The
// result has one entry per channel.
func Sum(src *GpuMat, _ ...*Stream) []float64 {
	requireNonEmpty(src, "Sum")
	return channelReduce(src, func(acc, v float64) float64 { return acc + v })
}

// AbsSum returns the per-channel sum of absolute sample values, mirroring
// cv::cuda::absSum. Because [cv.Mat] samples are unsigned this equals [Sum], but
// the function is provided for API parity.
func AbsSum(src *GpuMat, _ ...*Stream) []float64 {
	requireNonEmpty(src, "AbsSum")
	return channelReduce(src, func(acc, v float64) float64 { return acc + math.Abs(v) })
}

// SqrSum returns the per-channel sum of squared sample values, mirroring
// cv::cuda::sqrSum.
func SqrSum(src *GpuMat, _ ...*Stream) []float64 {
	requireNonEmpty(src, "SqrSum")
	return channelReduce(src, func(acc, v float64) float64 { return acc + v*v })
}

// channelReduce folds every sample of each channel with f, seeded at 0.
func channelReduce(src *GpuMat, f func(acc, v float64) float64) []float64 {
	ch := src.mat.Channels
	acc := make([]float64, ch)
	for i, s := range src.mat.Data {
		c := i % ch
		acc[c] = f(acc[c], float64(s))
	}
	return acc
}

// Mean returns the per-channel arithmetic mean of the samples, mirroring
// cv::cuda::mean.
func Mean(src *GpuMat, _ ...*Stream) []float64 {
	requireNonEmpty(src, "Mean")
	sums := Sum(src)
	n := float64(src.mat.Total())
	for c := range sums {
		sums[c] /= n
	}
	return sums
}

// MeanStdDev returns the per-channel mean and population standard deviation of
// the samples, mirroring cv::cuda::meanStdDev.
func MeanStdDev(src *GpuMat, _ ...*Stream) (mean, stddev []float64) {
	requireNonEmpty(src, "MeanStdDev")
	ch := src.mat.Channels
	n := float64(src.mat.Total())
	mean = Mean(src)
	stddev = make([]float64, ch)
	for i, s := range src.mat.Data {
		c := i % ch
		d := float64(s) - mean[c]
		stddev[c] += d * d
	}
	for c := range stddev {
		stddev[c] = math.Sqrt(stddev[c] / n)
	}
	return mean, stddev
}

// MinMax returns the minimum and maximum sample value of a single-channel
// GpuMat, mirroring cv::cuda::minMax. It panics unless src is single-channel.
func MinMax(src *GpuMat, _ ...*Stream) (minVal, maxVal float64) {
	requireChannels(src, 1, "MinMax")
	lo, hi := src.mat.Data[0], src.mat.Data[0]
	for _, s := range src.mat.Data {
		if s < lo {
			lo = s
		}
		if s > hi {
			hi = s
		}
	}
	return float64(lo), float64(hi)
}

// MinMaxLoc returns the minimum and maximum sample values of a single-channel
// GpuMat together with the (x, y) location of the first occurrence of each,
// mirroring cv::cuda::minMaxLoc. It panics unless src is single-channel.
func MinMaxLoc(src *GpuMat, _ ...*Stream) (minVal, maxVal float64, minX, minY, maxX, maxY int) {
	requireChannels(src, 1, "MinMaxLoc")
	cols := src.mat.Cols
	lo, hi := src.mat.Data[0], src.mat.Data[0]
	for i, s := range src.mat.Data {
		if s < lo {
			lo = s
			minX, minY = i%cols, i/cols
		}
		if s > hi {
			hi = s
			maxX, maxY = i%cols, i/cols
		}
	}
	return float64(lo), float64(hi), minX, minY, maxX, maxY
}

// CountNonZero returns the number of samples that are not zero, mirroring
// cv::cuda::countNonZero. All channels are considered.
func CountNonZero(src *GpuMat, _ ...*Stream) int {
	requireNonEmpty(src, "CountNonZero")
	n := 0
	for _, s := range src.mat.Data {
		if s != 0 {
			n++
		}
	}
	return n
}
