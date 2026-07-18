package cv

import "math"

// Mean returns the per-channel average of all samples in m as a Scalar (unused
// channels are zero), mirroring cv2.mean. It panics on an empty matrix.
func Mean(m *Mat) Scalar {
	if m.Empty() {
		panic("cv: Mean on empty matrix")
	}
	var sum [4]float64
	ch := m.Channels
	if ch > 4 {
		ch = 4
	}
	n := m.Rows * m.Cols
	for i := 0; i < n; i++ {
		base := i * m.Channels
		for c := 0; c < ch; c++ {
			sum[c] += float64(m.Data[base+c])
		}
	}
	var out Scalar
	for c := 0; c < ch; c++ {
		out[c] = sum[c] / float64(n)
	}
	return out
}

// MeanStdDev returns the per-channel mean and (population) standard deviation of
// m, mirroring cv2.meanStdDev. It panics on an empty matrix.
func MeanStdDev(m *Mat) (mean, stddev Scalar) {
	if m.Empty() {
		panic("cv: MeanStdDev on empty matrix")
	}
	ch := m.Channels
	if ch > 4 {
		ch = 4
	}
	n := m.Rows * m.Cols
	var sum, sumSq [4]float64
	for i := 0; i < n; i++ {
		base := i * m.Channels
		for c := 0; c < ch; c++ {
			v := float64(m.Data[base+c])
			sum[c] += v
			sumSq[c] += v * v
		}
	}
	for c := 0; c < ch; c++ {
		mu := sum[c] / float64(n)
		mean[c] = mu
		variance := sumSq[c]/float64(n) - mu*mu
		if variance < 0 {
			variance = 0
		}
		stddev[c] = math.Sqrt(variance)
	}
	return mean, stddev
}

// SumElems returns the per-channel sum of all samples in m, mirroring
// cv2.sumElems. It panics on an empty matrix.
func SumElems(m *Mat) Scalar {
	if m.Empty() {
		panic("cv: SumElems on empty matrix")
	}
	ch := m.Channels
	if ch > 4 {
		ch = 4
	}
	var sum Scalar
	n := m.Rows * m.Cols
	for i := 0; i < n; i++ {
		base := i * m.Channels
		for c := 0; c < ch; c++ {
			sum[c] += float64(m.Data[base+c])
		}
	}
	return sum
}

// NormL1Mat returns the L1 norm (sum of absolute sample values) of m, matching
// cv2.norm with NORM_L1.
func NormL1Mat(m *Mat) float64 {
	var s float64
	for _, v := range m.Data {
		s += float64(v)
	}
	return s
}

// NormL2Mat returns the L2 norm (square root of the sum of squared samples) of
// m, matching cv2.norm with NORM_L2.
func NormL2Mat(m *Mat) float64 {
	var s float64
	for _, v := range m.Data {
		s += float64(v) * float64(v)
	}
	return math.Sqrt(s)
}

// NormInfMat returns the infinity norm (largest absolute sample value) of m,
// matching cv2.norm with NORM_INF.
func NormInfMat(m *Mat) float64 {
	var mx float64
	for _, v := range m.Data {
		if float64(v) > mx {
			mx = float64(v)
		}
	}
	return mx
}

// PSNR returns the peak signal-to-noise ratio in decibels between two equally
// sized 8-bit matrices, mirroring cv2.PSNR. Identical inputs return +Inf.
func PSNR(a, b *Mat) float64 {
	if a.Rows != b.Rows || a.Cols != b.Cols || a.Channels != b.Channels {
		panic("cv: PSNR requires matrices of equal shape")
	}
	var sse float64
	for i := range a.Data {
		d := float64(a.Data[i]) - float64(b.Data[i])
		sse += d * d
	}
	if sse == 0 {
		return math.Inf(1)
	}
	mse := sse / float64(len(a.Data))
	return 10 * math.Log10(255*255/mse)
}
