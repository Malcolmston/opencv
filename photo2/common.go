package photo2

import (
	"fmt"
	"math"

	cv "github.com/malcolmston/opencv"
)

// photo2Clamp8 rounds v to the nearest integer and clamps it into [0,255].
func photo2Clamp8(v float64) uint8 {
	v = math.Round(v)
	if v <= 0 {
		return 0
	}
	if v >= 255 {
		return 255
	}
	return uint8(v)
}

// photo2Clamp01 clamps v into the closed interval [0,1].
func photo2Clamp01(v float64) float64 {
	if v <= 0 {
		return 0
	}
	if v >= 1 {
		return 1
	}
	return v
}

// photo2Reflect maps index i into [0,n) using OpenCV's BORDER_REFLECT_101
// (gfedcb|abcdefgh|gfedcba) mirroring, the same rule the root package uses for
// separable filtering. n must be positive.
func photo2Reflect(i, n int) int {
	if n == 1 {
		return 0
	}
	period := 2 * (n - 1)
	i = ((i % period) + period) % period
	if i >= n {
		i = period - i
	}
	return i
}

// photo2GaussianKernel returns a normalised 1-D Gaussian kernel for the given
// standard deviation. The radius is ceil(3*sigma) so the tails are negligible.
// A non-positive sigma yields the identity kernel {1}.
func photo2GaussianKernel(sigma float64) []float64 {
	if sigma <= 0 {
		return []float64{1}
	}
	radius := int(math.Ceil(3 * sigma))
	if radius < 1 {
		radius = 1
	}
	size := 2*radius + 1
	k := make([]float64, size)
	var sum float64
	twoSigma2 := 2 * sigma * sigma
	for i := -radius; i <= radius; i++ {
		v := math.Exp(-float64(i*i) / twoSigma2)
		k[i+radius] = v
		sum += v
	}
	for i := range k {
		k[i] /= sum
	}
	return k
}

// photo2Luma returns the Rec.709 relative luminance of a linear RGB triple.
func photo2Luma(r, g, b float64) float64 {
	return 0.2126*r + 0.7152*g + 0.0722*b
}

// photo2RequireRGB panics if img is nil, empty or not three-channel.
func photo2RequireRGB(img *cv.Mat, name string) {
	if img == nil || img.Empty() {
		panic(fmt.Sprintf("photo2: %s given an empty image", name))
	}
	if img.Channels != 3 {
		panic(fmt.Sprintf("photo2: %s requires 3 channels, got %d", name, img.Channels))
	}
}

// photo2RequireImage panics if img is nil or empty.
func photo2RequireImage(img *cv.Mat, name string) {
	if img == nil || img.Empty() {
		panic(fmt.Sprintf("photo2: %s given an empty image", name))
	}
}

// photo2RequireFloat panics if f is nil or has no data.
func photo2RequireFloat(f *cv.FloatMat, name string) {
	if f == nil || len(f.Data) == 0 {
		panic(fmt.Sprintf("photo2: %s given an empty FloatMat", name))
	}
}

// photo2RequireChannels panics if channels is empty, contains nil planes, or the
// planes disagree in size.
func photo2RequireChannels(channels []*cv.FloatMat, name string) (rows, cols int) {
	if len(channels) == 0 {
		panic(fmt.Sprintf("photo2: %s given no channels", name))
	}
	if channels[0] == nil || len(channels[0].Data) == 0 {
		panic(fmt.Sprintf("photo2: %s given an empty channel", name))
	}
	rows, cols = channels[0].Rows, channels[0].Cols
	for _, c := range channels {
		if c == nil || c.Rows != rows || c.Cols != cols {
			panic(fmt.Sprintf("photo2: %s channels must share dimensions", name))
		}
	}
	return rows, cols
}

// photo2OddAtLeast forces n to a positive odd integer, using def when n<=0.
func photo2OddAtLeast(n, def int) int {
	if n <= 0 {
		n = def
	}
	if n%2 == 0 {
		n++
	}
	return n
}

// photo2CloneFloat returns a deep copy of f.
func photo2CloneFloat(f *cv.FloatMat) *cv.FloatMat {
	out := cv.NewFloatMat(f.Rows, f.Cols)
	copy(out.Data, f.Data)
	return out
}
