package intensity

import (
	"fmt"
	"math"

	cv "github.com/malcolmston/opencv"
)

// GammaCorrection applies the power-law (gamma) transform
//
//	s = 255 · (r/255)^gamma
//
// to every channel of img and returns a new Mat. gamma must be positive; it
// panics otherwise. gamma == 1 is the identity. gamma < 1 brightens the image
// (it lifts midtones toward white), while gamma > 1 darkens it. This is the
// standard tool for compensating a display's gamma or for perceptual
// brightening.
func GammaCorrection(img *cv.Mat, gamma float64) *cv.Mat {
	requireImage(img, "GammaCorrection")
	if gamma <= 0 || math.IsNaN(gamma) || math.IsInf(gamma, 0) {
		panic(fmt.Sprintf("intensity: GammaCorrection requires gamma > 0, got %v", gamma))
	}
	lut := buildLUT(func(i int) float64 {
		return 255 * math.Pow(float64(i)/255, gamma)
	})
	return applyLUT(img, lut)
}

// logC is the scale factor 255/log(1+255) that makes [LogTransform] map the
// input range [0,255] onto [0,255].
var logC = 255 / math.Log(256)

// LogTransform applies the logarithmic intensity transform
//
//	s = c · log(1 + r),   c = 255 / log(256)
//
// to every channel of img and returns a new Mat. The map is monotonically
// increasing and reproduces the endpoints exactly (0→0, 255→255). It compresses
// the dynamic range, expanding detail in dark regions at the expense of bright
// ones — the classic remedy for images with a large intensity range such as
// Fourier spectra.
func LogTransform(img *cv.Mat) *cv.Mat {
	requireImage(img, "LogTransform")
	lut := buildLUT(func(i int) float64 {
		return logC * math.Log(1+float64(i))
	})
	return applyLUT(img, lut)
}

// ExpTransform applies the exponential intensity transform
//
//	s = exp(r/c) − 1,   c = 255 / log(256)
//
// to every channel of img and returns a new Mat. It is the exact inverse of
// [LogTransform]: it is monotonically increasing, reproduces the endpoints
// (0→0, 255→255) and expands detail in bright regions while compressing dark
// ones.
func ExpTransform(img *cv.Mat) *cv.Mat {
	requireImage(img, "ExpTransform")
	lut := buildLUT(func(i int) float64 {
		return math.Exp(float64(i)/logC) - 1
	})
	return applyLUT(img, lut)
}

// ContrastStretching applies a three-segment piecewise-linear intensity map
// defined by the two control points (r1,s1) and (r2,s2):
//
//	[0,  r1]  → [0,  s1]
//	[r1, r2]  → [s1, s2]
//	[r2, 255] → [s2, 255]
//
// to every channel of img and returns a new Mat. Choosing s1 < r1 and s2 > r2
// increases contrast in the mid-range. The control abscissae must satisfy
// 0 <= r1 < r2 <= 255; it panics otherwise. The map reproduces the control
// points exactly: input r1 yields s1 and input r2 yields s2 (each rounded to
// the nearest integer).
func ContrastStretching(img *cv.Mat, r1, s1, r2, s2 float64) *cv.Mat {
	requireImage(img, "ContrastStretching")
	if !(r1 >= 0 && r1 < r2 && r2 <= 255) {
		panic(fmt.Sprintf("intensity: ContrastStretching requires 0 <= r1 < r2 <= 255, got r1=%v r2=%v", r1, r2))
	}
	lut := buildLUT(func(i int) float64 {
		v := float64(i)
		switch {
		case v <= r1:
			if r1 == 0 {
				return s1
			}
			return s1 * v / r1
		case v <= r2:
			return s1 + (s2-s1)*(v-r1)/(r2-r1)
		default:
			if r2 == 255 {
				return s2
			}
			return s2 + (255-s2)*(v-r2)/(255-r2)
		}
	})
	return applyLUT(img, lut)
}

// IntensityLevelSlicing highlights the intensity band [low, high]. Samples whose
// value lies in the (inclusive) band are set to value; samples outside the band
// are left unchanged when preserveBackground is true, or set to 0 otherwise.
// The transform is applied to every channel of img and a new Mat is returned. It
// panics if low > high.
func IntensityLevelSlicing(img *cv.Mat, low, high, value uint8, preserveBackground bool) *cv.Mat {
	requireImage(img, "IntensityLevelSlicing")
	if low > high {
		panic(fmt.Sprintf("intensity: IntensityLevelSlicing requires low <= high, got low=%d high=%d", low, high))
	}
	lut := make([]uint8, 256)
	for i := 0; i < 256; i++ {
		switch {
		case uint8(i) >= low && uint8(i) <= high:
			lut[i] = value
		case preserveBackground:
			lut[i] = uint8(i)
		default:
			lut[i] = 0
		}
	}
	return applyLUT(img, lut)
}

// BitPlaneSlicing extracts a single bit plane of img as a binary image: an
// output sample is 255 where bit plane is set in the corresponding input sample
// and 0 otherwise. plane selects the bit, 0 being the least-significant and 7
// the most-significant; it panics if plane is outside [0,7]. The most
// significant planes carry the coarse structure of the image while the least
// significant carry fine detail and noise. The transform is applied to every
// channel of img.
func BitPlaneSlicing(img *cv.Mat, plane int) *cv.Mat {
	requireImage(img, "BitPlaneSlicing")
	if plane < 0 || plane > 7 {
		panic(fmt.Sprintf("intensity: BitPlaneSlicing requires plane in [0,7], got %d", plane))
	}
	lut := make([]uint8, 256)
	for i := 0; i < 256; i++ {
		if (i>>uint(plane))&1 == 1 {
			lut[i] = 255
		}
	}
	return applyLUT(img, lut)
}

// Solarize inverts every sample at or above threshold, leaving lower samples
// unchanged: s = 255 − r for r >= threshold, and s = r otherwise. This is the
// classic darkroom (Sabattier) solarisation effect. The transform is applied to
// every channel of img and a new Mat is returned.
func Solarize(img *cv.Mat, threshold uint8) *cv.Mat {
	requireImage(img, "Solarize")
	lut := make([]uint8, 256)
	for i := 0; i < 256; i++ {
		if uint8(i) >= threshold {
			lut[i] = uint8(255 - i)
		} else {
			lut[i] = uint8(i)
		}
	}
	return applyLUT(img, lut)
}

// Posterize quantises each channel of img to levels evenly spaced intensity
// values spanning [0,255] (the endpoints are always representable), reducing the
// number of distinct tones and producing flat colour bands. levels must be in
// [2,256]; it panics otherwise. levels == 256 is the identity and levels == 2
// thresholds each channel at its midpoint.
func Posterize(img *cv.Mat, levels int) *cv.Mat {
	requireImage(img, "Posterize")
	if levels < 2 || levels > 256 {
		panic(fmt.Sprintf("intensity: Posterize requires levels in [2,256], got %d", levels))
	}
	n := float64(levels - 1)
	lut := buildLUT(func(i int) float64 {
		q := math.Round(float64(i) / 255 * n)
		return q / n * 255
	})
	return applyLUT(img, lut)
}

// Invert returns the photographic negative of img, mapping every sample r to
// 255 − r on every channel.
func Invert(img *cv.Mat) *cv.Mat {
	requireImage(img, "Invert")
	lut := make([]uint8, 256)
	for i := 0; i < 256; i++ {
		lut[i] = uint8(255 - i)
	}
	return applyLUT(img, lut)
}
