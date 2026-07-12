package xphoto

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// OilIntensity selects how the per-pixel intensity used for oil-painting
// bucketing is derived from a colour pixel. It corresponds to OpenCV's
// oilPainting colorSpace argument, which chooses the single-channel image that
// drives the neighbourhood histogram.
type OilIntensity int

const (
	// OilIntensityLuma buckets by BT.601 luma (0.299R+0.587G+0.114B). This is
	// the behaviour of the plain [Oilpainting] entry point.
	OilIntensityLuma OilIntensity = iota
	// OilIntensityValue buckets by the HSV "value", i.e. the maximum channel.
	OilIntensityValue
	// OilIntensityAverage buckets by the unweighted channel average.
	OilIntensityAverage
)

// OilpaintingColorSpace is the parametrised oil-painting stylizer matching
// OpenCV's oilPainting(src, dst, size, dynRatio, colorSpace) overload: it works
// exactly like [Oilpainting] but lets the caller choose, via mode, how the
// intensity that drives the neighbourhood histogram is computed from each colour
// pixel. For every pixel it buckets the square neighbourhood of radius size by
// quantised intensity (intensity/dynRatio), finds the most common bucket and
// outputs the mean colour of the neighbours in that bucket, flattening texture
// into painterly patches.
//
// size is the neighbourhood radius (values <= 0 default to 1); dynRatio is the
// intensity quantisation step (values <= 0 default to 1). Borders are
// replicated. For single-channel input mode is ignored and the sample value is
// used directly, so the result matches [Oilpainting]. The input is not modified.
func OilpaintingColorSpace(src *cv.Mat, size, dynRatio int, mode OilIntensity) *cv.Mat {
	requireNonEmpty(src, "OilpaintingColorSpace")
	if size <= 0 {
		size = 1
	}
	if dynRatio <= 0 {
		dynRatio = 1
	}
	rows, cols, ch := src.Rows, src.Cols, src.Channels
	dst := cv.NewMat(rows, cols, ch)

	nBuckets := 256/dynRatio + 1
	counts := make([]int, nBuckets)
	sums := make([]int, nBuckets*ch)

	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			for i := range counts {
				counts[i] = 0
			}
			for i := range sums {
				sums[i] = 0
			}
			for dy := -size; dy <= size; dy++ {
				for dx := -size; dx <= size; dx++ {
					iy := y + dy
					ix := x + dx
					intensity := int(oilIntensityRep(src, iy, ix, mode))
					bucket := intensity / dynRatio
					if bucket >= nBuckets {
						bucket = nBuckets - 1
					}
					counts[bucket]++
					base := bucket * ch
					for c := 0; c < ch; c++ {
						sums[base+c] += int(atRep(src, iy, ix, c))
					}
				}
			}
			best := 0
			for b := 1; b < nBuckets; b++ {
				if counts[b] > counts[best] {
					best = b
				}
			}
			base := best * ch
			n := counts[best]
			for c := 0; c < ch; c++ {
				dst.Set(y, x, c, uint8((sums[base+c]+n/2)/n))
			}
		}
	}
	return dst
}

// oilIntensityRep returns the bucketing intensity at (y,x) under the given mode,
// with replicated borders. Single-channel input always uses the raw sample.
func oilIntensityRep(m *cv.Mat, y, x int, mode OilIntensity) float64 {
	if m.Channels == 1 {
		return float64(atRep(m, y, x, 0))
	}
	r := float64(atRep(m, y, x, 0))
	g := float64(atRep(m, y, x, 1))
	b := float64(atRep(m, y, x, 2))
	switch mode {
	case OilIntensityValue:
		return math.Max(r, math.Max(g, b))
	case OilIntensityAverage:
		return (r + g + b) / 3
	default:
		return luma(r, g, b)
	}
}
