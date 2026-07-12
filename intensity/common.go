package intensity

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// lumaWeights are the Rec. 601 luma coefficients used whenever a colour image
// must be reduced to a single brightness plane. They match the ordering of a
// three-channel [cv.Mat], whose samples are stored red, green, blue.
var lumaWeights = [3]float64{0.299, 0.587, 0.114}

// gaussKernel1D returns a normalised, truncated 1-D Gaussian with standard
// deviation sigma. The kernel radius is ceil(3·sigma) so that the discarded
// tails carry well under 1% of the mass. A non-positive sigma yields the unit
// (identity) kernel.
func gaussKernel1D(sigma float64) []float64 {
	if sigma <= 0 {
		return []float64{1}
	}
	radius := int(math.Ceil(3 * sigma))
	if radius < 1 {
		radius = 1
	}
	k := make([]float64, 2*radius+1)
	var sum float64
	twoSigmaSq := 2 * sigma * sigma
	for i := -radius; i <= radius; i++ {
		v := math.Exp(-float64(i*i) / twoSigmaSq)
		k[i+radius] = v
		sum += v
	}
	for i := range k {
		k[i] /= sum
	}
	return k
}

// blurPlaneFloat applies a separable Gaussian blur of standard deviation sigma
// to a single-channel float plane laid out row-major (rows×cols), replicating
// the border. It keeps full floating-point precision, which the log-domain
// Retinex and weighted-least-squares routines rely on, and returns a new slice.
func blurPlaneFloat(src []float64, rows, cols int, sigma float64) []float64 {
	k := gaussKernel1D(sigma)
	r := len(k) / 2
	tmp := make([]float64, len(src))
	for y := 0; y < rows; y++ {
		row := y * cols
		for x := 0; x < cols; x++ {
			var acc float64
			for t := -r; t <= r; t++ {
				xx := x + t
				if xx < 0 {
					xx = 0
				} else if xx >= cols {
					xx = cols - 1
				}
				acc += k[t+r] * src[row+xx]
			}
			tmp[row+x] = acc
		}
	}
	out := make([]float64, len(src))
	for x := 0; x < cols; x++ {
		for y := 0; y < rows; y++ {
			var acc float64
			for t := -r; t <= r; t++ {
				yy := y + t
				if yy < 0 {
					yy = 0
				} else if yy >= rows {
					yy = rows - 1
				}
				acc += k[t+r] * tmp[yy*cols+x]
			}
			out[y*cols+x] = acc
		}
	}
	return out
}

// channelFloat extracts channel c of img into a fresh float slice of length
// Total() holding the raw [0,255] sample values.
func channelFloat(img *cv.Mat, c int) []float64 {
	n := img.Total()
	ch := img.Channels
	out := make([]float64, n)
	for p := 0; p < n; p++ {
		out[p] = float64(img.Data[p*ch+c])
	}
	return out
}

// lumaFloat reduces img to a single brightness plane in [0,255]. A
// single-channel image is returned as-is; a three-channel image is combined
// with the Rec. 601 [lumaWeights]; any other channel count falls back to the
// first channel.
func lumaFloat(img *cv.Mat) []float64 {
	n := img.Total()
	ch := img.Channels
	out := make([]float64, n)
	switch ch {
	case 1:
		for p := 0; p < n; p++ {
			out[p] = float64(img.Data[p])
		}
	case 3:
		for p := 0; p < n; p++ {
			base := p * 3
			out[p] = lumaWeights[0]*float64(img.Data[base]) +
				lumaWeights[1]*float64(img.Data[base+1]) +
				lumaWeights[2]*float64(img.Data[base+2])
		}
	default:
		for p := 0; p < n; p++ {
			out[p] = float64(img.Data[p*ch])
		}
	}
	return out
}

// meanStd returns the mean and (population) standard deviation of a float slice.
// An empty slice yields (0, 0).
func meanStd(v []float64) (mean, std float64) {
	if len(v) == 0 {
		return 0, 0
	}
	var sum float64
	for _, x := range v {
		sum += x
	}
	mean = sum / float64(len(v))
	var sq float64
	for _, x := range v {
		d := x - mean
		sq += d * d
	}
	std = math.Sqrt(sq / float64(len(v)))
	return mean, std
}

// histFloat bins values in [0,255] into a 256-entry integer histogram, clamping
// out-of-range values to the nearest end bin.
func histFloat(v []float64) [256]int {
	var h [256]int
	for _, x := range v {
		i := int(x + 0.5)
		if i < 0 {
			i = 0
		} else if i > 255 {
			i = 255
		}
		h[i]++
	}
	return h
}

// entropy256 returns the Shannon entropy, in bits, of a 256-bin histogram whose
// samples sum to total. An empty histogram has zero entropy.
func entropy256(hist [256]int, total int) float64 {
	if total <= 0 {
		return 0
	}
	inv := 1.0 / float64(total)
	var e float64
	for _, c := range hist {
		if c == 0 {
			continue
		}
		p := float64(c) * inv
		e -= p * math.Log2(p)
	}
	return e
}

// clipBounds finds the intensity levels [lo,hi] that remain after discarding the
// lowest lowFrac and highest highFrac of the population described by hist (which
// must sum to total). Fractions are clamped into [0,0.49]. When the surviving
// range would collapse it returns the full [0,255] span so callers can fall back
// to a no-op stretch.
func clipBounds(hist [256]int, total int, lowFrac, highFrac float64) (lo, hi int) {
	if lowFrac < 0 {
		lowFrac = 0
	}
	if highFrac < 0 {
		highFrac = 0
	}
	if lowFrac > 0.49 {
		lowFrac = 0.49
	}
	if highFrac > 0.49 {
		highFrac = 0.49
	}
	lowCut := int(lowFrac * float64(total))
	highCut := int(highFrac * float64(total))
	lo, hi = 0, 255
	acc := 0
	for i := 0; i < 256; i++ {
		acc += hist[i]
		if acc > lowCut {
			lo = i
			break
		}
	}
	acc = 0
	for i := 255; i >= 0; i-- {
		acc += hist[i]
		if acc > highCut {
			hi = i
			break
		}
	}
	if hi <= lo {
		return 0, 255
	}
	return lo, hi
}
