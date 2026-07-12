package dnn_superres

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// clampByte rounds v to the nearest integer and clamps it to the uint8 range
// [0,255]. It reimplements the root package's unexported clampToUint8 so this
// subpackage stays dependency-free beyond cv itself.
func clampByte(v float64) uint8 {
	v = math.Round(v)
	if v <= 0 {
		return 0
	}
	if v >= 255 {
		return 255
	}
	return uint8(v)
}

// clampInt clamps i to [lo, hi].
func clampInt(i, lo, hi int) int {
	if i < lo {
		return lo
	}
	if i > hi {
		return hi
	}
	return i
}

// sampleReplicate returns channel c of src at integer pixel (y, x), replicating
// the nearest border sample for out-of-range coordinates (BORDER_REPLICATE).
func sampleReplicate(src *cv.Mat, y, x, c int) float64 {
	y = clampInt(y, 0, src.Rows-1)
	x = clampInt(x, 0, src.Cols-1)
	return float64(src.Data[(y*src.Cols+x)*src.Channels+c])
}

// bilinearAt samples channel c of src at fractional coordinates (fx, fy) with
// border replication. Used by the edge-directed pass to read the base image
// along arbitrary directions.
func bilinearAt(src *cv.Mat, fx, fy float64, c int) float64 {
	x0 := int(math.Floor(fx))
	y0 := int(math.Floor(fy))
	dx := fx - float64(x0)
	dy := fy - float64(y0)
	v00 := sampleReplicate(src, y0, x0, c)
	v01 := sampleReplicate(src, y0, x0+1, c)
	v10 := sampleReplicate(src, y0+1, x0, c)
	v11 := sampleReplicate(src, y0+1, x0+1, c)
	top := v00*(1-dx) + v01*dx
	bot := v10*(1-dx) + v11*dx
	return top*(1-dy) + bot*dy
}

// kernelFunc is a 1-D interpolation weight as a function of the signed distance
// t between a sample tap and the resampling centre.
type kernelFunc func(t float64) float64

// keysCubic is the Keys / Catmull-Rom cubic convolution kernel with a = -0.5,
// the classical bicubic weight (support radius 2). This is the kernel behind
// cv2.INTER_CUBIC.
func keysCubic(t float64) float64 {
	const a = -0.5
	t = math.Abs(t)
	switch {
	case t <= 1:
		return (a+2)*t*t*t - (a+3)*t*t + 1
	case t < 2:
		return a*t*t*t - 5*a*t*t + 8*a*t - 4*a
	default:
		return 0
	}
}

// sinc returns the normalized sinc function sin(pi*x)/(pi*x), with sinc(0)=1.
func sinc(x float64) float64 {
	if x == 0 {
		return 1
	}
	px := math.Pi * x
	return math.Sin(px) / px
}

// lanczos4 is the Lanczos windowed-sinc kernel with parameter a = 4 (support
// radius 4), matching cv2.INTER_LANCZOS4.
func lanczos4(t float64) float64 {
	const a = 4.0
	if t <= -a || t >= a {
		return 0
	}
	return sinc(t) * sinc(t/a)
}

// resampleSeparable resizes src to (dstH, dstW) by applying the 1-D kernel k
// (with the given integer support radius) separably: a horizontal pass into a
// float buffer followed by a vertical pass. Weights are normalized per output
// sample so constant regions are preserved exactly. Borders are replicated.
//
// This is the shared core used by the bicubic and Lanczos upsamplers; it works
// for any positive destination size, not just integer multiples.
func resampleSeparable(src *cv.Mat, dstW, dstH int, k kernelFunc, radius int) *cv.Mat {
	ch := src.Channels
	scaleX := float64(src.Cols) / float64(dstW)
	scaleY := float64(src.Rows) / float64(dstH)

	// Precompute horizontal taps and weights per destination column.
	xTaps := make([][]int, dstW)
	xW := make([][]float64, dstW)
	for x := 0; x < dstW; x++ {
		center := (float64(x)+0.5)*scaleX - 0.5
		base := int(math.Floor(center))
		taps := make([]int, 0, 2*radius)
		ws := make([]float64, 0, 2*radius)
		var sum float64
		for t := base - radius + 1; t <= base+radius; t++ {
			w := k(center - float64(t))
			if w == 0 {
				continue
			}
			taps = append(taps, clampInt(t, 0, src.Cols-1))
			ws = append(ws, w)
			sum += w
		}
		if sum != 0 {
			for i := range ws {
				ws[i] /= sum
			}
		}
		xTaps[x] = taps
		xW[x] = ws
	}

	// Horizontal pass: src.Rows x dstW float image, per channel.
	inter := make([]float64, src.Rows*dstW*ch)
	for y := 0; y < src.Rows; y++ {
		for x := 0; x < dstW; x++ {
			taps := xTaps[x]
			ws := xW[x]
			for c := 0; c < ch; c++ {
				var acc float64
				for i, tx := range taps {
					acc += ws[i] * float64(src.Data[(y*src.Cols+tx)*ch+c])
				}
				inter[(y*dstW+x)*ch+c] = acc
			}
		}
	}

	// Vertical pass into the destination Mat.
	dst := cv.NewMat(dstH, dstW, ch)
	for y := 0; y < dstH; y++ {
		center := (float64(y)+0.5)*scaleY - 0.5
		base := int(math.Floor(center))
		taps := make([]int, 0, 2*radius)
		ws := make([]float64, 0, 2*radius)
		var sum float64
		for t := base - radius + 1; t <= base+radius; t++ {
			w := k(center - float64(t))
			if w == 0 {
				continue
			}
			taps = append(taps, clampInt(t, 0, src.Rows-1))
			ws = append(ws, w)
			sum += w
		}
		if sum != 0 {
			for i := range ws {
				ws[i] /= sum
			}
		}
		for x := 0; x < dstW; x++ {
			for c := 0; c < ch; c++ {
				var acc float64
				for i, ty := range taps {
					acc += ws[i] * inter[(ty*dstW+x)*ch+c]
				}
				dst.Data[(y*dstW+x)*ch+c] = clampByte(acc)
			}
		}
	}
	return dst
}
