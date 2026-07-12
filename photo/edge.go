package photo

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// DomainTransformFilter is an edge-preserving smoothing filter based on the
// domain transform of Gastal and Oliveira (2011), the algorithm OpenCV uses to
// back its edgePreservingFilter. The image is warped into a 1-D "domain
// transform" in which spatial distance grows with local colour change, so that
// neighbouring pixels separated by an edge become far apart; ordinary linear
// smoothing in that warped domain therefore blurs flat regions while leaving
// edges sharp. Filtering is performed as a sequence of horizontal and vertical
// 1-D passes, repeated over several iterations of shrinking spatial extent.
//
// flags selects the 1-D filter applied in the transformed domain:
//   - [RecursFilter] uses the recursive (RF) filter — an exponential IIR pass
//     whose feedback coefficient varies per pixel with the domain transform.
//   - [NormconvFilter] uses normalized convolution (NC) — a normalized box
//     filter whose window is defined by the domain transform.
//
// sigmaS is the spatial standard deviation (larger means more smoothing; a good
// range is roughly 10..200) and sigmaR is the range/intensity standard
// deviation in [0,1] (larger blurs across bigger intensity gaps). The input may
// be single- or three-channel; the output has the same shape. The original is
// not modified.
func DomainTransformFilter(img *cv.Mat, flags EdgePreservingFlag, sigmaS, sigmaR float64) *cv.Mat {
	if img == nil || img.Empty() {
		panic("photo: DomainTransformFilter given an empty image")
	}
	if sigmaS <= 0 {
		sigmaS = 60
	}
	if sigmaR <= 0 {
		sigmaR = 0.4
	}
	rows, cols, ch := img.Rows, img.Cols, img.Channels
	// Range sigma is expressed on the [0,1] scale; convert to 8-bit units.
	sr := sigmaR * 255

	// Per-pixel domain-transform derivatives, computed once from the input.
	// dHdx[y*cols+x] is the transformed distance between (x-1) and x on row y;
	// dVdy[y*cols+x] is the transformed distance between (y-1) and y on column x.
	dHdx := make([]float64, rows*cols)
	dVdy := make([]float64, rows*cols)
	ratio := sigmaS / sr
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			var accH, accV float64
			for c := 0; c < ch; c++ {
				if x > 0 {
					accH += math.Abs(float64(img.At(y, x, c)) - float64(img.At(y, x-1, c)))
				}
				if y > 0 {
					accV += math.Abs(float64(img.At(y, x, c)) - float64(img.At(y-1, x, c)))
				}
			}
			dHdx[y*cols+x] = 1 + ratio*accH
			dVdy[y*cols+x] = 1 + ratio*accV
		}
	}

	// Work in float per channel.
	planes := make([][]float64, ch)
	for c := 0; c < ch; c++ {
		p := make([]float64, rows*cols)
		for i := 0; i < rows*cols; i++ {
			p[i] = float64(img.Data[i*ch+c])
		}
		planes[c] = p
	}

	const iters = 3
	for i := 0; i < iters; i++ {
		// Spatial extent of this iteration (shrinks geometrically).
		sigmaH := sigmaS * math.Sqrt(3) * math.Pow(2, float64(iters-(i+1))) /
			math.Sqrt(math.Pow(4, float64(iters))-1)
		switch flags {
		case NormconvFilter:
			radius := sigmaH * math.Sqrt(3)
			for c := 0; c < ch; c++ {
				ncHorizontal(planes[c], dHdx, rows, cols, radius)
				ncVertical(planes[c], dVdy, rows, cols, radius)
			}
		default: // RecursFilter
			a := math.Exp(-math.Sqrt2 / sigmaH)
			for c := 0; c < ch; c++ {
				rfHorizontal(planes[c], dHdx, rows, cols, a)
				rfVertical(planes[c], dVdy, rows, cols, a)
			}
		}
	}

	out := cv.NewMat(rows, cols, ch)
	for c := 0; c < ch; c++ {
		for i := 0; i < rows*cols; i++ {
			out.Data[i*ch+c] = clampU8(planes[c][i])
		}
	}
	return out
}

// rfHorizontal applies the recursive (IIR) domain-transform filter along each
// row in place. The feedback weight at a step is a raised to the transformed
// distance of that step, so large edges (large distance) attenuate propagation.
func rfHorizontal(p, dHdx []float64, rows, cols int, a float64) {
	for y := 0; y < rows; y++ {
		base := y * cols
		// Left to right.
		for x := 1; x < cols; x++ {
			w := math.Pow(a, dHdx[base+x])
			p[base+x] += w * (p[base+x-1] - p[base+x])
		}
		// Right to left.
		for x := cols - 2; x >= 0; x-- {
			w := math.Pow(a, dHdx[base+x+1])
			p[base+x] += w * (p[base+x+1] - p[base+x])
		}
	}
}

// rfVertical applies the recursive domain-transform filter down each column.
func rfVertical(p, dVdy []float64, rows, cols int, a float64) {
	for x := 0; x < cols; x++ {
		// Top to bottom.
		for y := 1; y < rows; y++ {
			w := math.Pow(a, dVdy[y*cols+x])
			p[y*cols+x] += w * (p[(y-1)*cols+x] - p[y*cols+x])
		}
		// Bottom to top.
		for y := rows - 2; y >= 0; y-- {
			w := math.Pow(a, dVdy[(y+1)*cols+x])
			p[y*cols+x] += w * (p[(y+1)*cols+x] - p[y*cols+x])
		}
	}
}

// ncHorizontal applies a normalized box filter along each row, where the box
// spans a fixed radius in the transformed domain (cumulative domain distance).
func ncHorizontal(p, dHdx []float64, rows, cols int, radius float64) {
	ct := make([]float64, cols)
	prefix := make([]float64, cols+1)
	for y := 0; y < rows; y++ {
		base := y * cols
		ct[0] = 0
		for x := 1; x < cols; x++ {
			ct[x] = ct[x-1] + dHdx[base+x]
		}
		prefix[0] = 0
		for x := 0; x < cols; x++ {
			prefix[x+1] = prefix[x] + p[base+x]
		}
		lo, hi := 0, 0
		for x := 0; x < cols; x++ {
			for lo < cols && ct[lo] < ct[x]-radius {
				lo++
			}
			for hi < cols && ct[hi] <= ct[x]+radius {
				hi++
			}
			// Window is [lo, hi); hi is exclusive.
			p[base+x] = (prefix[hi] - prefix[lo]) / float64(hi-lo)
		}
	}
}

// ncVertical applies the normalized box filter down each column.
func ncVertical(p, dVdy []float64, rows, cols int, radius float64) {
	ct := make([]float64, rows)
	prefix := make([]float64, rows+1)
	for x := 0; x < cols; x++ {
		ct[0] = 0
		for y := 1; y < rows; y++ {
			ct[y] = ct[y-1] + dVdy[y*cols+x]
		}
		prefix[0] = 0
		for y := 0; y < rows; y++ {
			prefix[y+1] = prefix[y] + p[y*cols+x]
		}
		lo, hi := 0, 0
		for y := 0; y < rows; y++ {
			for lo < rows && ct[lo] < ct[y]-radius {
				lo++
			}
			for hi < rows && ct[hi] <= ct[y]+radius {
				hi++
			}
			val := (prefix[hi] - prefix[lo]) / float64(hi-lo)
			p[y*cols+x] = val
		}
	}
}
