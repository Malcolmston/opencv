package cv

import (
	"fmt"
	"math"
)

// Canny runs the full Canny edge-detection pipeline on a single-channel image
// and returns a binary edge map (edges are 255, background 0).
//
// The stages are: a small Gaussian smoothing, 3×3 Sobel gradients, non-maximum
// suppression along the gradient direction, and double-threshold hysteresis
// that keeps weak edges only when connected to a strong one. lowThresh and
// highThresh are gradient-magnitude thresholds with lowThresh < highThresh. It
// panics if src is not single-channel.
func Canny(src *Mat, lowThresh, highThresh float64) *Mat {
	requireChannels(src, 1, "Canny")
	if lowThresh > highThresh {
		lowThresh, highThresh = highThresh, lowThresh
	}
	rows, cols := src.Rows, src.Cols

	// 1. Smooth to suppress noise.
	blurred := GaussianBlur(src, 5, 1.4)

	// 2. Gradients (signed) via Sobel.
	gx := SobelFloat(blurred, 1, 0, 3)[0]
	gy := SobelFloat(blurred, 0, 1, 3)[0]

	mag := make([]float64, rows*cols)
	for i := range mag {
		mag[i] = math.Hypot(gx[i], gy[i])
	}

	// 3. Non-maximum suppression.
	suppressed := make([]float64, rows*cols)
	for y := 1; y < rows-1; y++ {
		for x := 1; x < cols-1; x++ {
			i := y*cols + x
			angle := math.Atan2(gy[i], gx[i]) * 180 / math.Pi
			if angle < 0 {
				angle += 180
			}
			var a, b float64
			switch {
			case angle < 22.5 || angle >= 157.5:
				a, b = mag[i-1], mag[i+1]
			case angle < 67.5:
				a, b = mag[i-cols+1], mag[i+cols-1]
			case angle < 112.5:
				a, b = mag[i-cols], mag[i+cols]
			default:
				a, b = mag[i-cols-1], mag[i+cols+1]
			}
			if mag[i] >= a && mag[i] >= b {
				suppressed[i] = mag[i]
			}
		}
	}

	// 4. Double threshold classification.
	const (
		weak   = 1
		strong = 2
	)
	label := make([]uint8, rows*cols)
	for i, v := range suppressed {
		switch {
		case v >= highThresh:
			label[i] = strong
		case v >= lowThresh:
			label[i] = weak
		}
	}

	// 5. Hysteresis: keep weak pixels reachable from a strong one.
	dst := NewMat(rows, cols, 1)
	stack := make([]int, 0, rows*cols)
	for i, l := range label {
		if l == strong {
			dst.Data[i] = 255
			stack = append(stack, i)
		}
	}
	for len(stack) > 0 {
		i := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		y := i / cols
		x := i % cols
		for dy := -1; dy <= 1; dy++ {
			ny := y + dy
			if ny < 0 || ny >= rows {
				continue
			}
			for dx := -1; dx <= 1; dx++ {
				nx := x + dx
				if nx < 0 || nx >= cols {
					continue
				}
				ni := ny*cols + nx
				if label[ni] == weak && dst.Data[ni] == 0 {
					dst.Data[ni] = 255
					stack = append(stack, ni)
				}
			}
		}
	}
	return dst
}

// TemplateMatchMode selects the similarity measure used by [MatchTemplate].
type TemplateMatchMode int

const (
	// TmSqdiff is the sum of squared differences; the best match is the
	// minimum (0 is a perfect match).
	TmSqdiff TemplateMatchMode = iota
	// TmSqdiffNormed is the normalised sum of squared differences in [0,1];
	// the best match is the minimum.
	TmSqdiffNormed
	// TmCcoeff is the correlation coefficient (covariance of mean-subtracted
	// patches); the best match is the maximum.
	TmCcoeff
	// TmCcoeffNormed is the normalised correlation coefficient in [-1,1]; the
	// best match is the maximum.
	TmCcoeffNormed
)

// MatchTemplate slides templ over src and returns a single-channel float result
// map of shape (src.Rows-templ.Rows+1) × (src.Cols-templ.Cols+1). Each entry
// holds the similarity of templ against the patch of src at that top-left
// position under the chosen mode. Both inputs must have the same channel count
// and templ must fit inside src.
//
// The result is returned as a [FloatMat] because match scores are not bounded
// to [0,255]; use [MinMaxLoc] to locate the best match.
func MatchTemplate(src, templ *Mat, mode TemplateMatchMode) *FloatMat {
	if src.Channels != templ.Channels {
		panic("cv: MatchTemplate channel mismatch")
	}
	if templ.Rows > src.Rows || templ.Cols > src.Cols {
		panic("cv: MatchTemplate template larger than source")
	}
	resRows := src.Rows - templ.Rows + 1
	resCols := src.Cols - templ.Cols + 1
	res := NewFloatMat(resRows, resCols)
	ch := src.Channels
	tn := float64(templ.Rows * templ.Cols * ch)

	// Precompute template mean for the coefficient modes.
	var tMean float64
	for _, v := range templ.Data {
		tMean += float64(v)
	}
	tMean /= tn

	for ry := 0; ry < resRows; ry++ {
		for rx := 0; rx < resCols; rx++ {
			switch mode {
			case TmSqdiff, TmSqdiffNormed:
				var ssd, sSrc, sTempl float64
				for ty := 0; ty < templ.Rows; ty++ {
					for tx := 0; tx < templ.Cols; tx++ {
						for c := 0; c < ch; c++ {
							s := float64(src.Data[src.index(ry+ty, rx+tx)+c])
							t := float64(templ.Data[templ.index(ty, tx)+c])
							d := s - t
							ssd += d * d
							sSrc += s * s
							sTempl += t * t
						}
					}
				}
				if mode == TmSqdiffNormed {
					denom := math.Sqrt(sSrc * sTempl)
					if denom == 0 {
						res.Data[ry*resCols+rx] = 0
					} else {
						res.Data[ry*resCols+rx] = ssd / denom
					}
				} else {
					res.Data[ry*resCols+rx] = ssd
				}
			case TmCcoeff, TmCcoeffNormed:
				var sMean float64
				for ty := 0; ty < templ.Rows; ty++ {
					for tx := 0; tx < templ.Cols; tx++ {
						for c := 0; c < ch; c++ {
							sMean += float64(src.Data[src.index(ry+ty, rx+tx)+c])
						}
					}
				}
				sMean /= tn
				var cov, sVar, tVar float64
				for ty := 0; ty < templ.Rows; ty++ {
					for tx := 0; tx < templ.Cols; tx++ {
						for c := 0; c < ch; c++ {
							s := float64(src.Data[src.index(ry+ty, rx+tx)+c]) - sMean
							t := float64(templ.Data[templ.index(ty, tx)+c]) - tMean
							cov += s * t
							sVar += s * s
							tVar += t * t
						}
					}
				}
				if mode == TmCcoeffNormed {
					denom := math.Sqrt(sVar * tVar)
					if denom == 0 {
						res.Data[ry*resCols+rx] = 0
					} else {
						res.Data[ry*resCols+rx] = cov / denom
					}
				} else {
					res.Data[ry*resCols+rx] = cov
				}
			default:
				panic(fmt.Sprintf("cv: MatchTemplate unknown mode %d", mode))
			}
		}
	}
	return res
}

// FloatMat is a single-channel matrix of float64 values, used for results such
// as [MatchTemplate] scores whose range is not confined to [0,255].
type FloatMat struct {
	Rows int
	Cols int
	Data []float64
}

// NewFloatMat allocates a zero-filled FloatMat.
func NewFloatMat(rows, cols int) *FloatMat {
	return &FloatMat{Rows: rows, Cols: cols, Data: make([]float64, rows*cols)}
}

// At returns the value at row y, column x.
func (f *FloatMat) At(y, x int) float64 {
	return f.Data[y*f.Cols+x]
}

// MinMaxLoc scans a FloatMat and returns the minimum and maximum values with
// their (x, y) locations. It panics on an empty matrix.
func MinMaxLoc(f *FloatMat) (minVal, maxVal float64, minX, minY, maxX, maxY int) {
	if len(f.Data) == 0 {
		panic("cv: MinMaxLoc on empty matrix")
	}
	minVal = math.Inf(1)
	maxVal = math.Inf(-1)
	for y := 0; y < f.Rows; y++ {
		for x := 0; x < f.Cols; x++ {
			v := f.Data[y*f.Cols+x]
			if v < minVal {
				minVal = v
				minX, minY = x, y
			}
			if v > maxVal {
				maxVal = v
				maxX, maxY = x, y
			}
		}
	}
	return
}
