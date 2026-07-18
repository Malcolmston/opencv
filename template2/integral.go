package template2

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// ToGrayscale returns a single-channel copy of src. A one-channel image is
// cloned; a three-channel image is converted with the Rec.601 luma weights via
// [cv.RGBToGray601]; any other channel count is reduced by averaging the
// channels. The result is always a fresh [cv.Mat] with Channels == 1.
func ToGrayscale(src *cv.Mat) *cv.Mat {
	switch src.Channels {
	case 1:
		return src.Clone()
	case 3:
		return cv.RGBToGray601(src)
	default:
		dst := cv.NewMat(src.Rows, src.Cols, 1)
		ch := src.Channels
		for p := 0; p < src.Rows*src.Cols; p++ {
			var sum int
			base := p * ch
			for c := 0; c < ch; c++ {
				sum += int(src.Data[base+c])
			}
			dst.Data[p] = uint8((sum + ch/2) / ch)
		}
		return dst
	}
}

// Integral holds the summed-area tables of a grayscale image: the running sum
// of samples and of squared samples. Both tables have (Rows+1) rows and
// (Cols+1) columns, with a zero first row and column, so any axis-aligned
// rectangle sum can be read in constant time. Construct one with [NewIntegral].
type Integral struct {
	// Rows is the height of the source image.
	Rows int
	// Cols is the width of the source image.
	Cols int
	// Sum is the summed-area table of sample values, size (Rows+1)*(Cols+1).
	Sum []float64
	// SqSum is the summed-area table of squared sample values.
	SqSum []float64
}

// NewIntegral computes the summed-area tables of src. Multi-channel input is
// reduced to grayscale with [ToGrayscale] first.
func NewIntegral(src *cv.Mat) *Integral {
	gray := ToGrayscale(src)
	rows, cols := gray.Rows, gray.Cols
	w := cols + 1
	sum := make([]float64, (rows+1)*w)
	sq := make([]float64, (rows+1)*w)
	for y := 0; y < rows; y++ {
		var rowSum, rowSq float64
		for x := 0; x < cols; x++ {
			v := float64(gray.Data[y*cols+x])
			rowSum += v
			rowSq += v * v
			idx := (y+1)*w + (x + 1)
			sum[idx] = sum[y*w+(x+1)] + rowSum
			sq[idx] = sq[y*w+(x+1)] + rowSq
		}
	}
	return &Integral{Rows: rows, Cols: cols, Sum: sum, SqSum: sq}
}

// rectValue reads a rectangle sum from a summed-area table. The rectangle spans
// columns [x0,x1) and rows [y0,y1).
func (in *Integral) rectValue(table []float64, x0, y0, x1, y1 int) float64 {
	w := in.Cols + 1
	a := table[y1*w+x1]
	b := table[y0*w+x1]
	c := table[y1*w+x0]
	d := table[y0*w+x0]
	return a - b - c + d
}

// RegionSum returns the sum of sample values over the rectangle spanning
// columns [x0,x1) and rows [y0,y1). Coordinates must satisfy
// 0 <= x0 <= x1 <= Cols and 0 <= y0 <= y1 <= Rows.
func (in *Integral) RegionSum(x0, y0, x1, y1 int) float64 {
	return in.rectValue(in.Sum, x0, y0, x1, y1)
}

// RegionSqSum returns the sum of squared sample values over the rectangle
// spanning columns [x0,x1) and rows [y0,y1).
func (in *Integral) RegionSqSum(x0, y0, x1, y1 int) float64 {
	return in.rectValue(in.SqSum, x0, y0, x1, y1)
}

// RegionMean returns the mean sample value over the rectangle spanning columns
// [x0,x1) and rows [y0,y1). It returns 0 for a degenerate (zero-area) rectangle.
func (in *Integral) RegionMean(x0, y0, x1, y1 int) float64 {
	n := float64((x1 - x0) * (y1 - y0))
	if n <= 0 {
		return 0
	}
	return in.RegionSum(x0, y0, x1, y1) / n
}

// RegionVariance returns the population variance of sample values over the
// rectangle spanning columns [x0,x1) and rows [y0,y1). It returns 0 for a
// degenerate rectangle. The result is clamped at 0 to absorb rounding noise.
func (in *Integral) RegionVariance(x0, y0, x1, y1 int) float64 {
	n := float64((x1 - x0) * (y1 - y0))
	if n <= 0 {
		return 0
	}
	s := in.RegionSum(x0, y0, x1, y1)
	sq := in.RegionSqSum(x0, y0, x1, y1)
	v := sq/n - (s/n)*(s/n)
	if v < 0 {
		return 0
	}
	return v
}

// FastZNCC computes the zero-mean normalised cross-correlation score map using
// integral images to obtain each window's mean and energy in constant time. It
// is mathematically equivalent to [MatchZNCC] on grayscale input but avoids
// re-accumulating the patch statistics at every shift.
//
// Both images are reduced to grayscale with [ToGrayscale]. The returned map has
// the same shape and orientation as [MatchTemplate]; higher scores are better.
func FastZNCC(src, templ *cv.Mat) (*cv.FloatMat, error) {
	return fastNormalized(src, templ, true)
}

// FastNCC computes the (non-zero-mean) normalised cross-correlation score map
// using an integral image for each window's energy. It is mathematically
// equivalent to [MatchNCC] on grayscale input. Both images are reduced to
// grayscale with [ToGrayscale]; higher scores are better.
func FastNCC(src, templ *cv.Mat) (*cv.FloatMat, error) {
	return fastNormalized(src, templ, false)
}

// fastNormalized implements FastNCC (zeroMean=false) and FastZNCC
// (zeroMean=true).
func fastNormalized(src, templ *cv.Mat, zeroMean bool) (*cv.FloatMat, error) {
	if src.Empty() || templ.Empty() {
		return nil, ErrEmptyImage
	}
	if templ.Rows > src.Rows || templ.Cols > src.Cols {
		return nil, ErrTemplateLarger
	}
	g := ToGrayscale(src)
	t := ToGrayscale(templ)
	integ := NewIntegral(g)

	tw, th := t.Cols, t.Rows
	n := float64(tw * th)

	// Template moments.
	var sumT, sumT2 float64
	for _, v := range t.Data {
		f := float64(v)
		sumT += f
		sumT2 += f * f
	}
	meanT := sumT / n
	var tDen float64
	if zeroMean {
		tv := sumT2 - n*meanT*meanT
		if tv < 0 {
			tv = 0
		}
		tDen = math.Sqrt(tv)
	} else {
		tDen = math.Sqrt(sumT2)
	}

	resRows := src.Rows - th + 1
	resCols := src.Cols - tw + 1
	res := cv.NewFloatMat(resRows, resCols)

	for ry := 0; ry < resRows; ry++ {
		for rx := 0; rx < resCols; rx++ {
			// Raw cross-correlation numerator (direct).
			var sumIT float64
			for ty := 0; ty < th; ty++ {
				gRow := (ry + ty) * g.Cols
				tRow := ty * tw
				for tx := 0; tx < tw; tx++ {
					sumIT += float64(g.Data[gRow+rx+tx]) * float64(t.Data[tRow+tx])
				}
			}
			// Window statistics from the integral image in O(1).
			wSum := integ.RegionSum(rx, ry, rx+tw, ry+th)
			wSqSum := integ.RegionSqSum(rx, ry, rx+tw, ry+th)

			var num, iDen float64
			if zeroMean {
				meanI := wSum / n
				num = sumIT - n*meanI*meanT
				iv := wSqSum - wSum*wSum/n
				if iv < 0 {
					iv = 0
				}
				iDen = math.Sqrt(iv)
			} else {
				num = sumIT
				iDen = math.Sqrt(wSqSum)
			}
			denom := iDen * tDen
			if denom == 0 {
				res.Data[ry*resCols+rx] = 0
			} else {
				res.Data[ry*resCols+rx] = num / denom
			}
		}
	}
	return res, nil
}
