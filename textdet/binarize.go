package textdet

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// Polarity selects which side of a threshold is treated as ink (foreground,
// rendered 255) by the binarizers in this file.
type Polarity int

const (
	// DarkText treats samples below the threshold as ink, the usual case of
	// dark writing on a light page.
	DarkText Polarity = iota
	// BrightText treats samples above the threshold as ink, for light writing
	// on a dark background.
	BrightText
)

// textdetHistogram builds a 256-bin luma histogram of src.
func textdetHistogram(gray []uint8) (hist [256]int) {
	for _, v := range gray {
		hist[v]++
	}
	return hist
}

// OtsuThreshold computes the global Otsu threshold of src and returns the grey
// level in [0,255]. Colour input is reduced to luma. It returns [ErrEmpty] for
// an empty image.
func OtsuThreshold(src *cv.Mat) (int, error) {
	gray, _, _, err := textdetGray(src)
	if err != nil {
		return 0, err
	}
	hist := textdetHistogram(gray)
	return textdetOtsu(hist, len(gray)), nil
}

// Binarize thresholds src at a fixed grey level and returns a single-channel
// 0/255 mask in which ink pixels (selected by p) are 255. The threshold is
// clamped to [0,255]. Colour input is reduced to luma.
func Binarize(src *cv.Mat, thresh int, p Polarity) (*cv.Mat, error) {
	gray, rows, cols, err := textdetGray(src)
	if err != nil {
		return nil, err
	}
	if thresh < 0 {
		thresh = 0
	} else if thresh > 255 {
		thresh = 255
	}
	dst := cv.NewMat(rows, cols, 1)
	for i, v := range gray {
		ink := int(v) < thresh
		if p == BrightText {
			ink = int(v) > thresh
		}
		if ink {
			dst.Data[i] = 255
		}
	}
	return dst, nil
}

// Otsu binarizes src with a global Otsu threshold and returns the 0/255 ink
// mask together with the grey level that was used. Ink is selected by p.
func Otsu(src *cv.Mat, p Polarity) (*cv.Mat, int, error) {
	t, err := OtsuThreshold(src)
	if err != nil {
		return nil, 0, err
	}
	mask, err := Binarize(src, t, p)
	if err != nil {
		return nil, 0, err
	}
	return mask, t, nil
}

// ForegroundRatio reports the fraction of ink pixels (selected by p) that a
// fixed-level threshold at thresh would produce, a value in [0,1].
func ForegroundRatio(src *cv.Mat, thresh int, p Polarity) (float64, error) {
	gray, _, _, err := textdetGray(src)
	if err != nil {
		return 0, err
	}
	ink := 0
	for _, v := range gray {
		is := int(v) < thresh
		if p == BrightText {
			is = int(v) > thresh
		}
		if is {
			ink++
		}
	}
	return float64(ink) / float64(len(gray)), nil
}

// IntegralImage is a summed-area representation of a single-channel image that
// answers rectangle-sum, mean and standard-deviation queries in O(1). It stores
// both the sum and the sum of squares of the source samples.
type IntegralImage struct {
	// Rows is the source height.
	Rows int
	// Cols is the source width.
	Cols  int
	sum   []float64
	sqSum []float64
}

// NewIntegralImage builds the summed-area tables of src (reduced to luma if
// multi-channel). It returns [ErrEmpty] for an empty image.
func NewIntegralImage(src *cv.Mat) (*IntegralImage, error) {
	gray, rows, cols, err := textdetGray(src)
	if err != nil {
		return nil, err
	}
	fs := make([]float64, rows*cols)
	fsq := make([]float64, rows*cols)
	for i, v := range gray {
		fv := float64(v)
		fs[i] = fv
		fsq[i] = fv * fv
	}
	return &IntegralImage{
		Rows:  rows,
		Cols:  cols,
		sum:   textdetIntegral(fs, rows, cols),
		sqSum: textdetIntegral(fsq, rows, cols),
	}, nil
}

// clampRect clamps rect to the image and returns its inclusive corners and the
// pixel count. ok is false when the clamped rectangle is empty.
func (ii *IntegralImage) clampRect(rect cv.Rect) (x0, y0, x1, y1, n int, ok bool) {
	x0, y0 = rect.X, rect.Y
	x1, y1 = rect.X+rect.Width-1, rect.Y+rect.Height-1
	if x0 < 0 {
		x0 = 0
	}
	if y0 < 0 {
		y0 = 0
	}
	if x1 >= ii.Cols {
		x1 = ii.Cols - 1
	}
	if y1 >= ii.Rows {
		y1 = ii.Rows - 1
	}
	if x1 < x0 || y1 < y0 {
		return 0, 0, 0, 0, 0, false
	}
	return x0, y0, x1, y1, (x1 - x0 + 1) * (y1 - y0 + 1), true
}

// Sum returns the sum of source samples over rect (clamped to the image). An
// empty rectangle yields 0.
func (ii *IntegralImage) Sum(rect cv.Rect) float64 {
	x0, y0, x1, y1, _, ok := ii.clampRect(rect)
	if !ok {
		return 0
	}
	return textdetRectSum(ii.sum, ii.Cols, x0, y0, x1, y1)
}

// Mean returns the average source sample over rect (clamped to the image). An
// empty rectangle yields 0.
func (ii *IntegralImage) Mean(rect cv.Rect) float64 {
	x0, y0, x1, y1, n, ok := ii.clampRect(rect)
	if !ok {
		return 0
	}
	return textdetRectSum(ii.sum, ii.Cols, x0, y0, x1, y1) / float64(n)
}

// MeanStdDev returns the mean and population standard deviation of the source
// samples over rect (clamped to the image). An empty rectangle yields (0,0).
func (ii *IntegralImage) MeanStdDev(rect cv.Rect) (mean, std float64) {
	x0, y0, x1, y1, n, ok := ii.clampRect(rect)
	if !ok {
		return 0, 0
	}
	s := textdetRectSum(ii.sum, ii.Cols, x0, y0, x1, y1)
	sq := textdetRectSum(ii.sqSum, ii.Cols, x0, y0, x1, y1)
	nf := float64(n)
	mean = s / nf
	variance := sq/nf - mean*mean
	if variance < 0 {
		variance = 0
	}
	return mean, math.Sqrt(variance)
}

// textdetWindowRect returns the (2*radius+1)-square window centred on (x, y).
func textdetWindowRect(x, y, radius int) cv.Rect {
	return cv.Rect{X: x - radius, Y: y - radius, Width: 2*radius + 1, Height: 2*radius + 1}
}

// AdaptiveMean binarizes src with a local mean threshold: a pixel is ink when
// its value differs from the mean of its (2*radius+1)-square neighbourhood by
// more than c on the ink side selected by p. This is the classic adaptive-mean
// (a.k.a. mean-C) method. It returns [ErrInvalidArgument] for radius < 1.
func AdaptiveMean(src *cv.Mat, radius int, c float64, p Polarity) (*cv.Mat, error) {
	if radius < 1 {
		return nil, ErrInvalidArgument
	}
	ii, err := NewIntegralImage(src)
	if err != nil {
		return nil, err
	}
	gray, rows, cols, _ := textdetGray(src)
	dst := cv.NewMat(rows, cols, 1)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			mean := ii.Mean(textdetWindowRect(x, y, radius))
			v := float64(gray[y*cols+x])
			ink := v < mean-c
			if p == BrightText {
				ink = v > mean+c
			}
			if ink {
				dst.Data[y*cols+x] = 255
			}
		}
	}
	return dst, nil
}

// Niblack binarizes src with Niblack's local method. For each pixel the
// threshold is T = mean + k*std over a (2*radius+1)-square window; a pixel is
// ink when it is darker than T (for [DarkText]) or brighter than it (for
// [BrightText]). A typical k is -0.2 for dark text. It returns
// [ErrInvalidArgument] for radius < 1.
func Niblack(src *cv.Mat, radius int, k float64, p Polarity) (*cv.Mat, error) {
	if radius < 1 {
		return nil, ErrInvalidArgument
	}
	ii, err := NewIntegralImage(src)
	if err != nil {
		return nil, err
	}
	gray, rows, cols, _ := textdetGray(src)
	dst := cv.NewMat(rows, cols, 1)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			mean, std := ii.MeanStdDev(textdetWindowRect(x, y, radius))
			t := mean + k*std
			v := float64(gray[y*cols+x])
			ink := v < t
			if p == BrightText {
				ink = v > t
			}
			if ink {
				dst.Data[y*cols+x] = 255
			}
		}
	}
	return dst, nil
}

// Sauvola binarizes src with Sauvola's local method, an improvement on Niblack
// for document images. The threshold is T = mean*(1 + k*(std/r - 1)), where r
// is the dynamic range of standard deviation (typically 128) and k is a small
// positive constant (typically 0.2..0.5). A pixel is ink when darker than T for
// [DarkText] or brighter for [BrightText]. It returns [ErrInvalidArgument] for
// radius < 1 or r <= 0.
func Sauvola(src *cv.Mat, radius int, k, r float64, p Polarity) (*cv.Mat, error) {
	if radius < 1 || r <= 0 {
		return nil, ErrInvalidArgument
	}
	ii, err := NewIntegralImage(src)
	if err != nil {
		return nil, err
	}
	gray, rows, cols, _ := textdetGray(src)
	dst := cv.NewMat(rows, cols, 1)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			mean, std := ii.MeanStdDev(textdetWindowRect(x, y, radius))
			t := mean * (1 + k*(std/r-1))
			v := float64(gray[y*cols+x])
			ink := v < t
			if p == BrightText {
				ink = v > t
			}
			if ink {
				dst.Data[y*cols+x] = 255
			}
		}
	}
	return dst, nil
}

// Wolf binarizes src with the Wolf-Jolion local method, which normalises
// Niblack by the global minimum grey level and the maximum local standard
// deviation: T = (1-k)*mean + k*minGray + k*(std/maxStd)*(mean - minGray). It is
// robust to low-contrast backgrounds. k is typically 0.5. It returns
// [ErrInvalidArgument] for radius < 1.
func Wolf(src *cv.Mat, radius int, k float64, p Polarity) (*cv.Mat, error) {
	if radius < 1 {
		return nil, ErrInvalidArgument
	}
	ii, err := NewIntegralImage(src)
	if err != nil {
		return nil, err
	}
	gray, rows, cols, _ := textdetGray(src)

	minGray := 255.0
	maxStd := 0.0
	stds := make([]float64, rows*cols)
	means := make([]float64, rows*cols)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			mean, std := ii.MeanStdDev(textdetWindowRect(x, y, radius))
			means[y*cols+x] = mean
			stds[y*cols+x] = std
			if std > maxStd {
				maxStd = std
			}
			if fv := float64(gray[y*cols+x]); fv < minGray {
				minGray = fv
			}
		}
	}
	if maxStd == 0 {
		maxStd = 1
	}
	dst := cv.NewMat(rows, cols, 1)
	for i := 0; i < rows*cols; i++ {
		mean := means[i]
		std := stds[i]
		t := (1-k)*mean + k*minGray + k*(std/maxStd)*(mean-minGray)
		v := float64(gray[i])
		ink := v < t
		if p == BrightText {
			ink = v > t
		}
		if ink {
			dst.Data[i] = 255
		}
	}
	return dst, nil
}

// Bernsen binarizes src with Bernsen's local method. For each pixel it takes the
// midrange (min+max)/2 of the (2*radius+1)-square window as the threshold, but
// where the local contrast (max-min) is below contrastThresh the pixel is
// assigned to background, avoiding noise amplification in flat regions. It
// returns [ErrInvalidArgument] for radius < 1.
func Bernsen(src *cv.Mat, radius int, contrastThresh float64, p Polarity) (*cv.Mat, error) {
	if radius < 1 {
		return nil, ErrInvalidArgument
	}
	gray, rows, cols, err := textdetGray(src)
	if err != nil {
		return nil, err
	}
	dst := cv.NewMat(rows, cols, 1)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			lo, hi := 255, 0
			for wy := y - radius; wy <= y+radius; wy++ {
				yy := wy
				if yy < 0 {
					yy = 0
				} else if yy >= rows {
					yy = rows - 1
				}
				for wx := x - radius; wx <= x+radius; wx++ {
					xx := wx
					if xx < 0 {
						xx = 0
					} else if xx >= cols {
						xx = cols - 1
					}
					v := int(gray[yy*cols+xx])
					if v < lo {
						lo = v
					}
					if v > hi {
						hi = v
					}
				}
			}
			contrast := float64(hi - lo)
			mid := float64(lo+hi) / 2
			v := float64(gray[y*cols+x])
			var ink bool
			if contrast < contrastThresh {
				// Low-contrast: assume background.
				ink = false
			} else if p == BrightText {
				ink = v > mid
			} else {
				ink = v < mid
			}
			if ink {
				dst.Data[y*cols+x] = 255
			}
		}
	}
	return dst, nil
}
