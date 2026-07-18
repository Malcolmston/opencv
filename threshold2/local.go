package threshold2

import (
	"errors"
	"math"
	"sort"

	cv "github.com/malcolmston/opencv"
)

// threshold2integral builds summed-area tables of the grey values and of the
// squared grey values. Both tables are (rows+1)*(cols+1) with a zero first row
// and column, so the sum over an inclusive pixel rectangle is obtained in O(1).
func threshold2integral(gray []uint8, rows, cols int) (sum, sqSum []float64) {
	w := cols + 1
	sum = make([]float64, (rows+1)*w)
	sqSum = make([]float64, (rows+1)*w)
	for y := 0; y < rows; y++ {
		var rowSum, rowSq float64
		for x := 0; x < cols; x++ {
			v := float64(gray[y*cols+x])
			rowSum += v
			rowSq += v * v
			i := (y+1)*w + (x + 1)
			sum[i] = sum[i-w] + rowSum
			sqSum[i] = sqSum[i-w] + rowSq
		}
	}
	return sum, sqSum
}

// threshold2window returns the count, sum and squared-sum of the window of the
// given radius centred on (y, x), clamped to the image bounds.
func threshold2window(sum, sqSum []float64, rows, cols, y, x, radius int) (count, s, sq float64) {
	w := cols + 1
	y0 := y - radius
	x0 := x - radius
	y1 := y + radius
	x1 := x + radius
	if y0 < 0 {
		y0 = 0
	}
	if x0 < 0 {
		x0 = 0
	}
	if y1 >= rows {
		y1 = rows - 1
	}
	if x1 >= cols {
		x1 = cols - 1
	}
	a := y0 * w
	b := (y1 + 1) * w
	s = sum[b+x1+1] - sum[b+x0] - sum[a+x1+1] + sum[a+x0]
	sq = sqSum[b+x1+1] - sqSum[b+x0] - sqSum[a+x1+1] + sqSum[a+x0]
	count = float64((y1 - y0 + 1) * (x1 - x0 + 1))
	return count, s, sq
}

// threshold2checkWindow validates an odd window size and returns its radius.
func threshold2checkWindow(window int) (int, error) {
	if window < 3 || window%2 == 0 {
		return 0, errors.New("threshold2: window must be an odd integer >= 3")
	}
	return window / 2, nil
}

// threshold2localApply computes a per-pixel threshold from the local mean and
// standard deviation over the given window and binarizes accordingly. The
// caller supplies thresh, which maps (mean, std) to a grey threshold.
func threshold2localApply(src *cv.Mat, window int, p Polarity, thresh func(mean, std float64) float64) (*cv.Mat, error) {
	radius, err := threshold2checkWindow(window)
	if err != nil {
		return nil, err
	}
	gray, rows, cols, err := threshold2gray(src)
	if err != nil {
		return nil, err
	}
	sum, sqSum := threshold2integral(gray, rows, cols)
	dst := cv.NewMat(rows, cols, 1)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			count, s, sq := threshold2window(sum, sqSum, rows, cols, y, x, radius)
			mean := s / count
			variance := sq/count - mean*mean
			if variance < 0 {
				variance = 0
			}
			t := thresh(mean, math.Sqrt(variance))
			i := y*cols + x
			fg := float64(gray[i]) > t
			if p == ObjectDark {
				fg = float64(gray[i]) <= t
			}
			if fg {
				dst.Data[i] = 255
			}
		}
	}
	return dst, nil
}

// AdaptiveMean binarizes src by comparing each pixel to the mean of its
// window-by-window neighbourhood minus the constant c. window must be an odd
// integer of at least 3. This is the classic adaptive-mean (local average)
// threshold.
func AdaptiveMean(src *cv.Mat, window int, c float64, p Polarity) (*cv.Mat, error) {
	return threshold2localApply(src, window, p, func(mean, _ float64) float64 {
		return mean - c
	})
}

// AdaptiveGaussian binarizes src by comparing each pixel to a Gaussian-weighted
// mean of its neighbourhood minus the constant c. window must be an odd integer
// of at least 3; the Gaussian sigma defaults to window/6 as in OpenCV.
func AdaptiveGaussian(src *cv.Mat, window int, c float64, p Polarity) (*cv.Mat, error) {
	radius, err := threshold2checkWindow(window)
	if err != nil {
		return nil, err
	}
	gray, rows, cols, err := threshold2gray(src)
	if err != nil {
		return nil, err
	}
	sigma := float64(window) / 6.0
	if sigma <= 0 {
		sigma = 1
	}
	kernel := make([]float64, window)
	var ksum float64
	for i := -radius; i <= radius; i++ {
		v := math.Exp(-float64(i*i) / (2 * sigma * sigma))
		kernel[i+radius] = v
		ksum += v
	}
	for i := range kernel {
		kernel[i] /= ksum
	}
	at := func(y, x int) float64 {
		if y < 0 {
			y = 0
		} else if y >= rows {
			y = rows - 1
		}
		if x < 0 {
			x = 0
		} else if x >= cols {
			x = cols - 1
		}
		return float64(gray[y*cols+x])
	}
	// Separable convolution: horizontal then vertical.
	tmp := make([]float64, rows*cols)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			var acc float64
			for k := -radius; k <= radius; k++ {
				acc += kernel[k+radius] * at(y, x+k)
			}
			tmp[y*cols+x] = acc
		}
	}
	dst := cv.NewMat(rows, cols, 1)
	vat := func(y, x int) float64 {
		if y < 0 {
			y = 0
		} else if y >= rows {
			y = rows - 1
		}
		return tmp[y*cols+x]
	}
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			var acc float64
			for k := -radius; k <= radius; k++ {
				acc += kernel[k+radius] * vat(y+k, x)
			}
			t := acc - c
			i := y*cols + x
			fg := float64(gray[i]) > t
			if p == ObjectDark {
				fg = float64(gray[i]) <= t
			}
			if fg {
				dst.Data[i] = 255
			}
		}
	}
	return dst, nil
}

// AdaptiveMedian binarizes src by comparing each pixel to the median of its
// window-by-window neighbourhood minus the constant c. window must be an odd
// integer of at least 3. It is more robust to outliers than [AdaptiveMean] at
// higher cost.
func AdaptiveMedian(src *cv.Mat, window int, c float64, p Polarity) (*cv.Mat, error) {
	radius, err := threshold2checkWindow(window)
	if err != nil {
		return nil, err
	}
	gray, rows, cols, err := threshold2gray(src)
	if err != nil {
		return nil, err
	}
	dst := cv.NewMat(rows, cols, 1)
	buf := make([]int, 0, window*window)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			buf = buf[:0]
			for dy := -radius; dy <= radius; dy++ {
				yy := y + dy
				if yy < 0 {
					yy = 0
				} else if yy >= rows {
					yy = rows - 1
				}
				for dx := -radius; dx <= radius; dx++ {
					xx := x + dx
					if xx < 0 {
						xx = 0
					} else if xx >= cols {
						xx = cols - 1
					}
					buf = append(buf, int(gray[yy*cols+xx]))
				}
			}
			sort.Ints(buf)
			med := float64(buf[len(buf)/2])
			t := med - c
			i := y*cols + x
			fg := float64(gray[i]) > t
			if p == ObjectDark {
				fg = float64(gray[i]) <= t
			}
			if fg {
				dst.Data[i] = 255
			}
		}
	}
	return dst, nil
}

// Sauvola binarizes src with Sauvola's adaptive method, using the local
// threshold mean * (1 + k*(std/r - 1)). Typical parameters are window 15, k
// 0.5 and r 128 (r defaults to 128 when zero). It excels at document images
// with uneven illumination; use [ObjectDark] for dark text on a light page.
func Sauvola(src *cv.Mat, window int, k, r float64, p Polarity) (*cv.Mat, error) {
	if r == 0 {
		r = 128
	}
	return threshold2localApply(src, window, p, func(mean, std float64) float64 {
		return mean * (1 + k*(std/r-1))
	})
}

// Niblack binarizes src with Niblack's adaptive method, using the local
// threshold mean + k*std. k is typically negative (about -0.2). Typical window
// is 15. Foreground selection follows p.
func Niblack(src *cv.Mat, window int, k float64, p Polarity) (*cv.Mat, error) {
	return threshold2localApply(src, window, p, func(mean, std float64) float64 {
		return mean + k*std
	})
}

// Bernsen binarizes src with Bernsen's method. For each pixel the local
// contrast (window max minus min) is measured: if it is at least
// contrastThreshold the pixel is compared to the mid-range (max+min)/2,
// otherwise the whole neighbourhood is assigned to one class according to
// whether the mid-range reaches 128. window must be odd and >= 3; a typical
// contrastThreshold is 15.
func Bernsen(src *cv.Mat, window int, contrastThreshold int, p Polarity) (*cv.Mat, error) {
	radius, err := threshold2checkWindow(window)
	if err != nil {
		return nil, err
	}
	gray, rows, cols, err := threshold2gray(src)
	if err != nil {
		return nil, err
	}
	dst := cv.NewMat(rows, cols, 1)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			mn, mx := 255, 0
			for dy := -radius; dy <= radius; dy++ {
				yy := y + dy
				if yy < 0 {
					yy = 0
				} else if yy >= rows {
					yy = rows - 1
				}
				for dx := -radius; dx <= radius; dx++ {
					xx := x + dx
					if xx < 0 {
						xx = 0
					} else if xx >= cols {
						xx = cols - 1
					}
					v := int(gray[yy*cols+xx])
					if v < mn {
						mn = v
					}
					if v > mx {
						mx = v
					}
				}
			}
			mid := (mn + mx) / 2
			i := y*cols + x
			var fg bool
			if mx-mn >= contrastThreshold {
				if p == ObjectDark {
					fg = int(gray[i]) <= mid
				} else {
					fg = int(gray[i]) > mid
				}
			} else {
				// Uniform region: assign the whole block by its brightness.
				bright := mid >= 128
				if p == ObjectDark {
					fg = !bright
				} else {
					fg = bright
				}
			}
			if fg {
				dst.Data[i] = 255
			}
		}
	}
	return dst, nil
}

// Wolf binarizes src with the Wolf-Jolion method, an improvement on Niblack
// that normalises by the global maximum local contrast. The local threshold is
// mean - k*(1 - std/rMax)*(mean - globalMin), where rMax is the largest local
// standard deviation and globalMin the darkest pixel. k is typically 0.5 and
// window 15. Foreground selection follows p.
func Wolf(src *cv.Mat, window int, k float64, p Polarity) (*cv.Mat, error) {
	radius, err := threshold2checkWindow(window)
	if err != nil {
		return nil, err
	}
	gray, rows, cols, err := threshold2gray(src)
	if err != nil {
		return nil, err
	}
	sum, sqSum := threshold2integral(gray, rows, cols)
	means := make([]float64, rows*cols)
	stds := make([]float64, rows*cols)
	rMax := 0.0
	globalMin := 255
	for _, v := range gray {
		if int(v) < globalMin {
			globalMin = int(v)
		}
	}
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			count, s, sq := threshold2window(sum, sqSum, rows, cols, y, x, radius)
			mean := s / count
			variance := sq/count - mean*mean
			if variance < 0 {
				variance = 0
			}
			std := math.Sqrt(variance)
			i := y*cols + x
			means[i] = mean
			stds[i] = std
			if std > rMax {
				rMax = std
			}
		}
	}
	if rMax == 0 {
		rMax = 1
	}
	dst := cv.NewMat(rows, cols, 1)
	for i := range gray {
		t := means[i] - k*(1-stds[i]/rMax)*(means[i]-float64(globalMin))
		fg := float64(gray[i]) > t
		if p == ObjectDark {
			fg = float64(gray[i]) <= t
		}
		if fg {
			dst.Data[i] = 255
		}
	}
	return dst, nil
}

// NICK binarizes src with the NICK method, which extends Niblack for low
// contrast images using the local threshold mean + k*sqrt((sumSq - mean^2)/n),
// where the statistics are taken over the window. k is typically between -0.2
// and -0.1 and window 15. Foreground selection follows p.
func NICK(src *cv.Mat, window int, k float64, p Polarity) (*cv.Mat, error) {
	return threshold2localApplyRaw(src, window, p, func(count, s, sq float64) float64 {
		mean := s / count
		val := (sq - mean*mean) / count
		if val < 0 {
			val = 0
		}
		return mean + k*math.Sqrt(val)
	})
}

// Phansalkar binarizes src with Phansalkar's method, tuned for low-contrast
// stained images. Working on intensities normalised to [0,1], the local
// threshold is mean * (1 + p0*exp(-q*mean) + k*(std/r - 1)) with defaults
// p0 = 2, q = 10, k = 0.25 and r = 0.5. window must be odd and >= 3.
// Foreground selection follows the polarity argument.
func Phansalkar(src *cv.Mat, window int, k, r float64, p Polarity) (*cv.Mat, error) {
	if r == 0 {
		r = 0.5
	}
	const p0 = 2.0
	const q = 10.0
	return threshold2localApplyRaw(src, window, p, func(count, s, sq float64) float64 {
		mean := (s / count) / 255.0
		variance := (sq/count)/(255.0*255.0) - mean*mean
		if variance < 0 {
			variance = 0
		}
		std := math.Sqrt(variance)
		t := mean * (1 + p0*math.Exp(-q*mean) + k*(std/r-1))
		return t * 255.0
	})
}

// threshold2localApplyRaw is like threshold2localApply but passes the raw
// window count, sum and squared-sum to the threshold function.
func threshold2localApplyRaw(src *cv.Mat, window int, p Polarity, thresh func(count, s, sq float64) float64) (*cv.Mat, error) {
	radius, err := threshold2checkWindow(window)
	if err != nil {
		return nil, err
	}
	gray, rows, cols, err := threshold2gray(src)
	if err != nil {
		return nil, err
	}
	sum, sqSum := threshold2integral(gray, rows, cols)
	dst := cv.NewMat(rows, cols, 1)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			count, s, sq := threshold2window(sum, sqSum, rows, cols, y, x, radius)
			t := thresh(count, s, sq)
			i := y*cols + x
			fg := float64(gray[i]) > t
			if p == ObjectDark {
				fg = float64(gray[i]) <= t
			}
			if fg {
				dst.Data[i] = 255
			}
		}
	}
	return dst, nil
}
