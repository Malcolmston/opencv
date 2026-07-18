package threshold2

import (
	"errors"
	"math"

	cv "github.com/malcolmston/opencv"
)

// threshold2binsOf returns the 256-bin histogram of src as a plain array plus
// the pixel total, or an error for empty input.
func threshold2binsOf(src *cv.Mat) ([256]int, int, error) {
	var bins [256]int
	gray, _, _, err := threshold2gray(src)
	if err != nil {
		return bins, 0, err
	}
	for _, v := range gray {
		bins[v]++
	}
	return bins, len(gray), nil
}

// threshold2clamp forces v into [0,255].
func threshold2clamp(v int) int {
	if v < 0 {
		return 0
	}
	if v > 255 {
		return 255
	}
	return v
}

// otsuFromBins computes the Otsu threshold of a histogram by maximising the
// between-class variance.
func otsuFromBins(bins [256]int, total int) int {
	if total == 0 {
		return 0
	}
	var sum float64
	for i := 0; i < 256; i++ {
		sum += float64(i) * float64(bins[i])
	}
	var sumB, wB, maxVar float64
	best := 0
	for t := 0; t < 256; t++ {
		wB += float64(bins[t])
		if wB == 0 {
			continue
		}
		wF := float64(total) - wB
		if wF == 0 {
			break
		}
		sumB += float64(t) * float64(bins[t])
		mB := sumB / wB
		mF := (sum - sumB) / wF
		between := wB * wF * (mB - mF) * (mB - mF)
		if between > maxVar {
			maxVar = between
			best = t
		}
	}
	return best
}

// OtsuThreshold returns the global threshold in [0,255] that maximises the
// between-class variance of src's intensity histogram (Otsu's method).
func OtsuThreshold(src *cv.Mat) (int, error) {
	bins, total, err := threshold2binsOf(src)
	if err != nil {
		return 0, err
	}
	return otsuFromBins(bins, total), nil
}

// Otsu binarizes src using the threshold from [OtsuThreshold] and returns the
// mask (foreground selected by p) together with the threshold used.
func Otsu(src *cv.Mat, p Polarity) (*cv.Mat, int, error) {
	t, err := OtsuThreshold(src)
	if err != nil {
		return nil, 0, err
	}
	dst, _ := Binarize(src, t, p)
	return dst, t, nil
}

// TriangleThreshold returns the global threshold chosen by the triangle
// (Zack) method, which finds the level furthest from the line joining the
// histogram peak to its far tail.
func TriangleThreshold(src *cv.Mat) (int, error) {
	bins, total, err := threshold2binsOf(src)
	if err != nil {
		return 0, err
	}
	if total == 0 {
		return 0, nil
	}
	d := bins // local mutable copy

	minIdx := 0
	for i := 0; i < 256; i++ {
		if d[i] > 0 {
			minIdx = i
			break
		}
	}
	if minIdx > 0 {
		minIdx--
	}
	min2 := 255
	for i := 255; i > 0; i-- {
		if d[i] > 0 {
			min2 = i
			break
		}
	}
	if min2 < 255 {
		min2++
	}
	maxIdx, dmax := 0, 0
	for i := 0; i < 256; i++ {
		if d[i] > dmax {
			maxIdx = i
			dmax = d[i]
		}
	}
	inverted := false
	if (maxIdx - minIdx) < (min2 - maxIdx) {
		inverted = true
		l, r := 0, 255
		for l < r {
			d[l], d[r] = d[r], d[l]
			l++
			r--
		}
		minIdx = 255 - min2
		maxIdx = 255 - maxIdx
	}
	if minIdx == maxIdx {
		return minIdx, nil
	}
	nx := float64(d[maxIdx])
	ny := float64(minIdx - maxIdx)
	norm := math.Sqrt(nx*nx + ny*ny)
	nx /= norm
	ny /= norm
	dd := nx*float64(minIdx) + ny*float64(d[minIdx])
	split := minIdx
	var splitDist float64
	for i := minIdx + 1; i <= maxIdx; i++ {
		nd := nx*float64(i) + ny*float64(d[i]) - dd
		if nd > splitDist {
			split = i
			splitDist = nd
		}
	}
	split--
	if inverted {
		return threshold2clamp(255 - split), nil
	}
	return threshold2clamp(split), nil
}

// Triangle binarizes src using the threshold from [TriangleThreshold].
func Triangle(src *cv.Mat, p Polarity) (*cv.Mat, int, error) {
	t, err := TriangleThreshold(src)
	if err != nil {
		return nil, 0, err
	}
	dst, _ := Binarize(src, t, p)
	return dst, t, nil
}

// MeanThreshold returns the mean grey level of src, rounded down. It is the
// simplest global threshold and a useful baseline.
func MeanThreshold(src *cv.Mat) (int, error) {
	bins, total, err := threshold2binsOf(src)
	if err != nil {
		return 0, err
	}
	if total == 0 {
		return 0, nil
	}
	var sum float64
	for i := 0; i < 256; i++ {
		sum += float64(i) * float64(bins[i])
	}
	return int(math.Floor(sum / float64(total))), nil
}

// Mean binarizes src at its mean grey level (see [MeanThreshold]).
func Mean(src *cv.Mat, p Polarity) (*cv.Mat, int, error) {
	t, err := MeanThreshold(src)
	if err != nil {
		return nil, 0, err
	}
	dst, _ := Binarize(src, t, p)
	return dst, t, nil
}

// MedianThreshold returns the median grey level of src, i.e. the smallest
// level at which the cumulative histogram reaches half the pixels.
func MedianThreshold(src *cv.Mat) (int, error) {
	bins, total, err := threshold2binsOf(src)
	if err != nil {
		return 0, err
	}
	if total == 0 {
		return 0, nil
	}
	half := float64(total) / 2
	run := 0.0
	for i := 0; i < 256; i++ {
		run += float64(bins[i])
		if run >= half {
			return i, nil
		}
	}
	return 255, nil
}

// Median binarizes src at its median grey level (see [MedianThreshold]).
func Median(src *cv.Mat, p Polarity) (*cv.Mat, int, error) {
	t, err := MedianThreshold(src)
	if err != nil {
		return nil, 0, err
	}
	dst, _ := Binarize(src, t, p)
	return dst, t, nil
}

// IsoDataThreshold returns the threshold found by the ISODATA (Ridler-Calvard)
// inter-means iteration: starting from the histogram mean it repeatedly moves
// the threshold to the midpoint of the two class means until it stops changing.
func IsoDataThreshold(src *cv.Mat) (int, error) {
	bins, total, err := threshold2binsOf(src)
	if err != nil {
		return 0, err
	}
	if total == 0 {
		return 0, nil
	}
	var sum float64
	for i := 0; i < 256; i++ {
		sum += float64(i) * float64(bins[i])
	}
	t := int(sum / float64(total))
	for iter := 0; iter < 1000; iter++ {
		var sumB, wB float64
		for i := 0; i <= t; i++ {
			wB += float64(bins[i])
			sumB += float64(i) * float64(bins[i])
		}
		wF := float64(total) - wB
		sumF := sum - sumB
		if wB == 0 || wF == 0 {
			break
		}
		mB := sumB / wB
		mF := sumF / wF
		nt := int((mB + mF) / 2)
		if nt == t {
			break
		}
		t = nt
	}
	return t, nil
}

// IsoData binarizes src using the threshold from [IsoDataThreshold].
func IsoData(src *cv.Mat, p Polarity) (*cv.Mat, int, error) {
	t, err := IsoDataThreshold(src)
	if err != nil {
		return nil, 0, err
	}
	dst, _ := Binarize(src, t, p)
	return dst, t, nil
}

// LiThreshold returns the threshold that minimises the cross-entropy between
// the image and its binarized version, using Li and Tam's iterative method.
func LiThreshold(src *cv.Mat) (int, error) {
	bins, total, err := threshold2binsOf(src)
	if err != nil {
		return 0, err
	}
	if total == 0 {
		return 0, nil
	}
	var mean float64
	for i := 0; i < 256; i++ {
		mean += float64(i) * float64(bins[i])
	}
	mean /= float64(total)

	tolerance := 0.5
	newThresh := mean
	threshold := int(mean + 0.5)
	for iter := 0; iter < 1000; iter++ {
		oldThresh := newThresh
		threshold = int(oldThresh + 0.5)
		var sumBack, numBack float64
		for i := 0; i <= threshold && i < 256; i++ {
			sumBack += float64(i) * float64(bins[i])
			numBack += float64(bins[i])
		}
		var sumObj, numObj float64
		for i := threshold + 1; i < 256; i++ {
			sumObj += float64(i) * float64(bins[i])
			numObj += float64(bins[i])
		}
		if numBack == 0 || numObj == 0 {
			break
		}
		meanBack := sumBack / numBack
		meanObj := sumObj / numObj
		if meanBack <= 0 || meanObj <= 0 {
			break
		}
		temp := (meanBack - meanObj) / (math.Log(meanBack) - math.Log(meanObj))
		if temp < 0 {
			newThresh = math.Trunc(temp - 0.5)
		} else {
			newThresh = math.Trunc(temp + 0.5)
		}
		if math.Abs(newThresh-oldThresh) <= tolerance {
			break
		}
	}
	return threshold2clamp(threshold), nil
}

// Li binarizes src using the threshold from [LiThreshold].
func Li(src *cv.Mat, p Polarity) (*cv.Mat, int, error) {
	t, err := LiThreshold(src)
	if err != nil {
		return nil, 0, err
	}
	dst, _ := Binarize(src, t, p)
	return dst, t, nil
}

// KittlerThreshold returns the threshold that minimises the Kittler-Illingworth
// minimum-error criterion, which models each class as a Gaussian and minimises
// the classification error.
func KittlerThreshold(src *cv.Mat) (int, error) {
	bins, total, err := threshold2binsOf(src)
	if err != nil {
		return 0, err
	}
	if total == 0 {
		return 0, nil
	}
	tot := float64(total)
	best := -1
	bestJ := math.Inf(1)
	for t := 0; t < 256; t++ {
		var w1, s1, sq1 float64
		for i := 0; i <= t; i++ {
			c := float64(bins[i])
			w1 += c
			s1 += float64(i) * c
			sq1 += float64(i) * float64(i) * c
		}
		w2 := tot - w1
		if w1 == 0 || w2 == 0 {
			continue
		}
		var s2, sq2 float64
		for i := t + 1; i < 256; i++ {
			c := float64(bins[i])
			s2 += float64(i) * c
			sq2 += float64(i) * float64(i) * c
		}
		p1 := w1 / tot
		p2 := w2 / tot
		var1 := sq1/w1 - (s1/w1)*(s1/w1)
		var2 := sq2/w2 - (s2/w2)*(s2/w2)
		if var1 <= 0 || var2 <= 0 {
			continue
		}
		std1 := math.Sqrt(var1)
		std2 := math.Sqrt(var2)
		j := 1 + 2*(p1*math.Log(std1)+p2*math.Log(std2)) - 2*(p1*math.Log(p1)+p2*math.Log(p2))
		if j < bestJ {
			bestJ = j
			best = t
		}
	}
	if best < 0 {
		// Degenerate histogram (single populated level); fall back to mean.
		return MeanThreshold(src)
	}
	return best, nil
}

// Kittler binarizes src using the threshold from [KittlerThreshold].
func Kittler(src *cv.Mat, p Polarity) (*cv.Mat, int, error) {
	t, err := KittlerThreshold(src)
	if err != nil {
		return nil, 0, err
	}
	dst, _ := Binarize(src, t, p)
	return dst, t, nil
}

// PercentileThreshold returns the grey level whose cumulative histogram share
// is closest to the requested fraction, which must lie in [0,1]. A fraction of
// 0.5 gives the classic 50%-tile (median) threshold.
func PercentileThreshold(src *cv.Mat, fraction float64) (int, error) {
	if fraction < 0 || fraction > 1 {
		return 0, errors.New("threshold2: PercentileThreshold fraction must be in [0,1]")
	}
	bins, total, err := threshold2binsOf(src)
	if err != nil {
		return 0, err
	}
	if total == 0 {
		return 0, nil
	}
	tot := float64(total)
	best := 0
	bestDiff := math.Inf(1)
	run := 0.0
	for i := 0; i < 256; i++ {
		run += float64(bins[i])
		diff := math.Abs(run/tot - fraction)
		if diff < bestDiff {
			bestDiff = diff
			best = i
		}
	}
	return best, nil
}

// Percentile binarizes src at the level returned by [PercentileThreshold] for
// the given fraction.
func Percentile(src *cv.Mat, fraction float64, p Polarity) (*cv.Mat, int, error) {
	t, err := PercentileThreshold(src, fraction)
	if err != nil {
		return nil, 0, err
	}
	dst, _ := Binarize(src, t, p)
	return dst, t, nil
}

// MomentsThreshold returns the threshold chosen by Tsai's moment-preserving
// method, which selects the level whose binarization preserves the first three
// moments of the original grey-level distribution.
func MomentsThreshold(src *cv.Mat) (int, error) {
	bins, total, err := threshold2binsOf(src)
	if err != nil {
		return 0, err
	}
	if total == 0 {
		return 0, nil
	}
	var histo [256]float64
	tot := float64(total)
	for i := 0; i < 256; i++ {
		histo[i] = float64(bins[i]) / tot
	}
	m0 := 1.0
	var m1, m2, m3 float64
	for i := 0; i < 256; i++ {
		fi := float64(i)
		m1 += fi * histo[i]
		m2 += fi * fi * histo[i]
		m3 += fi * fi * fi * histo[i]
	}
	cd := m0*m2 - m1*m1
	if cd == 0 {
		return MeanThreshold(src)
	}
	c0 := (-m2*m2 + m1*m3) / cd
	c1 := (m0*(-m3) + m2*m1) / cd
	disc := c1*c1 - 4.0*c0
	if disc < 0 {
		return MeanThreshold(src)
	}
	z0 := 0.5 * (-c1 - math.Sqrt(disc))
	z1 := 0.5 * (-c1 + math.Sqrt(disc))
	p0 := (z1 - m1) / (z1 - z0)
	sum := 0.0
	for i := 0; i < 256; i++ {
		sum += histo[i]
		if sum > p0 {
			return i, nil
		}
	}
	return 255, nil
}

// Moments binarizes src using the threshold from [MomentsThreshold].
func Moments(src *cv.Mat, p Polarity) (*cv.Mat, int, error) {
	t, err := MomentsThreshold(src)
	if err != nil {
		return nil, 0, err
	}
	dst, _ := Binarize(src, t, p)
	return dst, t, nil
}

// threshold2bimodal reports whether a smoothed histogram has exactly two local
// maxima.
func threshold2bimodal(y []float64) bool {
	modes := 0
	for k := 1; k < len(y)-1; k++ {
		if y[k-1] < y[k] && y[k+1] < y[k] {
			modes++
			if modes > 2 {
				return false
			}
		}
	}
	return modes == 2
}

// threshold2smoothToBimodal repeatedly applies a 3-point moving average to the
// histogram until it becomes bimodal, and returns the smoothed histogram. The
// second return reports whether convergence succeeded within the iteration cap.
func threshold2smoothToBimodal(bins [256]int) ([]float64, bool) {
	h := make([]float64, 256)
	for i := 0; i < 256; i++ {
		h[i] = float64(bins[i])
	}
	tmp := make([]float64, 256)
	for iter := 0; iter < 10000; iter++ {
		if threshold2bimodal(h) {
			return h, true
		}
		tmp[0] = (h[0] + h[1]) / 3
		for i := 1; i < 255; i++ {
			tmp[i] = (h[i-1] + h[i] + h[i+1]) / 3
		}
		tmp[255] = (h[254] + h[255]) / 3
		copy(h, tmp)
	}
	return h, false
}

// MinimumThreshold returns the valley between the two peaks of the histogram
// after Prewitt-Mendelsohn smoothing (the minimum method). It returns an error
// if the histogram cannot be smoothed to a bimodal shape.
func MinimumThreshold(src *cv.Mat) (int, error) {
	bins, total, err := threshold2binsOf(src)
	if err != nil {
		return 0, err
	}
	if total == 0 {
		return 0, nil
	}
	h, ok := threshold2smoothToBimodal(bins)
	if !ok {
		return 0, errors.New("threshold2: MinimumThreshold histogram is not bimodal")
	}
	for i := 1; i < 255; i++ {
		if h[i-1] > h[i] && h[i+1] >= h[i] {
			return i, nil
		}
	}
	return 0, errors.New("threshold2: MinimumThreshold found no valley")
}

// Minimum binarizes src using the threshold from [MinimumThreshold].
func Minimum(src *cv.Mat, p Polarity) (*cv.Mat, int, error) {
	t, err := MinimumThreshold(src)
	if err != nil {
		return nil, 0, err
	}
	dst, _ := Binarize(src, t, p)
	return dst, t, nil
}

// IntermodesThreshold returns the average of the two peaks of the histogram
// after Prewitt-Mendelsohn smoothing (the intermodes method). It returns an
// error if the histogram cannot be smoothed to a bimodal shape.
func IntermodesThreshold(src *cv.Mat) (int, error) {
	bins, total, err := threshold2binsOf(src)
	if err != nil {
		return 0, err
	}
	if total == 0 {
		return 0, nil
	}
	h, ok := threshold2smoothToBimodal(bins)
	if !ok {
		return 0, errors.New("threshold2: IntermodesThreshold histogram is not bimodal")
	}
	peaks := 0
	sum := 0
	for i := 1; i < 255; i++ {
		if h[i-1] < h[i] && h[i+1] < h[i] {
			peaks++
			sum += i
		}
	}
	if peaks == 0 {
		return 0, errors.New("threshold2: IntermodesThreshold found no peaks")
	}
	return int(math.Floor(float64(sum) / float64(peaks))), nil
}

// Intermodes binarizes src using the threshold from [IntermodesThreshold].
func Intermodes(src *cv.Mat, p Polarity) (*cv.Mat, int, error) {
	t, err := IntermodesThreshold(src)
	if err != nil {
		return nil, 0, err
	}
	dst, _ := Binarize(src, t, p)
	return dst, t, nil
}
