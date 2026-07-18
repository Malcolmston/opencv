package threshold2

import (
	"errors"
	"math"

	cv "github.com/malcolmston/opencv"
)

// KapurThreshold returns the threshold that maximises the sum of the Shannon
// entropies of the foreground and background classes (Kapur, Sahoo and Wong's
// maximum-entropy method).
func KapurThreshold(src *cv.Mat) (int, error) {
	bins, total, err := threshold2binsOf(src)
	if err != nil {
		return 0, err
	}
	if total == 0 {
		return 0, nil
	}
	var norm [256]float64
	tot := float64(total)
	for i := 0; i < 256; i++ {
		norm[i] = float64(bins[i]) / tot
	}
	var p1 [256]float64
	p1[0] = norm[0]
	for i := 1; i < 256; i++ {
		p1[i] = p1[i-1] + norm[i]
	}

	first := 0
	for i := 0; i < 256; i++ {
		if p1[i] > 1e-16 {
			first = i
			break
		}
	}
	last := 255
	for i := 255; i >= first; i-- {
		if (1.0 - p1[i]) > 1e-16 {
			last = i
			break
		}
	}

	threshold := first
	maxEnt := math.Inf(-1)
	for it := first; it <= last; it++ {
		pb := p1[it]
		pf := 1.0 - pb
		if pb <= 0 || pf <= 0 {
			continue
		}
		var entBack float64
		for i := 0; i <= it; i++ {
			if bins[i] != 0 {
				q := norm[i] / pb
				entBack -= q * math.Log(q)
			}
		}
		var entObj float64
		for i := it + 1; i < 256; i++ {
			if bins[i] != 0 {
				q := norm[i] / pf
				entObj -= q * math.Log(q)
			}
		}
		if tot := entBack + entObj; tot > maxEnt {
			maxEnt = tot
			threshold = it
		}
	}
	return threshold, nil
}

// Kapur binarizes src using the threshold from [KapurThreshold].
func Kapur(src *cv.Mat, p Polarity) (*cv.Mat, int, error) {
	t, err := KapurThreshold(src)
	if err != nil {
		return nil, 0, err
	}
	dst, _ := Binarize(src, t, p)
	return dst, t, nil
}

// YenThreshold returns the threshold chosen by Yen, Chang and Chang's maximum
// correlation criterion, an entropic method that favours compact classes.
func YenThreshold(src *cv.Mat) (int, error) {
	bins, total, err := threshold2binsOf(src)
	if err != nil {
		return 0, err
	}
	if total == 0 {
		return 0, nil
	}
	var norm [256]float64
	tot := float64(total)
	for i := 0; i < 256; i++ {
		norm[i] = float64(bins[i]) / tot
	}
	var p1, p1sq, p2sq [256]float64
	p1[0] = norm[0]
	for i := 1; i < 256; i++ {
		p1[i] = p1[i-1] + norm[i]
	}
	p1sq[0] = norm[0] * norm[0]
	for i := 1; i < 256; i++ {
		p1sq[i] = p1sq[i-1] + norm[i]*norm[i]
	}
	p2sq[255] = 0
	for i := 254; i >= 0; i-- {
		p2sq[i] = p2sq[i+1] + norm[i+1]*norm[i+1]
	}

	threshold := 0
	maxCrit := math.Inf(-1)
	for it := 0; it < 256; it++ {
		var a, b float64
		if p1sq[it]*p2sq[it] > 0 {
			a = math.Log(p1sq[it] * p2sq[it])
		}
		if p1[it]*(1.0-p1[it]) > 0 {
			b = math.Log(p1[it] * (1.0 - p1[it]))
		}
		crit := -1.0*a + 2*b
		if crit > maxCrit {
			maxCrit = crit
			threshold = it
		}
	}
	return threshold, nil
}

// Yen binarizes src using the threshold from [YenThreshold].
func Yen(src *cv.Mat, p Polarity) (*cv.Mat, int, error) {
	t, err := YenThreshold(src)
	if err != nil {
		return nil, 0, err
	}
	dst, _ := Binarize(src, t, p)
	return dst, t, nil
}

// MultiKapur returns classes-1 ascending thresholds that partition the
// intensity range into the requested number of classes by maximising the total
// Shannon entropy summed over all classes (the multi-level Kapur method).
// classes must be at least 2 and at most 6.
func MultiKapur(src *cv.Mat, classes int) ([]int, error) {
	if classes < 2 || classes > 6 {
		return nil, errors.New("threshold2: MultiKapur classes must be in [2,6]")
	}
	bins, total, err := threshold2binsOf(src)
	if err != nil {
		return nil, err
	}
	if total == 0 {
		return make([]int, classes-1), nil
	}
	var norm [256]float64
	tot := float64(total)
	for i := 0; i < 256; i++ {
		norm[i] = float64(bins[i]) / tot
	}
	// classEntropy returns the Shannon entropy of the normalised histogram
	// over the half-open bin range [lo, hi].
	classEntropy := func(lo, hi int) float64 {
		var w float64
		for i := lo; i < hi; i++ {
			w += norm[i]
		}
		if w <= 0 {
			return 0
		}
		var e float64
		for i := lo; i < hi; i++ {
			if norm[i] > 0 {
				q := norm[i] / w
				e -= q * math.Log(q)
			}
		}
		return e
	}

	k := classes - 1
	thresholds := make([]int, k)
	best := make([]int, k)
	bestEnt := math.Inf(-1)

	var recurse func(depth, start int, acc float64)
	recurse = func(depth, start int, acc float64) {
		if depth == k {
			total := acc + classEntropy(thresholds[k-1], 256)
			if total > bestEnt {
				bestEnt = total
				copy(best, thresholds)
			}
			return
		}
		prev := 0
		if depth > 0 {
			prev = thresholds[depth-1]
		}
		for t := start; t < 256-(k-depth-1); t++ {
			thresholds[depth] = t
			recurse(depth+1, t+1, acc+classEntropy(prev, t+1))
		}
	}
	recurse(0, 1, 0)
	return best, nil
}

// MultiKapurQuantize applies [MultiKapur] and returns a single-channel image in
// which every pixel is mapped to the mid-grey of the class it falls into,
// together with the thresholds used. It is a posterization of src that keeps
// class boundaries at the maximum-entropy levels.
func MultiKapurQuantize(src *cv.Mat, classes int) (*cv.Mat, []int, error) {
	ts, err := MultiKapur(src, classes)
	if err != nil {
		return nil, nil, err
	}
	gray, rows, cols, err := threshold2gray(src)
	if err != nil {
		return nil, nil, err
	}
	dst := threshold2quantize(gray, rows, cols, ts)
	return dst, ts, nil
}

// threshold2quantize maps each grey pixel to the mid-level of its class, where
// classes are delimited by the ascending thresholds ts.
func threshold2quantize(gray []uint8, rows, cols int, ts []int) *cv.Mat {
	bounds := make([]int, 0, len(ts)+2)
	bounds = append(bounds, 0)
	bounds = append(bounds, ts...)
	bounds = append(bounds, 255)
	levels := make([]uint8, len(ts)+1)
	for c := 0; c < len(levels); c++ {
		lo := bounds[c]
		hi := bounds[c+1]
		levels[c] = uint8((lo + hi) / 2)
	}
	dst := cv.NewMat(rows, cols, 1)
	for i, v := range gray {
		class := len(ts)
		for c, t := range ts {
			if int(v) <= t {
				class = c
				break
			}
		}
		dst.Data[i] = levels[class]
	}
	return dst
}
