package segmentation

import (
	"fmt"

	cv "github.com/malcolmston/opencv"
)

// MultiOtsu computes classes-1 grey-level thresholds that partition the
// intensity histogram of img into classes groups by maximising the between-class
// variance, the multi-level generalisation of Otsu's method (Liao, Chen & Chung,
// 2001). The returned thresholds are strictly increasing values in [1, 255];
// class c contains the intensities t[c-1] <= v < t[c] with the usual open ends.
//
// img must be single-channel and classes must be in [2, 5]; the exhaustive
// search over threshold tuples is exact for that range. It panics otherwise, or
// if img is empty.
func MultiOtsu(img *cv.Mat, classes int) []int {
	if img.Empty() {
		panic("segmentation: MultiOtsu on empty image")
	}
	if img.Channels != 1 {
		panic("segmentation: MultiOtsu requires a single-channel image")
	}
	if classes < 2 || classes > 5 {
		panic(fmt.Sprintf("segmentation: MultiOtsu supports 2..5 classes, got %d", classes))
	}

	// Histogram and prefix sums of weight and weighted intensity, so the mean of
	// any intensity band [a, b) is O(1) to evaluate.
	var hist [256]float64
	for _, v := range img.Data {
		hist[v]++
	}
	total := float64(len(img.Data))
	var p [257]float64 // cumulative probability
	var s [257]float64 // cumulative weighted intensity
	for i := 0; i < 256; i++ {
		p[i+1] = p[i] + hist[i]/total
		s[i+1] = s[i] + float64(i)*hist[i]/total
	}
	globalMean := s[256]

	// Between-class variance contributed by the band [a, b): w*(mean-global)^2.
	bandVar := func(a, b int) float64 {
		w := p[b] - p[a]
		if w <= 0 {
			return 0
		}
		mean := (s[b] - s[a]) / w
		d := mean - globalMean
		return w * d * d
	}

	nThresh := classes - 1
	best := make([]int, nThresh)
	cur := make([]int, nThresh)
	bestVar := -1.0

	// Recursively place each threshold; boundaries are 0 and 256 with the chosen
	// thresholds in between.
	var search func(depth, start int)
	search = func(depth, start int) {
		if depth == nThresh {
			bounds := make([]int, 0, classes+1)
			bounds = append(bounds, 0)
			bounds = append(bounds, cur...)
			bounds = append(bounds, 256)
			v := 0.0
			for c := 0; c < classes; c++ {
				v += bandVar(bounds[c], bounds[c+1])
			}
			if v > bestVar {
				bestVar = v
				copy(best, cur)
			}
			return
		}
		// Leave room for the remaining thresholds (each strictly increasing).
		for t := start; t <= 256-(nThresh-depth); t++ {
			cur[depth] = t
			search(depth+1, t+1)
		}
	}
	search(0, 1)
	return best
}

// MultiOtsuThreshold applies [MultiOtsu] and returns both the chosen thresholds
// and a single-channel [cv.Mat] whose pixels hold the class index (0 for the
// darkest class up to classes-1 for the brightest) of each input pixel. It is a
// convenience wrapper that turns the multi-level thresholds into a segmented
// label image.
//
// img must be single-channel and classes in [2, 5]. It panics otherwise.
func MultiOtsuThreshold(img *cv.Mat, classes int) (*cv.Mat, []int) {
	thresh := MultiOtsu(img, classes)
	out := cv.NewMat(img.Rows, img.Cols, 1)
	for i, v := range img.Data {
		class := 0
		for _, t := range thresh {
			if int(v) >= t {
				class++
			} else {
				break
			}
		}
		out.Data[i] = uint8(class)
	}
	return out, thresh
}
