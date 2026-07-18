package threshold2

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// Histogram is the 256-bin intensity histogram of a single-channel image.
// Bins[v] holds the number of pixels with grey level v, and Total is the sum
// of all bins.
type Histogram struct {
	// Bins holds the per-level pixel counts.
	Bins [256]int
	// Total is the number of pixels counted (the sum of Bins).
	Total int
}

// ComputeHistogram builds the 256-bin intensity histogram of src. Colour input
// is reduced to luma with the Rec. 601 weights before counting. It returns
// [ErrEmpty] if src has no samples.
func ComputeHistogram(src *cv.Mat) (*Histogram, error) {
	gray, _, _, err := threshold2gray(src)
	if err != nil {
		return nil, err
	}
	return threshold2histFromGray(gray), nil
}

// threshold2histFromGray tallies a grey buffer into a Histogram.
func threshold2histFromGray(gray []uint8) *Histogram {
	h := &Histogram{Total: len(gray)}
	for _, v := range gray {
		h.Bins[v]++
	}
	return h
}

// Density returns the normalised histogram, where each entry is the fraction
// of pixels at that grey level. The entries sum to 1 unless the histogram is
// empty, in which case all entries are 0.
func (h *Histogram) Density() [256]float64 {
	var d [256]float64
	if h.Total == 0 {
		return d
	}
	inv := 1.0 / float64(h.Total)
	for i := 0; i < 256; i++ {
		d[i] = float64(h.Bins[i]) * inv
	}
	return d
}

// Mean returns the mean grey level of the histogram, or 0 for an empty
// histogram.
func (h *Histogram) Mean() float64 {
	if h.Total == 0 {
		return 0
	}
	var sum float64
	for i := 0; i < 256; i++ {
		sum += float64(i) * float64(h.Bins[i])
	}
	return sum / float64(h.Total)
}

// Variance returns the variance of the grey levels about the mean, or 0 for an
// empty histogram.
func (h *Histogram) Variance() float64 {
	if h.Total == 0 {
		return 0
	}
	m := h.Mean()
	var s float64
	for i := 0; i < 256; i++ {
		d := float64(i) - m
		s += d * d * float64(h.Bins[i])
	}
	return s / float64(h.Total)
}

// Cumulative returns the cumulative histogram, where entry v is the number of
// pixels with grey level at most v.
func (h *Histogram) Cumulative() [256]int {
	var c [256]int
	run := 0
	for i := 0; i < 256; i++ {
		run += h.Bins[i]
		c[i] = run
	}
	return c
}

// Peak returns the grey level of the most populated bin. Ties resolve to the
// lowest level. An empty histogram returns 0.
func (h *Histogram) Peak() int {
	best, bestCount := 0, -1
	for i := 0; i < 256; i++ {
		if h.Bins[i] > bestCount {
			bestCount = h.Bins[i]
			best = i
		}
	}
	return best
}

// Range returns the lowest and highest grey levels that have a non-zero count.
// For an empty histogram it returns (0, 0).
func (h *Histogram) Range() (first, last int) {
	first, last = 0, 0
	for i := 0; i < 256; i++ {
		if h.Bins[i] != 0 {
			first = i
			break
		}
	}
	for i := 255; i >= 0; i-- {
		if h.Bins[i] != 0 {
			last = i
			break
		}
	}
	return first, last
}

// Smoothed returns a new Histogram whose bins are the moving average of this
// one over a window of radius r (each output bin averages the 2r+1 input bins
// centred on it, clamped at the ends and rounded to the nearest integer). A
// radius of 0 returns a copy. It panics if r is negative.
func (h *Histogram) Smoothed(r int) *Histogram {
	if r < 0 {
		panic("threshold2: Histogram.Smoothed requires r >= 0")
	}
	out := &Histogram{Total: h.Total}
	if r == 0 {
		out.Bins = h.Bins
		return out
	}
	for i := 0; i < 256; i++ {
		lo := i - r
		if lo < 0 {
			lo = 0
		}
		hi := i + r
		if hi > 255 {
			hi = 255
		}
		sum := 0
		for j := lo; j <= hi; j++ {
			sum += h.Bins[j]
		}
		out.Bins[i] = int(math.Round(float64(sum) / float64(hi-lo+1)))
	}
	return out
}
