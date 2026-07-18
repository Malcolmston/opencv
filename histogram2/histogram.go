package histogram2

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// Histogram1D is a one-dimensional histogram of sample values distributed over
// a half-open value range [Min, Max) split into Bins equal-width bins. The
// count (or, after normalisation, the weight) of bin i is stored in
// Counts[i].
type Histogram1D struct {
	// Counts holds the per-bin totals, length BinCount.
	Counts []float64
	// BinCount is the number of bins.
	BinCount int
	// Min is the inclusive lower edge of the value range.
	Min float64
	// Max is the exclusive upper edge of the value range.
	Max float64
}

// NewHistogram1D allocates a zeroed one-dimensional histogram with binCount
// bins spanning the half-open range [min, max). It panics if binCount is not
// positive or max is not greater than min.
func NewHistogram1D(binCount int, min, max float64) *Histogram1D {
	if binCount <= 0 {
		panic("histogram2: NewHistogram1D requires a positive bin count")
	}
	if !(max > min) {
		panic("histogram2: NewHistogram1D requires max > min")
	}
	return &Histogram1D{
		Counts:   make([]float64, binCount),
		BinCount: binCount,
		Min:      min,
		Max:      max,
	}
}

// BinIndex maps a sample value to its bin index, clamping values below Min to
// bin 0 and values at or above Max to the final bin.
func (h *Histogram1D) BinIndex(value float64) int {
	frac := (value - h.Min) / (h.Max - h.Min)
	idx := int(frac * float64(h.BinCount))
	if idx < 0 {
		return 0
	}
	if idx >= h.BinCount {
		return h.BinCount - 1
	}
	return idx
}

// At returns the count stored in bin i. It panics if i is out of range.
func (h *Histogram1D) At(i int) float64 {
	return h.Counts[i]
}

// Total returns the sum of all bin counts.
func (h *Histogram1D) Total() float64 {
	var s float64
	for _, v := range h.Counts {
		s += v
	}
	return s
}

// Clone returns a deep copy of the histogram with its own backing storage.
func (h *Histogram1D) Clone() *Histogram1D {
	c := &Histogram1D{
		Counts:   make([]float64, len(h.Counts)),
		BinCount: h.BinCount,
		Min:      h.Min,
		Max:      h.Max,
	}
	copy(c.Counts, h.Counts)
	return c
}

// Normalize rescales the histogram in place so that its bin counts sum to one,
// turning it into a probability distribution. A histogram whose total is zero
// is left unchanged.
func (h *Histogram1D) Normalize() {
	total := h.Total()
	if total == 0 {
		return
	}
	for i := range h.Counts {
		h.Counts[i] /= total
	}
}

// NormalizeMax rescales the histogram in place so that its largest bin equals
// scale. A histogram whose maximum is zero is left unchanged.
func (h *Histogram1D) NormalizeMax(scale float64) {
	mx := 0.0
	for _, v := range h.Counts {
		if v > mx {
			mx = v
		}
	}
	if mx == 0 {
		return
	}
	for i := range h.Counts {
		h.Counts[i] = h.Counts[i] / mx * scale
	}
}

// Cumulative returns a new histogram whose bin i holds the sum of bins 0..i of
// the receiver — the (unnormalised) cumulative distribution.
func (h *Histogram1D) Cumulative() *Histogram1D {
	c := NewHistogram1D(h.BinCount, h.Min, h.Max)
	acc := 0.0
	for i, v := range h.Counts {
		acc += v
		c.Counts[i] = acc
	}
	return c
}

// Peak returns the index of the bin with the largest count. For an all-zero
// histogram it returns 0.
func (h *Histogram1D) Peak() int {
	best := 0
	mx := h.Counts[0]
	for i, v := range h.Counts {
		if v > mx {
			mx = v
			best = i
		}
	}
	return best
}

// Mean returns the count-weighted mean of the bin-centre values. It returns 0
// for an empty histogram.
func (h *Histogram1D) Mean() float64 {
	total := h.Total()
	if total == 0 {
		return 0
	}
	width := (h.Max - h.Min) / float64(h.BinCount)
	var s float64
	for i, v := range h.Counts {
		centre := h.Min + (float64(i)+0.5)*width
		s += centre * v
	}
	return s / total
}

// Variance returns the count-weighted variance of the bin-centre values. It
// returns 0 for an empty histogram.
func (h *Histogram1D) Variance() float64 {
	total := h.Total()
	if total == 0 {
		return 0
	}
	mean := h.Mean()
	width := (h.Max - h.Min) / float64(h.BinCount)
	var s float64
	for i, v := range h.Counts {
		centre := h.Min + (float64(i)+0.5)*width
		d := centre - mean
		s += d * d * v
	}
	return s / total
}

// Entropy returns the Shannon entropy of the histogram in bits, treating the
// normalised bin counts as a probability distribution. It returns 0 for an
// empty histogram.
func (h *Histogram1D) Entropy() float64 {
	total := h.Total()
	if total == 0 {
		return 0
	}
	var e float64
	for _, v := range h.Counts {
		if v > 0 {
			p := v / total
			e -= p * math.Log2(p)
		}
	}
	return e
}

// CalcHist1D computes a one-dimensional histogram of the given channel of src
// over the value range [0, 256) split into binCount bins. It returns
// [ErrEmptyImage] if src is empty, [ErrChannelRange] if channel is out of
// range and [ErrBinCount] if binCount is not positive.
func CalcHist1D(src *cv.Mat, channel, binCount int) (*Histogram1D, error) {
	return CalcHist1DMasked(src, channel, binCount, nil)
}

// CalcHist1DMasked is like [CalcHist1D] but only counts pixels for which the
// corresponding sample of mask is non-zero. A nil mask counts every pixel. The
// mask, when non-nil, must have the same number of pixels as src and at least
// one channel; only its first channel is consulted.
func CalcHist1DMasked(src *cv.Mat, channel, binCount int, mask *cv.Mat) (*Histogram1D, error) {
	if src.Empty() {
		return nil, ErrEmptyImage
	}
	if channel < 0 || channel >= src.Channels {
		return nil, ErrChannelRange
	}
	if binCount <= 0 {
		return nil, ErrBinCount
	}
	if mask != nil {
		if mask.Empty() {
			return nil, ErrEmptyImage
		}
		if mask.Rows != src.Rows || mask.Cols != src.Cols {
			return nil, ErrSizeMismatch
		}
	}
	h := NewHistogram1D(binCount, 0, 256)
	total := src.Total()
	for p := 0; p < total; p++ {
		if mask != nil && mask.Data[p*mask.Channels] == 0 {
			continue
		}
		v := float64(src.Data[p*src.Channels+channel])
		h.Counts[h.BinIndex(v)]++
	}
	return h, nil
}

// CumulativeDistribution returns the normalised cumulative distribution
// function of h as a slice of length h.BinCount, where entry i is the fraction
// of counts in bins 0..i. For an empty histogram it returns an all-zero slice.
func CumulativeDistribution(h *Histogram1D) []float64 {
	out := make([]float64, h.BinCount)
	total := h.Total()
	if total == 0 {
		return out
	}
	acc := 0.0
	for i, v := range h.Counts {
		acc += v
		out[i] = acc / total
	}
	return out
}
