package histogram2

import cv "github.com/malcolmston/opencv"

// Histogram2D is a two-dimensional joint histogram of two channels. Bin (x, y)
// covers the value range [0,256) split into BinsX bins along the first channel
// and BinsY bins along the second. Counts are stored row-major in a flat slice
// of length BinsX*BinsY.
type Histogram2D struct {
	// Counts holds the joint counts in row-major order: the count for bin
	// (x, y) is at index y*BinsX + x.
	Counts []float64
	// BinsX is the number of bins along the first (x) channel.
	BinsX int
	// BinsY is the number of bins along the second (y) channel.
	BinsY int
}

// NewHistogram2D allocates a zeroed two-dimensional histogram. It panics if
// either bin count is not positive.
func NewHistogram2D(binsX, binsY int) *Histogram2D {
	if binsX <= 0 || binsY <= 0 {
		panic("histogram2: NewHistogram2D requires positive bin counts")
	}
	return &Histogram2D{
		Counts: make([]float64, binsX*binsY),
		BinsX:  binsX,
		BinsY:  binsY,
	}
}

// At returns the count in bin (x, y). It panics if the indices are out of
// range.
func (h *Histogram2D) At(x, y int) float64 {
	if x < 0 || x >= h.BinsX || y < 0 || y >= h.BinsY {
		panic("histogram2: Histogram2D.At index out of range")
	}
	return h.Counts[y*h.BinsX+x]
}

// Set stores value in bin (x, y). It panics if the indices are out of range.
func (h *Histogram2D) Set(x, y int, value float64) {
	if x < 0 || x >= h.BinsX || y < 0 || y >= h.BinsY {
		panic("histogram2: Histogram2D.Set index out of range")
	}
	h.Counts[y*h.BinsX+x] = value
}

// Total returns the sum of all bin counts.
func (h *Histogram2D) Total() float64 {
	var s float64
	for _, v := range h.Counts {
		s += v
	}
	return s
}

// Clone returns a deep copy of the histogram with its own backing storage.
func (h *Histogram2D) Clone() *Histogram2D {
	c := NewHistogram2D(h.BinsX, h.BinsY)
	copy(c.Counts, h.Counts)
	return c
}

// Normalize rescales the histogram in place so that its bin counts sum to one.
// A histogram whose total is zero is left unchanged.
func (h *Histogram2D) Normalize() {
	total := h.Total()
	if total == 0 {
		return
	}
	for i := range h.Counts {
		h.Counts[i] /= total
	}
}

// CalcHist2D computes the joint histogram of channels chX and chY of src over
// the range [0,256), using binsX and binsY bins respectively. It returns
// [ErrEmptyImage], [ErrChannelRange] or [ErrBinCount] on invalid input.
func CalcHist2D(src *cv.Mat, chX, chY, binsX, binsY int) (*Histogram2D, error) {
	if src.Empty() {
		return nil, ErrEmptyImage
	}
	if chX < 0 || chX >= src.Channels || chY < 0 || chY >= src.Channels {
		return nil, ErrChannelRange
	}
	if binsX <= 0 || binsY <= 0 {
		return nil, ErrBinCount
	}
	h := NewHistogram2D(binsX, binsY)
	total := src.Total()
	for p := 0; p < total; p++ {
		vx := int(src.Data[p*src.Channels+chX]) * binsX / 256
		vy := int(src.Data[p*src.Channels+chY]) * binsY / 256
		h.Counts[vy*binsX+vx]++
	}
	return h, nil
}

// Histogram3D is a three-dimensional joint histogram of a three-channel image,
// with Bins bins per channel over the range [0,256). Counts are stored in a
// flat slice of length Bins*Bins*Bins indexed by z*Bins*Bins + y*Bins + x,
// where x, y and z are the bins of channels 0, 1 and 2.
type Histogram3D struct {
	// Counts holds the joint counts; the count for bin (x, y, z) is at index
	// (z*Bins+y)*Bins + x.
	Counts []float64
	// Bins is the number of bins along each channel.
	Bins int
}

// NewHistogram3D allocates a zeroed three-dimensional histogram with bins bins
// per channel. It panics if bins is not positive.
func NewHistogram3D(bins int) *Histogram3D {
	if bins <= 0 {
		panic("histogram2: NewHistogram3D requires a positive bin count")
	}
	return &Histogram3D{
		Counts: make([]float64, bins*bins*bins),
		Bins:   bins,
	}
}

// At returns the count in bin (x, y, z). It panics if any index is out of
// range.
func (h *Histogram3D) At(x, y, z int) float64 {
	if x < 0 || x >= h.Bins || y < 0 || y >= h.Bins || z < 0 || z >= h.Bins {
		panic("histogram2: Histogram3D.At index out of range")
	}
	return h.Counts[(z*h.Bins+y)*h.Bins+x]
}

// Set stores value in bin (x, y, z). It panics if any index is out of range.
func (h *Histogram3D) Set(x, y, z int, value float64) {
	if x < 0 || x >= h.Bins || y < 0 || y >= h.Bins || z < 0 || z >= h.Bins {
		panic("histogram2: Histogram3D.Set index out of range")
	}
	h.Counts[(z*h.Bins+y)*h.Bins+x] = value
}

// Total returns the sum of all bin counts.
func (h *Histogram3D) Total() float64 {
	var s float64
	for _, v := range h.Counts {
		s += v
	}
	return s
}

// Normalize rescales the histogram in place so that its bin counts sum to one.
// A histogram whose total is zero is left unchanged.
func (h *Histogram3D) Normalize() {
	total := h.Total()
	if total == 0 {
		return
	}
	for i := range h.Counts {
		h.Counts[i] /= total
	}
}

// CalcHist3D computes the joint histogram of the first three channels of src
// (interpreted as an RGB triple) using bins bins per channel over [0,256). It
// returns [ErrEmptyImage] if src is empty, [ErrChannelRange] if src has fewer
// than three channels and [ErrBinCount] if bins is not positive.
func CalcHist3D(src *cv.Mat, bins int) (*Histogram3D, error) {
	if src.Empty() {
		return nil, ErrEmptyImage
	}
	if src.Channels < 3 {
		return nil, ErrChannelRange
	}
	if bins <= 0 {
		return nil, ErrBinCount
	}
	h := NewHistogram3D(bins)
	total := src.Total()
	for p := 0; p < total; p++ {
		base := p * src.Channels
		x := int(src.Data[base]) * bins / 256
		y := int(src.Data[base+1]) * bins / 256
		z := int(src.Data[base+2]) * bins / 256
		h.Counts[(z*bins+y)*bins+x]++
	}
	return h, nil
}
