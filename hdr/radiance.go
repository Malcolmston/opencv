package hdr

import (
	"fmt"
	"math"

	cv "github.com/malcolmston/opencv"
)

// Radiance is a dense, row-major, channel-interleaved image of float64 linear
// radiance values — the high-dynamic-range counterpart of [cv.Mat]. The value
// for row y, column x and channel c is stored at index (y*Cols+x)*Channels + c.
//
// The root cv package's [cv.FloatMat] is single-channel; Radiance exists so a
// three-channel (RGB) radiance map can be represented as one value. A single
// channel can be extracted as a cv.FloatMat with [Radiance.ChannelFloatMat].
type Radiance struct {
	// Rows is the image height.
	Rows int
	// Cols is the image width.
	Cols int
	// Channels is the number of samples per pixel (typically 1 or 3).
	Channels int
	// Data holds Rows*Cols*Channels samples in row-major, channel-interleaved
	// order.
	Data []float64
}

// NewRadiance allocates a zero-filled Radiance. It panics if any dimension is
// not positive.
func NewRadiance(rows, cols, channels int) *Radiance {
	if rows <= 0 || cols <= 0 || channels <= 0 {
		panic(fmt.Sprintf("hdr: NewRadiance requires positive dimensions, got rows=%d cols=%d channels=%d", rows, cols, channels))
	}
	return &Radiance{Rows: rows, Cols: cols, Channels: channels, Data: make([]float64, rows*cols*channels)}
}

func (r *Radiance) index(y, x int) int { return (y*r.Cols + x) * r.Channels }

// At returns the radiance sample at row y, column x, channel c.
func (r *Radiance) At(y, x, c int) float64 { return r.Data[r.index(y, x)+c] }

// Set stores value at row y, column x, channel c.
func (r *Radiance) Set(y, x, c int, value float64) { r.Data[r.index(y, x)+c] = value }

// Clone returns a deep copy with independent backing storage.
func (r *Radiance) Clone() *Radiance {
	out := NewRadiance(r.Rows, r.Cols, r.Channels)
	copy(out.Data, r.Data)
	return out
}

// ChannelFloatMat extracts channel c as a single-channel [cv.FloatMat]. It
// panics if c is out of range.
func (r *Radiance) ChannelFloatMat(c int) *cv.FloatMat {
	if c < 0 || c >= r.Channels {
		panic(fmt.Sprintf("hdr: ChannelFloatMat channel %d out of range for %d channels", c, r.Channels))
	}
	out := cv.NewFloatMat(r.Rows, r.Cols)
	for p := 0; p < r.Rows*r.Cols; p++ {
		out.Data[p] = r.Data[p*r.Channels+c]
	}
	return out
}

// luminance returns a single-channel plane of relative luminance. For a
// three-channel image it uses the Rec.709 weights on the linear samples; for a
// single-channel image it copies the channel.
func (r *Radiance) luminance() *plane {
	p := newPlane(r.Rows, r.Cols)
	if r.Channels == 1 {
		copy(p.data, r.Data)
		return p
	}
	for i := 0; i < r.Rows*r.Cols; i++ {
		base := i * r.Channels
		p.data[i] = 0.2126*r.Data[base+0] + 0.7152*r.Data[base+1] + 0.0722*r.Data[base+2]
	}
	return p
}

// plane is an internal single-channel float image used by the blur, pyramid
// and tonemapping kernels. It mirrors cv.FloatMat but stays package-local so
// the numeric helpers can operate on it without depending on cv internals.
type plane struct {
	rows, cols int
	data       []float64
}

func newPlane(rows, cols int) *plane {
	return &plane{rows: rows, cols: cols, data: make([]float64, rows*cols)}
}

func (p *plane) at(y, x int) float64     { return p.data[y*p.cols+x] }
func (p *plane) set(y, x int, v float64) { p.data[y*p.cols+x] = v }

func (p *plane) clone() *plane {
	q := newPlane(p.rows, p.cols)
	copy(q.data, p.data)
	return q
}

// atReflect samples with refl(mirror) border handling.
func (p *plane) atReflect(y, x int) float64 {
	y = reflect(y, p.rows)
	x = reflect(x, p.cols)
	return p.data[y*p.cols+x]
}

func reflect(i, n int) int {
	if n == 1 {
		return 0
	}
	for i < 0 || i >= n {
		if i < 0 {
			i = -i - 1
		}
		if i >= n {
			i = 2*n - i - 1
		}
	}
	return i
}

// gaussianKernel returns a normalized 1-D Gaussian of the given standard
// deviation. The radius is ceil(3*sigma).
func gaussianKernel(sigma float64) []float64 {
	if sigma <= 0 {
		return []float64{1}
	}
	radius := int(math.Ceil(3 * sigma))
	k := make([]float64, 2*radius+1)
	var sum float64
	for i := -radius; i <= radius; i++ {
		v := math.Exp(-float64(i*i) / (2 * sigma * sigma))
		k[i+radius] = v
		sum += v
	}
	for i := range k {
		k[i] /= sum
	}
	return k
}

// blur applies a separable Gaussian blur with mirror borders.
func (p *plane) blur(sigma float64) *plane {
	k := gaussianKernel(sigma)
	radius := len(k) / 2
	tmp := newPlane(p.rows, p.cols)
	// Horizontal pass.
	for y := 0; y < p.rows; y++ {
		for x := 0; x < p.cols; x++ {
			var acc float64
			for t := -radius; t <= radius; t++ {
				acc += k[t+radius] * p.atReflect(y, x+t)
			}
			tmp.set(y, x, acc)
		}
	}
	out := newPlane(p.rows, p.cols)
	// Vertical pass.
	for y := 0; y < p.rows; y++ {
		for x := 0; x < p.cols; x++ {
			var acc float64
			for t := -radius; t <= radius; t++ {
				acc += k[t+radius] * tmp.atReflect(y+t, x)
			}
			out.set(y, x, acc)
		}
	}
	return out
}

// laplacian returns the absolute Laplacian (a 3x3 discrete filter) magnitude,
// used as the contrast measure for exposure fusion.
func (p *plane) laplacianAbs() *plane {
	out := newPlane(p.rows, p.cols)
	for y := 0; y < p.rows; y++ {
		for x := 0; x < p.cols; x++ {
			v := 4*p.at(y, x) - p.atReflect(y-1, x) - p.atReflect(y+1, x) - p.atReflect(y, x-1) - p.atReflect(y, x+1)
			out.set(y, x, math.Abs(v))
		}
	}
	return out
}

// clamp8 rounds to nearest and clamps into [0,255].
func clamp8(v float64) uint8 {
	v = math.Round(v)
	if v <= 0 {
		return 0
	}
	if v >= 255 {
		return 255
	}
	return uint8(v)
}

// clamp01 clamps into [0,1].
func clamp01(v float64) float64 {
	if v < 0 {
		return 0
	}
	if v > 1 {
		return 1
	}
	return v
}
