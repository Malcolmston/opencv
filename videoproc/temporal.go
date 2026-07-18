package videoproc

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// videoprocCheckStack panics unless frames is non-empty and every frame shares
// the dimensions and channel count of the first.
func videoprocCheckStack(fn string, frames []*cv.Mat) {
	if len(frames) == 0 {
		panic("videoproc: " + fn + " requires at least one frame")
	}
	f0 := frames[0]
	if f0 == nil || f0.Empty() {
		panic("videoproc: " + fn + " frame 0 is empty")
	}
	for i, f := range frames {
		if f == nil || f.Empty() {
			panic("videoproc: " + fn + " frame is empty")
		}
		if !videoprocSameSize(f, f0) {
			panic("videoproc: " + fn + " frame size/channel mismatch at index " + itoa(i))
		}
	}
}

// itoa is a tiny base-10 formatter used only in panic messages.
func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	neg := n < 0
	if neg {
		n = -n
	}
	var buf [20]byte
	i := len(buf)
	for n > 0 {
		i--
		buf[i] = byte('0' + n%10)
		n /= 10
	}
	if neg {
		i--
		buf[i] = '-'
	}
	return string(buf[i:])
}

// TemporalAverage returns the per-sample mean of a stack of frames (all the same
// size and channel count), rounded and clamped to 8-bit. It is the standard way
// to build a clean background plate from a jittery static camera. It panics on
// an empty stack or a size mismatch.
func TemporalAverage(frames []*cv.Mat) *cv.Mat {
	videoprocCheckStack("TemporalAverage", frames)
	f0 := frames[0]
	out := cv.NewMat(f0.Rows, f0.Cols, f0.Channels)
	inv := 1.0 / float64(len(frames))
	for i := range out.Data {
		var sum float64
		for _, f := range frames {
			sum += float64(f.Data[i])
		}
		out.Data[i] = videoprocClampU8(sum*inv + 0.5)
	}
	return out
}

// TemporalMedian returns the per-sample median of a stack of frames. The median
// removes transient objects (people, cars) that occupy any pixel for less than
// half the sequence, yielding a robust background estimate. It panics on an
// empty stack or a size mismatch.
func TemporalMedian(frames []*cv.Mat) *cv.Mat {
	videoprocCheckStack("TemporalMedian", frames)
	f0 := frames[0]
	out := cv.NewMat(f0.Rows, f0.Cols, f0.Channels)
	vals := make([]uint8, len(frames))
	for i := range out.Data {
		for k, f := range frames {
			vals[k] = f.Data[i]
		}
		out.Data[i] = videoprocMedianU8(vals)
	}
	return out
}

// TemporalMinimum returns the per-sample minimum across a stack of frames, the
// darkest value each sample takes over time. It panics on an empty stack or a
// size mismatch.
func TemporalMinimum(frames []*cv.Mat) *cv.Mat {
	videoprocCheckStack("TemporalMinimum", frames)
	f0 := frames[0]
	out := cv.NewMat(f0.Rows, f0.Cols, f0.Channels)
	for i := range out.Data {
		m := frames[0].Data[i]
		for _, f := range frames[1:] {
			if f.Data[i] < m {
				m = f.Data[i]
			}
		}
		out.Data[i] = m
	}
	return out
}

// TemporalMaximum returns the per-sample maximum across a stack of frames, the
// brightest value each sample takes over time. It panics on an empty stack or a
// size mismatch.
func TemporalMaximum(frames []*cv.Mat) *cv.Mat {
	videoprocCheckStack("TemporalMaximum", frames)
	f0 := frames[0]
	out := cv.NewMat(f0.Rows, f0.Cols, f0.Channels)
	for i := range out.Data {
		m := frames[0].Data[i]
		for _, f := range frames[1:] {
			if f.Data[i] > m {
				m = f.Data[i]
			}
		}
		out.Data[i] = m
	}
	return out
}

// TemporalGaussian returns a Gaussian-weighted temporal average of a stack of
// frames: frame k receives weight exp(-((k-c)²)/(2σ²)) where c is the centre
// index (len/2). This smooths a sequence along time, attenuating noise while
// preserving the central frame more than a flat average would. sigma must be
// positive. It panics on an empty stack, a size mismatch or sigma <= 0.
func TemporalGaussian(frames []*cv.Mat, sigma float64) *cv.Mat {
	videoprocCheckStack("TemporalGaussian", frames)
	if sigma <= 0 {
		panic("videoproc: TemporalGaussian requires sigma > 0")
	}
	f0 := frames[0]
	n := len(frames)
	c := float64(n-1) / 2
	weights := make([]float64, n)
	var wsum float64
	for k := 0; k < n; k++ {
		d := float64(k) - c
		weights[k] = math.Exp(-(d * d) / (2 * sigma * sigma))
		wsum += weights[k]
	}
	out := cv.NewMat(f0.Rows, f0.Cols, f0.Channels)
	invw := 1.0 / wsum
	for i := range out.Data {
		var sum float64
		for k, f := range frames {
			sum += weights[k] * float64(f.Data[i])
		}
		out.Data[i] = videoprocClampU8(sum*invw + 0.5)
	}
	return out
}

// ExponentialMovingAverage is an online per-pixel low-pass temporal filter:
// state = (1-Alpha)*state + Alpha*frame. It denoises a live stream without
// storing a window of frames, trading lag for smoothness as Alpha shrinks.
type ExponentialMovingAverage struct {
	// Alpha is the update weight in (0,1]; smaller means smoother and laggier.
	Alpha float64

	state *cv.FloatMat
	chans int
	rows  int
	cols  int
}

// NewExponentialMovingAverage returns an EMA filter with the given update weight
// alpha in (0,1]. It panics on an out-of-range alpha.
func NewExponentialMovingAverage(alpha float64) *ExponentialMovingAverage {
	if alpha <= 0 || alpha > 1 {
		panic("videoproc: NewExponentialMovingAverage requires alpha in (0,1]")
	}
	return &ExponentialMovingAverage{Alpha: alpha}
}

// Update folds frame into the running state and returns the current smoothed
// frame as a fresh Mat with the same shape as the input. The first frame
// initialises the state and is returned unchanged. Frame dimensions and channel
// count must stay constant across calls.
func (e *ExponentialMovingAverage) Update(frame *cv.Mat) *cv.Mat {
	if frame == nil || frame.Empty() {
		panic("videoproc: ExponentialMovingAverage.Update requires a non-empty frame")
	}
	if e.state == nil {
		e.rows, e.cols, e.chans = frame.Rows, frame.Cols, frame.Channels
		e.state = cv.NewFloatMat(frame.Rows, frame.Cols*frame.Channels)
		for i := range frame.Data {
			e.state.Data[i] = float64(frame.Data[i])
		}
	} else {
		if frame.Rows != e.rows || frame.Cols != e.cols || frame.Channels != e.chans {
			panic("videoproc: ExponentialMovingAverage frame shape changed")
		}
		for i := range frame.Data {
			e.state.Data[i] = (1-e.Alpha)*e.state.Data[i] + e.Alpha*float64(frame.Data[i])
		}
	}
	return e.Value()
}

// Value returns the current smoothed frame, or nil before the first Update.
func (e *ExponentialMovingAverage) Value() *cv.Mat {
	if e.state == nil {
		return nil
	}
	out := cv.NewMat(e.rows, e.cols, e.chans)
	for i := range out.Data {
		out.Data[i] = videoprocClampU8(e.state.Data[i] + 0.5)
	}
	return out
}

// MovingWindowFilter is an online temporal filter that keeps the most recent
// Size frames in a ring buffer and, on each push, returns their per-sample
// average. Unlike the exponential filter it weights every frame in the window
// equally and forgets frames abruptly once they leave the window.
type MovingWindowFilter struct {
	size   int
	frames [][]uint8
	next   int
	filled int
	rows   int
	cols   int
	chans  int
}

// NewMovingWindowFilter returns a moving-window averaging filter over the last
// size frames (size >= 1). It panics if size < 1.
func NewMovingWindowFilter(size int) *MovingWindowFilter {
	if size < 1 {
		panic("videoproc: NewMovingWindowFilter requires size >= 1")
	}
	return &MovingWindowFilter{size: size, frames: make([][]uint8, size)}
}

// Push inserts frame into the window and returns the average of all frames
// currently in the window (including the one just pushed) as a fresh Mat. Frame
// dimensions and channel count must stay constant across calls.
func (w *MovingWindowFilter) Push(frame *cv.Mat) *cv.Mat {
	if frame == nil || frame.Empty() {
		panic("videoproc: MovingWindowFilter.Push requires a non-empty frame")
	}
	if w.filled == 0 {
		w.rows, w.cols, w.chans = frame.Rows, frame.Cols, frame.Channels
	} else if frame.Rows != w.rows || frame.Cols != w.cols || frame.Channels != w.chans {
		panic("videoproc: MovingWindowFilter frame shape changed")
	}
	cp := make([]uint8, len(frame.Data))
	copy(cp, frame.Data)
	w.frames[w.next] = cp
	w.next = (w.next + 1) % w.size
	if w.filled < w.size {
		w.filled++
	}
	out := cv.NewMat(w.rows, w.cols, w.chans)
	inv := 1.0 / float64(w.filled)
	for i := range out.Data {
		var sum float64
		for k := 0; k < w.filled; k++ {
			sum += float64(w.frames[k][i])
		}
		out.Data[i] = videoprocClampU8(sum*inv + 0.5)
	}
	return out
}

// Full reports whether the window has accumulated Size frames.
func (w *MovingWindowFilter) Full() bool {
	return w.filled == w.size
}
