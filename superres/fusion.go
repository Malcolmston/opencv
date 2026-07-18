package superres

import (
	"math"
	"sort"

	cv "github.com/malcolmston/opencv"
)

// RegisterFrames estimates the sub-pixel translation of each frame relative to
// the first (reference) frame using iterated Lucas-Kanade
// ([EstimateShiftRefine]). The returned slice has one Shift2D per input frame;
// the reference frame's entry is the zero shift. Each shift is such that
// frame[i] ≈ Shift(frames[0], Dx, Dy). All frames must share dimensions and
// channel count. It panics if fewer than one frame is given or shapes differ.
func RegisterFrames(frames []*cv.Mat) []Shift2D {
	if len(frames) == 0 {
		panic("superres: RegisterFrames requires at least one frame")
	}
	shifts := make([]Shift2D, len(frames))
	ref := frames[0]
	for i := 1; i < len(frames); i++ {
		shifts[i] = EstimateShiftRefine(ref, frames[i], 8)
	}
	return shifts
}

// FuseAverage aligns each frame back onto the reference grid by its (negated)
// shift and averages them, producing a denoised image at the original
// resolution. shifts[i] is the displacement of frames[i] relative to the
// reference (as returned by [RegisterFrames]); pass nil to treat all frames as
// already aligned. All frames must share dimensions and channel count. It
// panics if no frames are given, shapes differ, or len(shifts) is neither zero
// nor len(frames).
func FuseAverage(frames []*cv.Mat, shifts []Shift2D) *cv.Mat {
	superresValidateFrames(frames, shifts)
	ref := frames[0]
	ch := ref.Channels
	acc := make([]float64, len(ref.Data))
	for i, f := range frames {
		aligned := f
		if shifts != nil && (shifts[i].Dx != 0 || shifts[i].Dy != 0) {
			// Undo the frame's shift to bring it onto the reference grid.
			aligned = Shift(f, -shifts[i].Dx, -shifts[i].Dy)
		}
		for j := range acc {
			acc[j] += float64(aligned.Data[j])
		}
	}
	out := cv.NewMat(ref.Rows, ref.Cols, ch)
	n := float64(len(frames))
	for j := range out.Data {
		out.Data[j] = superresClamp8(acc[j] / n)
	}
	return out
}

// FuseMedian is like [FuseAverage] but combines the aligned frames by their
// per-sample median instead of the mean, which rejects outliers (moving
// objects, hot pixels) far better than averaging at the cost of more
// computation. All frames must share dimensions and channel count. It panics
// under the same conditions as [FuseAverage].
func FuseMedian(frames []*cv.Mat, shifts []Shift2D) *cv.Mat {
	superresValidateFrames(frames, shifts)
	ref := frames[0]
	ch := ref.Channels
	aligned := make([]*cv.Mat, len(frames))
	for i, f := range frames {
		if shifts != nil && (shifts[i].Dx != 0 || shifts[i].Dy != 0) {
			aligned[i] = Shift(f, -shifts[i].Dx, -shifts[i].Dy)
		} else {
			aligned[i] = f
		}
	}
	out := cv.NewMat(ref.Rows, ref.Cols, ch)
	buf := make([]float64, len(frames))
	for j := range out.Data {
		for i := range aligned {
			buf[i] = float64(aligned[i].Data[j])
		}
		sort.Float64s(buf)
		out.Data[j] = superresClamp8(superresMedianSorted(buf))
	}
	return out
}

// ShiftAndAddSR fuses several low-resolution frames of a static scene into a
// single high-resolution image by the shift-and-add method: it splats every
// sample of every frame onto a scale× finer grid at its registered sub-pixel
// position, accumulating a weighted average per high-resolution cell, then
// fills any cells that received no sample by bicubic interpolation from the
// reference frame. shifts[i] is the displacement of frames[i] relative to the
// reference. Giving several frames with varied sub-pixel offsets yields genuine
// resolution gain; giving one frame (or identical offsets) degrades gracefully
// to interpolation. All frames must share dimensions and channel count. It
// panics if scale < 1, no frames are given, shapes differ, or len(shifts) is
// neither zero nor len(frames).
func ShiftAndAddSR(frames []*cv.Mat, shifts []Shift2D, scale int) *cv.Mat {
	if scale < 1 {
		panic("superres: ShiftAndAddSR requires scale >= 1")
	}
	superresValidateFrames(frames, shifts)
	ref := frames[0]
	ch := ref.Channels
	hiH := ref.Rows * scale
	hiW := ref.Cols * scale
	acc := make([]float64, hiH*hiW*ch)
	wsum := make([]float64, hiH*hiW)

	for i, f := range frames {
		var dx, dy float64
		if shifts != nil {
			dx, dy = shifts[i].Dx, shifts[i].Dy
		}
		for y := 0; y < f.Rows; y++ {
			for x := 0; x < f.Cols; x++ {
				// The sample sits, on the reference grid, at (x-dx, y-dy);
				// map that to the high-resolution grid (pixel-centre aligned).
				hx := (float64(x)-dx+0.5)*float64(scale) - 0.5
				hy := (float64(y)-dy+0.5)*float64(scale) - 0.5
				// Bilinear splat into the four surrounding cells.
				x0 := int(math.Floor(hx))
				y0 := int(math.Floor(hy))
				fx := hx - float64(x0)
				fy := hy - float64(y0)
				for _, o := range [4]struct {
					dxi, dyi int
					wt       float64
				}{
					{0, 0, (1 - fx) * (1 - fy)},
					{1, 0, fx * (1 - fy)},
					{0, 1, (1 - fx) * fy},
					{1, 1, fx * fy},
				} {
					cx := x0 + o.dxi
					cy := y0 + o.dyi
					if cx < 0 || cx >= hiW || cy < 0 || cy >= hiH || o.wt == 0 {
						continue
					}
					cell := cy*hiW + cx
					wsum[cell] += o.wt
					for c := 0; c < ch; c++ {
						acc[cell*ch+c] += o.wt * float64(f.Data[(y*f.Cols+x)*ch+c])
					}
				}
			}
		}
	}

	// Bicubic upscale of the reference as a fallback for empty cells.
	fallback := BicubicResize(ref, hiW, hiH)
	out := cv.NewMat(hiH, hiW, ch)
	for cell := 0; cell < hiH*hiW; cell++ {
		if wsum[cell] > 1e-6 {
			for c := 0; c < ch; c++ {
				out.Data[cell*ch+c] = superresClamp8(acc[cell*ch+c] / wsum[cell])
			}
		} else {
			for c := 0; c < ch; c++ {
				out.Data[cell*ch+c] = fallback.Data[cell*ch+c]
			}
		}
	}
	return out
}

// MultiFrameSR is the recommended end-to-end multi-frame super-resolution
// pipeline: it registers the frames with [RegisterFrames], fuses them onto a
// scale× grid with [ShiftAndAddSR], and applies a light unsharp mask to
// recover the high-frequency detail that splatting slightly softens. It panics
// if scale < 1 or no frames are given.
func MultiFrameSR(frames []*cv.Mat, scale int) *cv.Mat {
	if len(frames) == 0 {
		panic("superres: MultiFrameSR requires at least one frame")
	}
	shifts := RegisterFrames(frames)
	fused := ShiftAndAddSR(frames, shifts, scale)
	if scale == 1 {
		return fused
	}
	return UnsharpMask(fused, 1.0, 0.4, 0)
}

// superresValidateFrames checks that frames is non-empty, every frame matches
// the first in shape, and shifts is either nil or one entry per frame.
func superresValidateFrames(frames []*cv.Mat, shifts []Shift2D) {
	if len(frames) == 0 {
		panic("superres: at least one frame is required")
	}
	if shifts != nil && len(shifts) != len(frames) {
		panic("superres: shifts must have one entry per frame")
	}
	ref := frames[0]
	for _, f := range frames {
		superresCheckSame(ref, f)
	}
}

// superresMedianSorted returns the median of an already-sorted slice.
func superresMedianSorted(sorted []float64) float64 {
	n := len(sorted)
	if n == 0 {
		return 0
	}
	if n%2 == 1 {
		return sorted[n/2]
	}
	return 0.5 * (sorted[n/2-1] + sorted[n/2])
}
