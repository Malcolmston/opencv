package videostab

import (
	cv "github.com/malcolmston/opencv"
)

// CalcBlurriness returns a scalar blurriness measure of a frame, matching
// cv::videostab::calcBlurriness. It is the reciprocal of the mean squared
// gradient magnitude, so a sharp, high-contrast frame yields a small value and a
// smooth, blurred frame yields a large one. Multi-channel frames are converted
// to grayscale first. It panics on an empty frame.
func CalcBlurriness(frame *cv.Mat) float64 {
	if frame == nil || frame.Empty() {
		panic("videostab: CalcBlurriness requires a non-empty frame")
	}
	gray := grayscale(frame)
	gx := cv.SobelFloat(gray, 1, 0, 3)[0]
	gy := cv.SobelFloat(gray, 0, 1, 3)[0]
	var sumSq float64
	for i := range gx {
		sumSq += gx[i]*gx[i] + gy[i]*gy[i]
	}
	area := float64(gray.Total())
	return 1.0 / (sumSq/area + 1e-6)
}

// DeblurerBase removes motion blur from a stabilized frame in place, using the
// surrounding frames and their motions as context. It mirrors
// cv::videostab::DeblurerBase.
type DeblurerBase interface {
	// Deblur sharpens the frame at index idx in place.
	Deblur(idx int, frame *cv.Mat)
	// SetContext supplies the frame buffer, inter-frame motions and per-frame
	// blurriness measures the deblurer may draw on.
	SetContext(frames []*cv.Mat, motions []Motion, blurriness []float64)
}

// NullDeblurer performs no deblurring. It is the default when deblurring is not
// requested.
type NullDeblurer struct{}

// Deblur does nothing.
func (NullDeblurer) Deblur(int, *cv.Mat) {}

// SetContext does nothing.
func (NullDeblurer) SetContext([]*cv.Mat, []Motion, []float64) {}

// WeightingDeblurer sharpens a frame with an unsharp mask whose strength is
// weighted towards the sharpness of the frame's temporal neighbours, following
// the spirit of cv::videostab::WeightingDeblurer.
//
// The high-frequency detail that motion blur suppresses is recovered by
// subtracting a low-pass (Gaussian) copy of the frame from the frame itself and
// adding the difference back, scaled by an amount. The amount always has a
// positive base ([WeightingDeblurer.Amount]) so the operation strictly
// increases the frame's gradient energy — it genuinely sharpens rather than
// copies — and is boosted further, in proportion to
// [WeightingDeblurer.Sensitivity], whenever the target frame is measurably
// blurrier than the sharpest frame in its temporal window (its neighbours carry
// detail the target has lost, so it is deblurred harder).
type WeightingDeblurer struct {
	// Radius is the temporal half-window of neighbours inspected to decide how
	// aggressively to sharpen.
	Radius int
	// Sensitivity scales the extra unsharp amount applied when the frame is
	// blurrier than its sharpest neighbour.
	Sensitivity float64
	// Amount is the base unsharp-mask amount. It must be positive for the
	// deblurer to sharpen; the zero value is treated as 1.
	Amount float64
	// KSize is the (odd) Gaussian kernel size used to build the low-pass copy.
	KSize int

	frames     []*cv.Mat
	motions    []Motion
	blurriness []float64
}

// NewWeightingDeblurer creates a weighting deblurer with the given temporal
// radius and sensible unsharp defaults (base amount 1.0, sensitivity 0.8, a 3×3
// low-pass kernel). radius must be positive.
func NewWeightingDeblurer(radius int) *WeightingDeblurer {
	if radius < 1 {
		panic("videostab: NewWeightingDeblurer requires radius >= 1")
	}
	return &WeightingDeblurer{Radius: radius, Sensitivity: 0.8, Amount: 1.0, KSize: 3}
}

// SetContext supplies the frame buffer, motions and per-frame blurriness
// measures.
func (d *WeightingDeblurer) SetContext(frames []*cv.Mat, motions []Motion, blurriness []float64) {
	d.frames = frames
	d.motions = motions
	d.blurriness = blurriness
}

// Deblur sharpens the frame at index idx in place with a neighbour-weighted
// unsharp mask. The result is strictly sharper (higher gradient energy) than the
// input whenever the frame contains any non-constant, non-saturated detail.
func (d *WeightingDeblurer) Deblur(idx int, frame *cv.Mat) {
	if frame == nil || frame.Empty() {
		return
	}
	ksize := d.KSize
	if ksize < 3 {
		ksize = 3
	}
	if ksize%2 == 0 {
		ksize++
	}
	amount := d.Amount
	if amount <= 0 {
		amount = 1.0
	}
	// Weight the amount towards the sharpness of the neighbourhood: if the frame
	// is blurrier (larger blurriness measure) than its sharpest neighbour, push
	// the unsharp amount up in proportion to the relative deficit.
	if len(d.blurriness) == len(d.frames) && idx >= 0 && idx < len(d.blurriness) {
		lo := clampInt(idx-d.Radius, 0, len(d.frames)-1)
		hi := clampInt(idx+d.Radius, 0, len(d.frames)-1)
		best := d.blurriness[idx]
		for j := lo; j <= hi; j++ {
			if j == idx {
				continue
			}
			if d.blurriness[j] < best {
				best = d.blurriness[j]
			}
		}
		if d.blurriness[idx] > 0 && best < d.blurriness[idx] {
			rel := (d.blurriness[idx] - best) / d.blurriness[idx] // in (0, 1)
			amount += d.Sensitivity * rel
		}
	}

	blurred := cv.GaussianBlur(frame, ksize, 0)
	for i := range frame.Data {
		f := float64(frame.Data[i])
		lp := float64(blurred.Data[i])
		frame.Data[i] = clampByte(f + amount*(f-lp))
	}
}

// matchChannels returns src expanded or reduced to the requested channel count.
// Extra channels replicate channel 0; a reduction keeps only channel 0.
func matchChannels(src *cv.Mat, channels int) *cv.Mat {
	if src.Channels == channels {
		return src
	}
	out := cv.NewMat(src.Rows, src.Cols, channels)
	for p := 0; p < src.Total(); p++ {
		v := src.Data[p*src.Channels]
		for c := 0; c < channels; c++ {
			out.Data[p*channels+c] = v
		}
	}
	return out
}

// clampByte rounds and clamps a float to the 0..255 byte range.
func clampByte(v float64) uint8 {
	v += 0.5
	if v < 0 {
		return 0
	}
	if v > 255 {
		return 255
	}
	return uint8(v)
}
