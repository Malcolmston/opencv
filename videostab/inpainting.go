package videostab

import (
	cv "github.com/malcolmston/opencv"
)

// Inpainter fills the masked (missing) region of a frame, for example the empty
// border introduced when a frame is warped for stabilization. It mirrors
// cv::videostab::InpainterBase.
type Inpainter interface {
	// Inpaint fills the holes of frame in place. mask is single-channel: a
	// value of 0 marks a pixel that must be filled, any non-zero value marks a
	// known pixel. After the call the filled pixels of mask are set to 255.
	Inpaint(idx int, frame, mask *cv.Mat)
	// SetContext supplies the frame buffer, inter-frame motions, per-frame
	// stabilization warps and temporal radius the inpainter may use.
	SetContext(frames []*cv.Mat, motions, stabilizationMotions []Motion, radius int)
}

// InpainterContext holds the shared context propagated to inpainters: the frame
// buffer, the inter-frame motions, the per-frame stabilization warps and the
// temporal radius.
type InpainterContext struct {
	Frames               []*cv.Mat
	Motions              []Motion
	StabilizationMotions []Motion
	Radius               int
}

// ColorInpainter fills holes by diffusion: each missing pixel is repeatedly
// replaced with the average of its currently-known neighbours until the region
// is filled. It needs no temporal context and is the analogue of OpenCV's
// single-frame ColorInpainter.
type ColorInpainter struct {
	// Iterations bounds the number of diffusion sweeps.
	Iterations int
}

// NewColorInpainter creates a diffusion inpainter with a sensible sweep count.
func NewColorInpainter() *ColorInpainter { return &ColorInpainter{Iterations: 256} }

// SetContext is a no-op; ColorInpainter uses only the frame itself.
func (c *ColorInpainter) SetContext([]*cv.Mat, []Motion, []Motion, int) {}

// Inpaint fills holes by iterative neighbour averaging (a discrete diffusion).
// It fills every hole that is connected to a known pixel; on a frame that has at
// least one known pixel this leaves no holes behind.
func (c *ColorInpainter) Inpaint(_ int, frame, mask *cv.Mat) {
	fillHolesByDiffusion(frame, mask, c.Iterations)
}

// ColorAverageInpainter fills each missing pixel with the average of the same
// pixel in the temporal neighbours where it is known, assuming the frames are
// already roughly registered. It mirrors cv::videostab::ColorAverageInpainter.
type ColorAverageInpainter struct {
	ctx InpainterContext
}

// NewColorAverageInpainter creates a temporal-average inpainter.
func NewColorAverageInpainter() *ColorAverageInpainter { return &ColorAverageInpainter{} }

// SetContext supplies the frame buffer and radius.
func (c *ColorAverageInpainter) SetContext(frames []*cv.Mat, motions, stab []Motion, radius int) {
	c.ctx = InpainterContext{Frames: frames, Motions: motions, StabilizationMotions: stab, Radius: radius}
}

// Inpaint averages known neighbour-frame pixels into the holes.
func (c *ColorAverageInpainter) Inpaint(idx int, frame, mask *cv.Mat) {
	frames := c.ctx.Frames
	if len(frames) == 0 {
		return
	}
	lo := clampInt(idx-c.ctx.Radius, 0, len(frames)-1)
	hi := clampInt(idx+c.ctx.Radius, 0, len(frames)-1)
	ch := frame.Channels
	for p := 0; p < frame.Total(); p++ {
		if mask.Data[p] != 0 {
			continue
		}
		var sum [4]float64
		n := 0
		for j := lo; j <= hi; j++ {
			if j == idx {
				continue
			}
			nf := frames[j]
			if nf.Channels != ch || nf.Total() != frame.Total() {
				continue
			}
			// A neighbour pixel counts only when it is itself valid (non-zero).
			valid := false
			for cc := 0; cc < ch; cc++ {
				if nf.Data[p*ch+cc] != 0 {
					valid = true
					break
				}
			}
			if !valid {
				continue
			}
			for cc := 0; cc < ch; cc++ {
				sum[cc] += float64(nf.Data[p*ch+cc])
			}
			n++
		}
		if n == 0 {
			continue
		}
		for cc := 0; cc < ch; cc++ {
			frame.Data[p*ch+cc] = clampByte(sum[cc] / float64(n))
		}
		mask.Data[p] = 255
	}
}

// MotionInpainter fills holes with content borrowed from neighbouring frames
// that are motion-compensated into the target frame's coordinate system, using
// the estimated inter-frame motions. It is the motion-aware counterpart of
// [ColorAverageInpainter] and mirrors cv::videostab::MotionInpainter.
//
// Neighbours are visited in order of increasing temporal distance, so each hole
// is filled from the closest (most reliable) source that covers it. A borrowed
// pixel is accepted only where the warp of the neighbour genuinely covers the
// output — coverage is tracked with an explicit warped mask rather than by
// treating black pixels as holes, so legitimately dark content is not skipped.
// Any pixel that no neighbour can supply (for example a corner outside every
// neighbour's field of view) is finished off by diffusion from the pixels that
// were filled, guaranteeing that no hole remains in the valid canvas.
type MotionInpainter struct {
	ctx InpainterContext
}

// NewMotionInpainter creates a motion-compensated inpainter.
func NewMotionInpainter() *MotionInpainter { return &MotionInpainter{} }

// SetContext supplies the frame buffer, motions, stabilization warps and radius.
func (m *MotionInpainter) SetContext(frames []*cv.Mat, motions, stab []Motion, radius int) {
	m.ctx = InpainterContext{Frames: frames, Motions: motions, StabilizationMotions: stab, Radius: radius}
}

// Inpaint fills holes from motion-compensated neighbour frames, nearest first,
// then guarantees a hole-free result with a diffusion pass.
func (m *MotionInpainter) Inpaint(idx int, frame, mask *cv.Mat) {
	frames := m.ctx.Frames
	ch := frame.Channels
	if len(frames) > 0 {
		radius := m.ctx.Radius
		if radius < 1 {
			radius = 1
		}
		for dist := 1; dist <= radius && countHoles(mask) > 0; dist++ {
			for _, j := range []int{idx - dist, idx + dist} {
				if j < 0 || j >= len(frames) || j == idx {
					continue
				}
				// Map neighbour j's content into idx's coordinate system
				// (GetMotion(j, idx) maps frame j onto frame idx), then apply the
				// stabilization warp of idx so the borrowed content lands where the
				// stabilized frame expects it.
				warp := GetMotion(j, idx, m.ctx.Motions)
				if idx < len(m.ctx.StabilizationMotions) {
					warp = m.ctx.StabilizationMotions[idx].Mul(warp)
				}
				aligned, cover := warpWithCoverage(frames[j], warp)
				if aligned.Channels != ch {
					aligned = matchChannels(aligned, ch)
				}
				for p := 0; p < frame.Total(); p++ {
					if mask.Data[p] != 0 || !cover[p] {
						continue
					}
					for cc := 0; cc < ch; cc++ {
						frame.Data[p*ch+cc] = aligned.Data[p*ch+cc]
					}
					mask.Data[p] = 255
				}
			}
		}
	}
	// Whatever no neighbour could supply is filled by diffusion so that the
	// valid canvas region is left completely hole-free.
	fillHolesByDiffusion(frame, mask, frame.Rows+frame.Cols+256)
}

// InpaintingPipeline chains several inpainters, running each in turn so that
// later inpainters only see the holes their predecessors could not fill. It
// mirrors cv::videostab::InpaintingPipeline.
type InpaintingPipeline struct {
	inpainters []Inpainter
}

// NewInpaintingPipeline creates an empty inpainting pipeline.
func NewInpaintingPipeline() *InpaintingPipeline { return &InpaintingPipeline{} }

// Add appends an inpainter and returns the pipeline for chaining.
func (p *InpaintingPipeline) Add(in Inpainter) *InpaintingPipeline {
	p.inpainters = append(p.inpainters, in)
	return p
}

// Len reports how many inpainters the pipeline contains.
func (p *InpaintingPipeline) Len() int { return len(p.inpainters) }

// SetContext propagates the shared context to every inpainter in the pipeline.
func (p *InpaintingPipeline) SetContext(frames []*cv.Mat, motions, stab []Motion, radius int) {
	for _, in := range p.inpainters {
		in.SetContext(frames, motions, stab, radius)
	}
}

// Inpaint runs every inpainter in sequence.
func (p *InpaintingPipeline) Inpaint(idx int, frame, mask *cv.Mat) {
	for _, in := range p.inpainters {
		in.Inpaint(idx, frame, mask)
	}
}

// countHoles returns the number of pixels marked as holes (mask == 0).
func countHoles(mask *cv.Mat) int {
	n := 0
	for _, v := range mask.Data {
		if v == 0 {
			n++
		}
	}
	return n
}

// warpWithCoverage warps src by the affine part of the motion and, alongside it,
// warps an all-ones coverage plane so that callers can tell exactly which output
// pixels were sampled from inside src (true) and which fell outside it (false).
func warpWithCoverage(src *cv.Mat, warp Motion) (*cv.Mat, []bool) {
	aligned := warp.warp(src)
	white := cv.NewMat(src.Rows, src.Cols, 1)
	white.SetTo(255)
	warpedMask := warp.warp(white)
	cover := make([]bool, aligned.Total())
	for p := 0; p < aligned.Total(); p++ {
		cover[p] = warpedMask.Data[p] >= 128
	}
	return aligned, cover
}

// fillHolesByDiffusion replaces every hole (mask == 0) that is connected to a
// known pixel with the running average of its known 4-neighbours, sweeping until
// no hole remains or no further progress is possible (bounded by maxSweeps).
// Newly filled pixels become sources within the same sweep, so the fill
// propagates inward from the boundary of the known region and, on any frame with
// at least one known pixel, leaves no holes behind.
func fillHolesByDiffusion(frame, mask *cv.Mat, maxSweeps int) {
	total := frame.Total()
	ch := frame.Channels
	known := make([]bool, total)
	remaining := 0
	for p := 0; p < total; p++ {
		if mask.Data[p] != 0 {
			known[p] = true
		} else {
			remaining++
		}
	}
	if remaining == 0 {
		return
	}
	if maxSweeps < 1 {
		maxSweeps = 1
	}
	rows, cols := frame.Rows, frame.Cols
	for sweep := 0; sweep < maxSweeps && remaining > 0; sweep++ {
		filled := 0
		for y := 0; y < rows; y++ {
			for x := 0; x < cols; x++ {
				p := y*cols + x
				if known[p] {
					continue
				}
				var sum [4]float64
				n := 0
				if y > 0 && known[p-cols] {
					accumPixel(&sum, frame, p-cols, ch)
					n++
				}
				if y < rows-1 && known[p+cols] {
					accumPixel(&sum, frame, p+cols, ch)
					n++
				}
				if x > 0 && known[p-1] {
					accumPixel(&sum, frame, p-1, ch)
					n++
				}
				if x < cols-1 && known[p+1] {
					accumPixel(&sum, frame, p+1, ch)
					n++
				}
				if n == 0 {
					continue
				}
				for cc := 0; cc < ch; cc++ {
					frame.Data[p*ch+cc] = clampByte(sum[cc] / float64(n))
				}
				known[p] = true
				mask.Data[p] = 255
				filled++
				remaining--
			}
		}
		if filled == 0 {
			break
		}
	}
}

// accumPixel adds pixel q's channel samples into sum.
func accumPixel(sum *[4]float64, frame *cv.Mat, q, ch int) {
	for cc := 0; cc < ch; cc++ {
		sum[cc] += float64(frame.Data[q*ch+cc])
	}
}
