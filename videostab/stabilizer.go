package videostab

import (
	cv "github.com/malcolmston/opencv"
)

// StabilizerBase holds the configuration and per-run state shared by
// [OnePassStabilizer] and [TwoPassStabilizer]: the frame buffer, the estimated
// inter-frame motions, the motion estimator and the optional inpainting,
// deblurring and border-trimming stages. It mirrors
// cv::videostab::StabilizerBase.
type StabilizerBase struct {
	radius    int
	trimRatio float64
	estimator ImageMotionEstimator
	inpainter Inpainter
	deblurer  DeblurerBase

	frames  []*cv.Mat
	motions []Motion // motions[i] maps frame i to frame i+1
	stab    []Motion // per-frame stabilization warps
	curPos  int
	ready   bool
}

// SetRadius sets the temporal radius used for trajectory smoothing and for the
// inpainting / deblurring neighbourhoods. It must be >= 1.
func (s *StabilizerBase) SetRadius(r int) {
	if r < 1 {
		panic("videostab: SetRadius requires radius >= 1")
	}
	s.radius = r
}

// Radius returns the temporal radius.
func (s *StabilizerBase) Radius() int { return s.radius }

// SetTrimRatio sets the fraction of the frame trimmed from every side (and
// zoomed back to full size) to hide the empty border produced by warping. It
// must lie in [0, 0.45].
func (s *StabilizerBase) SetTrimRatio(r float64) {
	if r < 0 || r > 0.45 {
		panic("videostab: SetTrimRatio requires ratio in [0, 0.45]")
	}
	s.trimRatio = r
}

// TrimRatio returns the current trim ratio.
func (s *StabilizerBase) TrimRatio() float64 { return s.trimRatio }

// SetMotionEstimator overrides the image motion estimator.
func (s *StabilizerBase) SetMotionEstimator(e ImageMotionEstimator) { s.estimator = e }

// SetInpainter enables border inpainting with the given inpainter.
func (s *StabilizerBase) SetInpainter(in Inpainter) { s.inpainter = in }

// SetDeblurer enables deblurring with the given deblurer.
func (s *StabilizerBase) SetDeblurer(d DeblurerBase) { s.deblurer = d }

// SetFrames sets the input frame sequence and resets the stabilizer. Every frame
// must have the same dimensions and channel count.
func (s *StabilizerBase) SetFrames(frames []*cv.Mat) {
	s.frames = frames
	s.motions = nil
	s.stab = nil
	s.curPos = 0
	s.ready = false
}

// Frames returns the input frame sequence.
func (s *StabilizerBase) Frames() []*cv.Mat { return s.frames }

// Motions returns the estimated inter-frame motions (available after the first
// NextFrame call). motions[i] maps frame i to frame i+1.
func (s *StabilizerBase) Motions() []Motion { return s.motions }

// StabilizationMotions returns the per-frame stabilization warps (available
// after the first NextFrame call).
func (s *StabilizerBase) StabilizationMotions() []Motion { return s.stab }

// estimateMotions fills s.motions by estimating the motion between every pair of
// consecutive frames with the configured estimator.
func (s *StabilizerBase) estimateMotions() {
	n := len(s.frames)
	s.motions = make([]Motion, max0(n-1))
	for i := 0; i+1 < n; i++ {
		m, ok := s.estimator.Estimate(s.frames[i], s.frames[i+1])
		if !ok {
			m = IdentityMotion()
		}
		s.motions[i] = m
	}
}

// zoomMotion returns the scale-about-centre transform used to trim borders.
func (s *StabilizerBase) zoomMotion() (Motion, bool) {
	if s.trimRatio <= 0 || len(s.frames) == 0 {
		return IdentityMotion(), false
	}
	scale := 1.0 / (1.0 - 2*s.trimRatio)
	cx := float64(s.frames[0].Cols-1) / 2
	cy := float64(s.frames[0].Rows-1) / 2
	return Motion{scale, 0, (1 - scale) * cx, 0, scale, (1 - scale) * cy, 0, 0, 1}, true
}

// postProcess warps frame idx by its stabilization motion and applies the
// optional trim, inpainting and deblurring stages.
func (s *StabilizerBase) postProcess(idx int) *cv.Mat {
	frame := s.frames[idx]
	warp := s.stab[idx]
	if zoom, ok := s.zoomMotion(); ok {
		warp = zoom.Mul(warp)
	}
	out := warp.warp(frame)

	if s.inpainter != nil {
		mask := coverageMask(frame, warp)
		s.inpainter.Inpaint(idx, out, mask)
	}
	if s.deblurer != nil {
		s.deblurer.Deblur(idx, out)
	}
	return out
}

// coverageMask returns a single-channel mask that is 255 where the warp of frame
// covers the output and 0 (a hole) elsewhere.
func coverageMask(frame *cv.Mat, warp Motion) *cv.Mat {
	white := cv.NewMat(frame.Rows, frame.Cols, 1)
	white.SetTo(255)
	warped := warp.warp(white)
	mask := cv.NewMat(frame.Rows, frame.Cols, 1)
	for i := range warped.Data {
		if warped.Data[i] >= 128 {
			mask.Data[i] = 255
		}
	}
	return mask
}

// setupHelpers wires the frame/motion context into the inpainter and deblurer.
func (s *StabilizerBase) setupHelpers() {
	if s.inpainter != nil {
		s.inpainter.SetContext(s.frames, s.motions, s.stab, s.radius)
	}
	if s.deblurer != nil {
		blur := make([]float64, len(s.frames))
		for i, f := range s.frames {
			blur[i] = CalcBlurriness(f)
		}
		s.deblurer.SetContext(s.frames, s.motions, blur)
	}
}

func max0(v int) int {
	if v < 0 {
		return 0
	}
	return v
}

// OnePassStabilizer stabilizes a sequence in a single pass by smoothing the
// camera trajectory online with a [GaussianMotionFilter]. It mirrors
// cv::videostab::OnePassStabilizer.
//
// Feed the frames with [OnePassStabilizer.SetFrames] and pull stabilized frames
// one at a time with [OnePassStabilizer.NextFrame], or obtain the whole result
// at once with [OnePassStabilizer.Stabilize].
type OnePassStabilizer struct {
	StabilizerBase
	filter MotionFilterBase
}

// NewOnePassStabilizer creates a one-pass stabilizer with the given smoothing
// radius. It defaults to keypoint-based affine motion estimation and a Gaussian
// trajectory filter.
func NewOnePassStabilizer(radius int) *OnePassStabilizer {
	if radius < 1 {
		panic("videostab: NewOnePassStabilizer requires radius >= 1")
	}
	s := &OnePassStabilizer{filter: NewGaussianMotionFilter(radius, 0)}
	s.radius = radius
	s.estimator = NewKeypointBasedMotionEstimator(NewMotionEstimatorRansacL2(MotionModelAffine))
	return s
}

// SetMotionFilter overrides the trajectory-smoothing filter.
func (s *OnePassStabilizer) SetMotionFilter(f MotionFilterBase) { s.filter = f }

// prepare runs the (single) estimation and smoothing pass. It is idempotent.
func (s *OnePassStabilizer) prepare() {
	if s.ready {
		return
	}
	s.estimateMotions()
	n := len(s.frames)
	s.stab = make([]Motion, n)
	rng := Range{First: 0, Last: max0(n - 1)}
	for i := 0; i < n; i++ {
		s.stab[i] = s.filter.StabilizeAt(i, s.motions, rng)
	}
	s.setupHelpers()
	s.ready = true
}

// NextFrame returns the next stabilized frame. The second result is false once
// the sequence is exhausted.
func (s *OnePassStabilizer) NextFrame() (*cv.Mat, bool) {
	if len(s.frames) == 0 {
		return nil, false
	}
	s.prepare()
	if s.curPos >= len(s.frames) {
		return nil, false
	}
	out := s.postProcess(s.curPos)
	s.curPos++
	return out, true
}

// Stabilize returns every stabilized frame of the sequence.
func (s *OnePassStabilizer) Stabilize() []*cv.Mat {
	return drain(s)
}

// TwoPassStabilizer stabilizes a sequence in two passes: the first pass
// estimates every inter-frame motion and runs a global [MotionStabilizer]
// (which can see the entire trajectory, so it produces a smoother, non-causal
// result than the one-pass filter); the second pass warps the frames. It mirrors
// cv::videostab::TwoPassStabilizer.
type TwoPassStabilizer struct {
	StabilizerBase
	stabilizer MotionStabilizer
}

// NewTwoPassStabilizer creates a two-pass stabilizer with the given radius. It
// defaults to keypoint-based affine motion estimation and a motion
// stabilization pipeline containing a Gaussian filter.
func NewTwoPassStabilizer(radius int) *TwoPassStabilizer {
	if radius < 1 {
		panic("videostab: NewTwoPassStabilizer requires radius >= 1")
	}
	s := &TwoPassStabilizer{
		stabilizer: NewMotionStabilizationPipeline().Add(NewGaussianMotionFilter(radius, 0)),
	}
	s.radius = radius
	s.estimator = NewKeypointBasedMotionEstimator(NewMotionEstimatorRansacL2(MotionModelAffine))
	return s
}

// SetMotionStabilizer overrides the global motion stabilizer used in the first
// pass.
func (s *TwoPassStabilizer) SetMotionStabilizer(m MotionStabilizer) { s.stabilizer = m }

// prepare runs the first pass: motion estimation and global stabilization.
func (s *TwoPassStabilizer) prepare() {
	if s.ready {
		return
	}
	s.estimateMotions()
	n := len(s.frames)
	s.stab = make([]Motion, n)
	rng := Range{First: 0, Last: max0(n - 1)}
	s.stabilizer.Stabilize(n, s.motions, rng, s.stab)
	s.setupHelpers()
	s.ready = true
}

// NextFrame returns the next stabilized frame. The second result is false once
// the sequence is exhausted.
func (s *TwoPassStabilizer) NextFrame() (*cv.Mat, bool) {
	if len(s.frames) == 0 {
		return nil, false
	}
	s.prepare()
	if s.curPos >= len(s.frames) {
		return nil, false
	}
	out := s.postProcess(s.curPos)
	s.curPos++
	return out, true
}

// Stabilize returns every stabilized frame of the sequence.
func (s *TwoPassStabilizer) Stabilize() []*cv.Mat {
	return drain(s)
}

// frameSource is the common NextFrame contract of both stabilizers.
type frameSource interface {
	NextFrame() (*cv.Mat, bool)
}

// drain pulls every frame from a stabilizer into a slice.
func drain(s frameSource) []*cv.Mat {
	var out []*cv.Mat
	for {
		f, ok := s.NextFrame()
		if !ok {
			break
		}
		out = append(out, f)
	}
	return out
}
