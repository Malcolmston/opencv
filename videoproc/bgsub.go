package videoproc

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// BackgroundSubtractor is the common interface implemented by every background
// model in this package. Apply feeds the next frame to the model and returns a
// single-channel foreground mask (255 = foreground, 0 = background) the same
// size as the frame. Background returns the model's current estimate of the
// static scene as a single-channel Mat, or nil before any frame has been seen.
type BackgroundSubtractor interface {
	Apply(frame *cv.Mat) *cv.Mat
	Background() *cv.Mat
}

// RunningAverageSubtractor models the background as a per-pixel exponential
// running average of intensity, mirroring the simplest cv::BackgroundSubtractor
// variant. Each incoming pixel is compared to the current average; pixels that
// deviate by more than Threshold are foreground, and the average is then updated
// toward the new frame at rate Alpha.
type RunningAverageSubtractor struct {
	// Alpha is the learning rate in (0,1]; larger adapts faster.
	Alpha float64
	// Threshold is the absolute intensity deviation (0..255) above which a pixel
	// is classified as foreground.
	Threshold float64

	bg   *cv.FloatMat
	rows int
	cols int
}

// NewRunningAverageSubtractor returns a running-average model with the given
// learning rate alpha (in (0,1]) and foreground threshold (0..255). It panics on
// an out-of-range alpha.
func NewRunningAverageSubtractor(alpha, threshold float64) *RunningAverageSubtractor {
	if alpha <= 0 || alpha > 1 {
		panic("videoproc: NewRunningAverageSubtractor requires alpha in (0,1]")
	}
	return &RunningAverageSubtractor{Alpha: alpha, Threshold: threshold}
}

// Apply classifies frame against the running-average background and updates the
// model. The first frame initialises the background and yields an all-background
// mask. Frame dimensions must stay constant across calls.
func (s *RunningAverageSubtractor) Apply(frame *cv.Mat) *cv.Mat {
	g := videoprocToGray(frame)
	out := cv.NewMat(g.Rows, g.Cols, 1)
	if s.bg == nil {
		s.rows, s.cols = g.Rows, g.Cols
		s.bg = cv.NewFloatMat(g.Rows, g.Cols)
		for i := range g.Data {
			s.bg.Data[i] = float64(g.Data[i])
		}
		return out
	}
	if g.Rows != s.rows || g.Cols != s.cols {
		panic("videoproc: RunningAverageSubtractor frame size changed")
	}
	for i := range g.Data {
		v := float64(g.Data[i])
		diff := v - s.bg.Data[i]
		if diff < 0 {
			diff = -diff
		}
		if diff > s.Threshold {
			out.Data[i] = 255
		}
		s.bg.Data[i] = (1-s.Alpha)*s.bg.Data[i] + s.Alpha*v
	}
	return out
}

// Background returns the current running-average background as an 8-bit Mat, or
// nil before the first frame.
func (s *RunningAverageSubtractor) Background() *cv.Mat {
	if s.bg == nil {
		return nil
	}
	return AccumulatorToMat(s.bg, 1)
}

// MedianBackgroundSubtractor models the background as the per-pixel temporal
// median over a sliding window of the last History frames. The median is highly
// robust to transient foreground objects, so a pixel briefly covered by a moving
// object does not corrupt the background estimate. A pixel is foreground when it
// deviates from the current median by more than Threshold.
type MedianBackgroundSubtractor struct {
	// Threshold is the absolute intensity deviation (0..255) for foreground.
	Threshold int

	history int
	rows    int
	cols    int
	frames  [][]uint8 // ring buffer of grayscale frames, each len rows*cols
	next    int
	filled  int
}

// NewMedianBackgroundSubtractor returns a median model that keeps the most
// recent history frames (history >= 1) and uses the given foreground threshold
// (0..255). It panics if history < 1.
func NewMedianBackgroundSubtractor(history, threshold int) *MedianBackgroundSubtractor {
	if history < 1 {
		panic("videoproc: NewMedianBackgroundSubtractor requires history >= 1")
	}
	return &MedianBackgroundSubtractor{
		Threshold: threshold,
		history:   history,
		frames:    make([][]uint8, history),
	}
}

// Ready reports whether at least one frame has been supplied, so that
// [MedianBackgroundSubtractor.Background] returns a usable estimate.
func (s *MedianBackgroundSubtractor) Ready() bool {
	return s.filled > 0
}

// Apply classifies frame against the current per-pixel median and then inserts
// it into the sliding window. The foreground mask is computed against the median
// of the frames seen before this one; the very first frame yields an
// all-background mask. Frame dimensions must stay constant across calls.
func (s *MedianBackgroundSubtractor) Apply(frame *cv.Mat) *cv.Mat {
	g := videoprocToGray(frame)
	if s.filled == 0 {
		s.rows, s.cols = g.Rows, g.Cols
	} else if g.Rows != s.rows || g.Cols != s.cols {
		panic("videoproc: MedianBackgroundSubtractor frame size changed")
	}
	out := cv.NewMat(g.Rows, g.Cols, 1)
	if s.filled > 0 {
		bg := s.median()
		for i := range g.Data {
			d := int(g.Data[i]) - int(bg[i])
			if d < 0 {
				d = -d
			}
			if d > s.Threshold {
				out.Data[i] = 255
			}
		}
	}
	// insert current frame into the ring buffer
	cp := make([]uint8, len(g.Data))
	copy(cp, g.Data)
	s.frames[s.next] = cp
	s.next = (s.next + 1) % s.history
	if s.filled < s.history {
		s.filled++
	}
	return out
}

// median computes the per-pixel median across the currently stored frames.
func (s *MedianBackgroundSubtractor) median() []uint8 {
	n := s.rows * s.cols
	bg := make([]uint8, n)
	vals := make([]uint8, 0, s.filled)
	for p := 0; p < n; p++ {
		vals = vals[:0]
		for k := 0; k < s.filled; k++ {
			vals = append(vals, s.frames[k][p])
		}
		bg[p] = videoprocMedianU8(vals)
	}
	return bg
}

// Background returns the current per-pixel median background as an 8-bit Mat, or
// nil before any frame has been supplied.
func (s *MedianBackgroundSubtractor) Background() *cv.Mat {
	if s.filled == 0 {
		return nil
	}
	bg := s.median()
	out := cv.NewMat(s.rows, s.cols, 1)
	copy(out.Data, bg)
	return out
}

// videoprocGaussian is one component of a per-pixel Gaussian mixture.
type videoprocGaussian struct {
	weight float64
	mean   float64
	varc   float64
}

// MOGBackgroundSubtractor models each pixel's intensity as an adaptive mixture
// of Gaussians (MOG), the algorithm of Stauffer & Grimson. Each incoming pixel
// is matched against its per-pixel Gaussians; a match updates that component's
// weight, mean and variance online, while an unmatched pixel replaces the
// weakest component. Components are ranked by weight/σ and the most reliable ones
// whose weights sum past BackgroundRatio form the background model. A pixel is
// foreground when it matches no background component within VarThreshold
// standard deviations.
type MOGBackgroundSubtractor struct {
	// LearningRate is the online update rate in (0,1].
	LearningRate float64
	// VarThreshold is the squared-Mahalanobis match threshold (e.g. 6.25 = 2.5σ).
	VarThreshold float64
	// BackgroundRatio is the cumulative weight fraction that counts as background.
	BackgroundRatio float64

	nmix    int
	initVar float64
	rows    int
	cols    int
	model   [][]videoprocGaussian
}

// NewMOGBackgroundSubtractor returns a MOG model with OpenCV-like defaults: five
// Gaussians per pixel, learning rate 0.01, a 2.5σ (6.25 squared) match threshold
// and a 0.7 background-weight ratio.
func NewMOGBackgroundSubtractor() *MOGBackgroundSubtractor {
	return &MOGBackgroundSubtractor{
		LearningRate:    0.01,
		VarThreshold:    6.25,
		BackgroundRatio: 0.7,
		nmix:            5,
		initVar:         225, // (15 intensity units)^2
	}
}

// Apply feeds the next frame to the mixture model and returns a single-channel
// foreground mask (255 = foreground). Multi-channel frames are converted to
// grayscale. Frame dimensions must stay constant across calls.
func (s *MOGBackgroundSubtractor) Apply(frame *cv.Mat) *cv.Mat {
	g := videoprocToGray(frame)
	n := g.Total()
	if s.model == nil {
		s.rows, s.cols = g.Rows, g.Cols
		s.model = make([][]videoprocGaussian, n)
	} else if g.Rows != s.rows || g.Cols != s.cols {
		panic("videoproc: MOGBackgroundSubtractor frame size changed")
	}
	out := cv.NewMat(g.Rows, g.Cols, 1)
	lr := s.LearningRate
	for p := 0; p < n; p++ {
		x := float64(g.Data[p])
		mix := s.model[p]
		if mix == nil {
			// initialise with a single dominant component on this pixel value.
			mix = []videoprocGaussian{{weight: 1, mean: x, varc: s.initVar}}
			s.model[p] = mix
			continue
		}
		matched := -1
		for i := range mix {
			d := x - mix[i].mean
			if d*d < s.VarThreshold*mix[i].varc {
				matched = i
				break
			}
		}
		if matched < 0 {
			// replace weakest (or append) with a new wide component.
			g2 := videoprocGaussian{weight: 0.05, mean: x, varc: s.initVar}
			if len(mix) < s.nmix {
				mix = append(mix, g2)
			} else {
				worst := 0
				for i := 1; i < len(mix); i++ {
					if mix[i].weight < mix[worst].weight {
						worst = i
					}
				}
				mix[worst] = g2
			}
		} else {
			d := x - mix[matched].mean
			rho := lr / mix[matched].weight
			if rho > 1 {
				rho = 1
			}
			mix[matched].mean += rho * d
			mix[matched].varc += rho * (d*d - mix[matched].varc)
			if mix[matched].varc < 4 {
				mix[matched].varc = 4
			}
		}
		// update and renormalise weights.
		var sum float64
		for i := range mix {
			hit := 0.0
			if i == matched {
				hit = 1.0
			}
			mix[i].weight = (1-lr)*mix[i].weight + lr*hit
			sum += mix[i].weight
		}
		if sum > 0 {
			for i := range mix {
				mix[i].weight /= sum
			}
		}
		s.model[p] = mix
		// classify: foreground if the matched component is not part of the
		// background set (top components by weight/σ up to BackgroundRatio).
		if matched < 0 || !s.isBackground(mix, matched) {
			out.Data[p] = 255
		}
	}
	return out
}

// isBackground reports whether component idx lies within the highest-ranked
// components (by weight/σ) whose cumulative weight is below BackgroundRatio.
func (s *MOGBackgroundSubtractor) isBackground(mix []videoprocGaussian, idx int) bool {
	order := make([]int, len(mix))
	for i := range order {
		order[i] = i
	}
	// selection sort by weight/sqrt(var) descending (small n).
	for i := 0; i < len(order); i++ {
		best := i
		for j := i + 1; j < len(order); j++ {
			if videoprocRank(mix[order[j]]) > videoprocRank(mix[order[best]]) {
				best = j
			}
		}
		order[i], order[best] = order[best], order[i]
	}
	var cum float64
	for _, oi := range order {
		cum += mix[oi].weight
		if oi == idx {
			return true
		}
		if cum >= s.BackgroundRatio {
			break
		}
	}
	return false
}

// videoprocRank returns the weight/σ sorting key of a Gaussian component.
func videoprocRank(gauss videoprocGaussian) float64 {
	if gauss.varc <= 0 {
		return 0
	}
	return gauss.weight / math.Sqrt(gauss.varc)
}

// Background returns the per-pixel mean of the most reliable Gaussian (largest
// weight/σ) as an 8-bit Mat, or nil before the first frame.
func (s *MOGBackgroundSubtractor) Background() *cv.Mat {
	if s.model == nil {
		return nil
	}
	out := cv.NewMat(s.rows, s.cols, 1)
	for p := range s.model {
		mix := s.model[p]
		if len(mix) == 0 {
			continue
		}
		best := 0
		for i := 1; i < len(mix); i++ {
			if videoprocRank(mix[i]) > videoprocRank(mix[best]) {
				best = i
			}
		}
		out.Data[p] = videoprocClampU8(mix[best].mean + 0.5)
	}
	return out
}
