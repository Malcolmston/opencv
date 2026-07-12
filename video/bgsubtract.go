package video

import (
	"sort"

	cv "github.com/malcolmston/opencv"
)

// gaussian is one component of a per-pixel Gaussian mixture.
type gaussian struct {
	weight float64
	mean   float64
	varc   float64
}

// BackgroundSubtractorMOG2 models each pixel as an adaptive mixture of Gaussians
// and classifies incoming pixels as foreground or background, mirroring
// cv::BackgroundSubtractorMOG2. Every call to [BackgroundSubtractorMOG2.Apply]
// updates the per-pixel mixtures with the new frame and returns a binary
// foreground mask. The model operates on grayscale intensity.
type BackgroundSubtractorMOG2 struct {
	// History bounds the effective learning rate (1/min(frame, History)).
	History int
	// VarThreshold is the squared-Mahalanobis threshold for matching a pixel to
	// an existing Gaussian.
	VarThreshold float64
	// BackgroundRatio is the cumulative weight fraction that counts as the
	// background model.
	BackgroundRatio float64

	nmixtures int
	initVar   float64
	minVar    float64
	rows      int
	cols      int
	frame     int
	model     [][]gaussian // rows*cols entries, each a mixture
}

// NewBackgroundSubtractorMOG2 returns a subtractor with OpenCV-like defaults
// (500-frame history, variance threshold 16).
func NewBackgroundSubtractorMOG2() *BackgroundSubtractorMOG2 {
	return &BackgroundSubtractorMOG2{
		History:         500,
		VarThreshold:    16,
		BackgroundRatio: 0.9,
		nmixtures:       5,
		initVar:         225, // (15 intensity units)^2
		minVar:          4,
	}
}

// Apply feeds the next frame to the model and returns a single-channel
// foreground mask (255 = foreground, 0 = background) the same size as the frame.
// Multi-channel frames are converted to grayscale. Frame dimensions must stay
// constant across calls.
func (b *BackgroundSubtractorMOG2) Apply(frame *cv.Mat) *cv.Mat {
	if frame == nil || frame.Empty() {
		panic("video: BackgroundSubtractorMOG2.Apply requires a non-empty frame")
	}
	gray := toGray(frame)
	if b.model == nil {
		b.rows, b.cols = gray.Rows, gray.Cols
		b.model = make([][]gaussian, b.rows*b.cols)
	} else if gray.Rows != b.rows || gray.Cols != b.cols {
		panic("video: BackgroundSubtractorMOG2.Apply frame size changed")
	}
	b.frame++
	alpha := 1.0 / float64(min2(b.frame, b.History))

	mask := cv.NewMat(b.rows, b.cols, 1)
	for i := 0; i < b.rows*b.cols; i++ {
		x := float64(gray.Data[i])
		fg := b.updatePixel(&b.model[i], x, alpha)
		if fg {
			mask.Data[i] = 255
		}
	}
	return mask
}

// updatePixel updates a single pixel's mixture with sample x and returns whether
// the sample is foreground.
func (b *BackgroundSubtractorMOG2) updatePixel(mix *[]gaussian, x, alpha float64) bool {
	gs := *mix
	match := -1
	for i := range gs {
		d := x - gs[i].mean
		if d*d < b.VarThreshold*gs[i].varc {
			match = i
			break
		}
	}
	if match >= 0 {
		for i := range gs {
			m := 0.0
			if i == match {
				m = 1.0
			}
			gs[i].weight = (1-alpha)*gs[i].weight + alpha*m
		}
		rho := alpha / gs[match].weight
		if rho > 1 {
			rho = 1
		}
		d := x - gs[match].mean
		gs[match].mean += rho * d
		gs[match].varc += rho * (d*d - gs[match].varc)
		if gs[match].varc < b.minVar {
			gs[match].varc = b.minVar
		}
	} else {
		ng := gaussian{weight: alpha, mean: x, varc: b.initVar}
		if len(gs) < b.nmixtures {
			gs = append(gs, ng)
		} else {
			// Replace the lowest-weight component.
			lo := 0
			for i := 1; i < len(gs); i++ {
				if gs[i].weight < gs[lo].weight {
					lo = i
				}
			}
			gs[lo] = ng
		}
	}

	// Normalise weights.
	var sum float64
	for i := range gs {
		sum += gs[i].weight
	}
	if sum > 0 {
		for i := range gs {
			gs[i].weight /= sum
		}
	}
	// Sort by descending weight so the strongest components form the background.
	sort.SliceStable(gs, func(i, j int) bool { return gs[i].weight > gs[j].weight })
	*mix = gs

	// Re-find the matched component's rank after sorting to decide background.
	if match < 0 {
		return true
	}
	var cum float64
	for i := range gs {
		cum += gs[i].weight
		d := x - gs[i].mean
		if d*d < b.VarThreshold*gs[i].varc {
			// This is the (or a) matching component; foreground iff it lies past
			// the background portion of the mixture.
			return cum > b.BackgroundRatio && i > 0
		}
	}
	return true
}

// BackgroundSubtractorKNN classifies pixels with a K-nearest-neighbours model,
// mirroring cv::BackgroundSubtractorKNN. It keeps a rolling set of recent
// grayscale samples per pixel; a pixel is background when enough stored samples
// lie within Dist2Threshold of the current value. Each
// [BackgroundSubtractorKNN.Apply] returns a binary foreground mask and updates
// one sample slot per pixel in round-robin order (deterministic, no randomness).
type BackgroundSubtractorKNN struct {
	// History is the number of samples retained per pixel.
	History int
	// Dist2Threshold is the squared-distance threshold for a "near" sample.
	Dist2Threshold float64
	// KNNSamples is the minimum number of near samples required for background.
	KNNSamples int

	rows    int
	cols    int
	filled  int // number of samples stored so far (up to History)
	slot    int // next round-robin slot to overwrite
	samples [][]uint8
}

// NewBackgroundSubtractorKNN returns a subtractor with OpenCV-like defaults
// (history 500, squared-distance threshold 400, requiring 3 near samples).
func NewBackgroundSubtractorKNN() *BackgroundSubtractorKNN {
	return &BackgroundSubtractorKNN{History: 500, Dist2Threshold: 400, KNNSamples: 3}
}

// Apply feeds the next frame to the model and returns a single-channel
// foreground mask (255 = foreground, 0 = background). Multi-channel frames are
// converted to grayscale. Frame dimensions must stay constant across calls.
func (k *BackgroundSubtractorKNN) Apply(frame *cv.Mat) *cv.Mat {
	if frame == nil || frame.Empty() {
		panic("video: BackgroundSubtractorKNN.Apply requires a non-empty frame")
	}
	gray := toGray(frame)
	n := gray.Rows * gray.Cols
	if k.samples == nil {
		k.rows, k.cols = gray.Rows, gray.Cols
		k.samples = make([][]uint8, k.History)
		for i := range k.samples {
			k.samples[i] = make([]uint8, n)
		}
	} else if gray.Rows != k.rows || gray.Cols != k.cols {
		panic("video: BackgroundSubtractorKNN.Apply frame size changed")
	}

	mask := cv.NewMat(k.rows, k.cols, 1)
	for i := 0; i < n; i++ {
		x := int(gray.Data[i])
		near := 0
		for s := 0; s < k.filled; s++ {
			d := x - int(k.samples[s][i])
			if float64(d*d) <= k.Dist2Threshold {
				near++
			}
		}
		if near < k.KNNSamples {
			mask.Data[i] = 255
		}
		// Update this pixel's sample in the current round-robin slot.
		k.samples[k.slot][i] = gray.Data[i]
	}
	if k.filled < k.History {
		k.filled++
	}
	k.slot = (k.slot + 1) % k.History
	return mask
}

// min2 returns the smaller of two ints.
func min2(a, b int) int {
	if a < b {
		return a
	}
	return b
}
