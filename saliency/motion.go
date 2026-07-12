package saliency

import (
	"fmt"
	"math"

	cv "github.com/malcolmston/opencv"
)

// MotionSaliencyBinWangApr2014 detects motion saliency — pixels belonging to
// moving objects — with a per-pixel background model built up over a sequence
// of frames, after Wang & Dudek, "A Fast Self-tuning Background Subtraction
// Algorithm" (CVPR Workshops, April 2014), the method behind OpenCV's
// cv::saliency::MotionSaliencyBinWangApr2014.
//
// Each pixel keeps a small set of background template samples. A new frame's
// pixel is classified as background when it matches enough templates within an
// intensity tolerance, and as foreground (moving) otherwise. Matched templates
// are nudged toward the observed value so the model tracks slow illumination
// drift; unmatched pixels receive a slow blind update so that genuinely
// permanent scene changes are eventually absorbed. This is a compact,
// deterministic rendering of the two-model Bin-Wang scheme: it uses a single
// multi-template model and fixed (rather than randomised) template replacement,
// which keeps results reproducible.
//
// The detector is stateful and processes frames in order. Create one with
// [NewMotionSaliencyBinWangApr2014] and call [MotionSaliencyBinWangApr2014.ComputeSaliency]
// once per frame. The first frame seeds the model and returns an all-zero map.
type MotionSaliencyBinWangApr2014 struct {
	// NumTemplates is the number of background samples kept per pixel.
	NumTemplates int
	// Threshold is the maximum absolute intensity difference (0–255) at which a
	// pixel is still considered to match a background template.
	Threshold float64
	// MinMatches is how many templates a pixel must match to count as
	// background.
	MinMatches int
	// LearningRate blends a matched background pixel into its closest template
	// (0 keeps the model frozen, 1 replaces it outright).
	LearningRate float64

	rows, cols  int
	templates   [][]float64
	initialized bool
}

// NewMotionSaliencyBinWangApr2014 returns a detector sized for rows×cols frames
// with sensible defaults (four templates, an intensity tolerance of 40 and a
// 0.1 learning rate). It panics if either dimension is not positive.
func NewMotionSaliencyBinWangApr2014(rows, cols int) *MotionSaliencyBinWangApr2014 {
	if rows <= 0 || cols <= 0 {
		panic(fmt.Sprintf("saliency: MotionSaliency requires positive size, got %dx%d", rows, cols))
	}
	return &MotionSaliencyBinWangApr2014{
		NumTemplates: 4,
		Threshold:    40,
		MinMatches:   1,
		LearningRate: 0.1,
		rows:         rows,
		cols:         cols,
	}
}

// Reset discards the learned background model so the next frame re-seeds it.
func (m *MotionSaliencyBinWangApr2014) Reset() {
	m.templates = nil
	m.initialized = false
}

// ComputeSaliency returns the binary motion-saliency map for one frame: a
// single-channel [cv.Mat] the same size as the configured frame size, with
// moving pixels set to 255 and background to 0. The frame must match the size
// passed to the constructor. The very first frame seeds the background model
// and yields an all-zero map. It panics on a size mismatch or empty frame.
func (m *MotionSaliencyBinWangApr2014) ComputeSaliency(img *cv.Mat) *cv.Mat {
	gray := grayPlane(img)
	if gray.rows != m.rows || gray.cols != m.cols {
		panic(fmt.Sprintf("saliency: MotionSaliency frame is %dx%d, want %dx%d",
			gray.rows, gray.cols, m.rows, m.cols))
	}

	out := cv.NewMat(m.rows, m.cols, 1)
	if !m.initialized {
		m.templates = make([][]float64, m.NumTemplates)
		for t := range m.templates {
			m.templates[t] = make([]float64, len(gray.data))
			copy(m.templates[t], gray.data)
		}
		m.initialized = true
		return out
	}

	minMatches := m.MinMatches
	if minMatches < 1 {
		minMatches = 1
	}
	for i, v := range gray.data {
		matches := 0
		best := math.Inf(1)
		bestIdx := 0
		for t := range m.templates {
			d := math.Abs(v - m.templates[t][i])
			if d <= m.Threshold {
				matches++
			}
			if d < best {
				best = d
				bestIdx = t
			}
		}
		if matches >= minMatches {
			// Background: track the value with the configured learning rate.
			m.templates[bestIdx][i] += m.LearningRate * (v - m.templates[bestIdx][i])
		} else {
			// Foreground: flag as moving and absorb slowly to adapt to
			// permanent changes without erasing the model.
			out.Data[i] = 255
			m.templates[bestIdx][i] += 0.01 * (v - m.templates[bestIdx][i])
		}
	}
	return out
}
