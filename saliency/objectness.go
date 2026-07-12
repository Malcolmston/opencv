package saliency

import (
	"math"
	"sort"

	cv "github.com/malcolmston/opencv"
)

// ObjectnessBox is a candidate object window returned by [ObjectnessBING]. X, Y
// are the top-left corner and W, H the size, all in pixels; Score is the
// objectness measure, higher meaning more object-like. Boxes are returned in
// descending score order.
type ObjectnessBox struct {
	X, Y, W, H int
	Score      float64
}

// ObjectnessBING is a lightweight ("BING-lite") objectness detector inspired by
// Cheng et al., "BING: Binarized Normed Gradients for Objectness Estimation at
// 300fps" (CVPR 2014), the basis of OpenCV's cv::saliency::ObjectnessBING.
//
// The full BING method scores 8×8 binarised normed-gradient (NG) windows with a
// linear model whose weights are learned offline. This port keeps the
// normed-gradient front end but replaces the learned classifier with a fixed
// heuristic: generic objects are gradient-dense regions surrounded by flatter
// background, so a window is scored by how much stronger the mean normed
// gradient is inside it than in a surrounding ring. No training data or weights
// are required, which is why it is a "lite" variant; the trade-off is that
// scores are relative cues rather than calibrated probabilities.
//
// Construct one with [NewObjectnessBING] and call
// [ObjectnessBING.ComputeObjectness] to obtain ranked proposals.
type ObjectnessBING struct {
	// WindowSizes are the side lengths (in pixels) of the square sliding
	// windows evaluated at every position. Windows larger than the image are
	// skipped.
	WindowSizes []int
	// MaxProposals caps the number of boxes returned (0 means no cap).
	MaxProposals int
}

// NewObjectnessBING returns a detector with a default range of window sizes and
// a cap of 64 proposals.
func NewObjectnessBING() *ObjectnessBING {
	return &ObjectnessBING{
		WindowSizes:  []int{8, 16, 32, 64},
		MaxProposals: 64,
	}
}

// normedGradient returns the normed-gradient magnitude of gray as a plane,
// using absolute forward differences (|dx|+|dy|) with replicated borders. This
// is the BING "NG" feature before binarisation.
func normedGradient(gray *plane) *plane {
	ng := newPlane(gray.rows, gray.cols)
	for y := 0; y < gray.rows; y++ {
		for x := 0; x < gray.cols; x++ {
			dx := math.Abs(gray.atReplicate(y, x+1) - gray.atReplicate(y, x-1))
			dy := math.Abs(gray.atReplicate(y+1, x) - gray.atReplicate(y-1, x))
			ng.data[y*gray.cols+x] = dx + dy
		}
	}
	return ng
}

// ComputeObjectness returns candidate object windows for img ranked by
// objectness score (highest first). It panics if img is nil or empty.
func (o *ObjectnessBING) ComputeObjectness(img *cv.Mat) []ObjectnessBox {
	gray := grayPlane(img)
	ng := normedGradient(gray)
	sat := ng.integral()
	rows, cols := gray.rows, gray.cols

	var boxes []ObjectnessBox
	for _, ws := range o.WindowSizes {
		if ws <= 0 || ws > rows || ws > cols {
			continue
		}
		ring := ws / 2
		if ring < 1 {
			ring = 1
		}
		step := ws / 4
		if step < 1 {
			step = 1
		}
		for y := 0; y+ws <= rows; y += step {
			for x := 0; x+ws <= cols; x += step {
				score := windowScore(sat, rows, cols, y, x, ws, ring)
				boxes = append(boxes, ObjectnessBox{X: x, Y: y, W: ws, H: ws, Score: score})
			}
		}
	}

	sort.SliceStable(boxes, func(i, j int) bool {
		return boxes[i].Score > boxes[j].Score
	})
	if o.MaxProposals > 0 && len(boxes) > o.MaxProposals {
		boxes = boxes[:o.MaxProposals]
	}
	return boxes
}

// windowScore returns the mean normed gradient inside the window at (x, y) of
// side ws minus the mean over the surrounding ring of width ring (clamped to
// the image). A high value marks a gradient-dense region set against a flatter
// surround — an object-like window.
func windowScore(sat []float64, rows, cols, y, x, ws, ring int) float64 {
	x1, y1 := x+ws-1, y+ws-1
	inSum := rectSum(sat, cols, y, x, y1, x1)
	inArea := float64(ws * ws)
	inMean := inSum / inArea

	ox0, oy0 := x-ring, y-ring
	ox1, oy1 := x1+ring, y1+ring
	if ox0 < 0 {
		ox0 = 0
	}
	if oy0 < 0 {
		oy0 = 0
	}
	if ox1 > cols-1 {
		ox1 = cols - 1
	}
	if oy1 > rows-1 {
		oy1 = rows - 1
	}
	outSum := rectSum(sat, cols, oy0, ox0, oy1, ox1) - inSum
	outArea := float64((ox1-ox0+1)*(oy1-oy0+1)) - inArea
	if outArea <= 0 {
		return inMean
	}
	return inMean - outSum/outArea
}
