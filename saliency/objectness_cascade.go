package saliency

import (
	"sort"

	cv "github.com/malcolmston/opencv"
)

// ObjectnessCascade is a two-stage objectness proposer in the spirit of the
// BING cascade of Cheng et al. (CVPR 2014): a fast first stage floods the image
// with cheaply-scored sliding windows, and a slower second stage re-scores only
// the survivors with richer cues and suppresses overlaps.
//
// Stage one scores every window by normed-gradient boundary contrast (the mean
// normed gradient inside the window minus that of a surrounding ring), exactly
// the [ObjectnessBING] cue, and keeps the top StageOneKeep windows. Stage two
// re-ranks each survivor with a combined objectness score:
//
//   - boundary contrast (does the window edge sit on strong gradients?);
//   - saliency coverage (how much of a frequency-tuned saliency map the window
//     captures relative to its area — favouring tight windows on the object);
//     and
//   - a mild size prior.
//
// Greedy non-maximum suppression by intersection-over-union then removes
// near-duplicate boxes, yielding a compact ranked proposal set. Because the
// second stage is a fixed heuristic rather than a trained SVM, scores are
// relative cues, not calibrated probabilities.
//
// Construct one with [NewObjectnessCascade].
type ObjectnessCascade struct {
	// WindowSizes are the square window side lengths evaluated in stage one.
	WindowSizes []int
	// StageOneKeep caps how many stage-one windows advance to stage two.
	StageOneKeep int
	// MaxProposals caps the number of boxes returned after suppression (0 means
	// no cap).
	MaxProposals int
	// NMSThreshold is the intersection-over-union above which the lower-scored
	// of two boxes is suppressed.
	NMSThreshold float64
}

// NewObjectnessCascade returns a cascade with a default range of window sizes,
// 256 stage-one survivors, an IoU suppression threshold of 0.5 and a cap of 32
// proposals.
func NewObjectnessCascade() *ObjectnessCascade {
	return &ObjectnessCascade{
		WindowSizes:  []int{8, 16, 24, 32, 48, 64},
		StageOneKeep: 256,
		MaxProposals: 32,
		NMSThreshold: 0.5,
	}
}

// ComputeObjectness returns candidate object windows for img ranked by
// objectness score (highest first) after two-stage scoring and non-maximum
// suppression. It panics if img is nil or empty.
func (o *ObjectnessCascade) ComputeObjectness(img *cv.Mat) []ObjectnessBox {
	gray := grayPlane(img)
	rows, cols := gray.rows, gray.cols
	ng := normedGradient(gray)
	ngSat := ng.integral()

	// Stage one: cheap boundary-contrast scoring of every window.
	var stage1 []ObjectnessBox
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
				score := windowScore(ngSat, rows, cols, y, x, ws, ring)
				stage1 = append(stage1, ObjectnessBox{X: x, Y: y, W: ws, H: ws, Score: score})
			}
		}
	}
	sort.SliceStable(stage1, func(i, j int) bool { return stage1[i].Score > stage1[j].Score })
	keep := o.StageOneKeep
	if keep > 0 && len(stage1) > keep {
		stage1 = stage1[:keep]
	}

	// Stage two: rich re-scoring using a saliency map's coverage.
	sal := NewStaticSaliencyFrequencyTuned().ComputeSaliency(img)
	salPlane := newPlane(rows, cols)
	for i, v := range sal.Data {
		salPlane.data[i] = float64(v)
	}
	salSat := salPlane.integral()
	maxSide := float64(rows)
	if cols > rows {
		maxSide = float64(cols)
	}
	for i := range stage1 {
		b := &stage1[i]
		ring := b.W / 2
		if ring < 1 {
			ring = 1
		}
		boundary := windowScore(ngSat, rows, cols, b.Y, b.X, b.W, ring)
		inSal := rectSum(salSat, cols, b.Y, b.X, b.Y+b.H-1, b.X+b.W-1)
		coverage := inSal / float64(b.W*b.H) // mean saliency inside window
		sizePrior := 1 - float64(b.W)/(2*maxSide)
		b.Score = 0.5*coverage + 0.4*boundary + 10*sizePrior
	}
	sort.SliceStable(stage1, func(i, j int) bool { return stage1[i].Score > stage1[j].Score })

	kept := nmsBoxes(stage1, o.NMSThreshold)
	if o.MaxProposals > 0 && len(kept) > o.MaxProposals {
		kept = kept[:o.MaxProposals]
	}
	return kept
}

// nmsBoxes performs greedy non-maximum suppression on score-sorted boxes,
// dropping any box whose IoU with an already-kept higher-scored box exceeds thr.
func nmsBoxes(boxes []ObjectnessBox, thr float64) []ObjectnessBox {
	var kept []ObjectnessBox
	for _, b := range boxes {
		drop := false
		for _, k := range kept {
			if iou(b, k) > thr {
				drop = true
				break
			}
		}
		if !drop {
			kept = append(kept, b)
		}
	}
	return kept
}

// iou returns the intersection-over-union of two boxes.
func iou(a, b ObjectnessBox) float64 {
	ax1, ay1 := a.X+a.W, a.Y+a.H
	bx1, by1 := b.X+b.W, b.Y+b.H
	ix0 := maxInt(a.X, b.X)
	iy0 := maxInt(a.Y, b.Y)
	ix1 := minInt(ax1, bx1)
	iy1 := minInt(ay1, by1)
	iw := ix1 - ix0
	ih := iy1 - iy0
	if iw <= 0 || ih <= 0 {
		return 0
	}
	inter := float64(iw * ih)
	union := float64(a.W*a.H+b.W*b.H) - inter
	if union <= 0 {
		return 0
	}
	return inter / union
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}
