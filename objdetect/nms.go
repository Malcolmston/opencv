package objdetect

import (
	"math"
	"sort"

	cv "github.com/malcolmston/opencv"
)

// RectIoU returns the intersection-over-union overlap of two rectangles, a value
// in [0,1]. It is 0 when the rectangles are disjoint (or either is degenerate)
// and 1 when they coincide exactly. This is the standard overlap metric used by
// non-maximum suppression and detection-tracking association.
func RectIoU(a, b cv.Rect) float64 {
	ix0 := maxInt(a.X, b.X)
	iy0 := maxInt(a.Y, b.Y)
	ix1 := minInt2(a.X+a.Width, b.X+b.Width)
	iy1 := minInt2(a.Y+a.Height, b.Y+b.Height)
	iw := ix1 - ix0
	ih := iy1 - iy0
	if iw <= 0 || ih <= 0 {
		return 0
	}
	inter := float64(iw * ih)
	areaA := float64(a.Width * a.Height)
	areaB := float64(b.Width * b.Height)
	union := areaA + areaB - inter
	if union <= 0 {
		return 0
	}
	return inter / union
}

// NMSBoxes performs greedy non-maximum suppression on a set of scored detection
// boxes, mirroring OpenCV's cv::dnn::NMSBoxes. It first discards every box whose
// score is below scoreThreshold, sorts the survivors by descending score, and
// then walks that order keeping a box only if its intersection-over-union with
// every already-kept box is at most nmsThreshold. The returned slice holds the
// indices (into the original boxes slice) of the kept detections, ordered by
// descending score.
//
// boxes and scores must have the same length; it panics otherwise. An
// nmsThreshold of 1 keeps every above-threshold box (no suppression); a value
// near 0 keeps only boxes that barely touch.
func NMSBoxes(boxes []cv.Rect, scores []float64, scoreThreshold, nmsThreshold float64) []int {
	if len(boxes) != len(scores) {
		panic("objdetect: NMSBoxes boxes and scores length mismatch")
	}
	order := make([]int, 0, len(boxes))
	for i, s := range scores {
		if s >= scoreThreshold {
			order = append(order, i)
		}
	}
	sort.SliceStable(order, func(a, b int) bool {
		return scores[order[a]] > scores[order[b]]
	})

	var kept []int
	for _, idx := range order {
		suppressed := false
		for _, k := range kept {
			if RectIoU(boxes[idx], boxes[k]) > nmsThreshold {
				suppressed = true
				break
			}
		}
		if !suppressed {
			kept = append(kept, idx)
		}
	}
	return kept
}

// SoftNMSBoxes performs Gaussian Soft-NMS (Bodla et al., 2017). Instead of hard
// removing overlapping boxes, it decays the score of each remaining box by a
// Gaussian factor exp(-iou²/sigma) relative to the current highest-scoring box,
// repeating until every box has been selected once. Boxes whose decayed score
// falls below scoreThreshold are dropped. It returns the kept indices (into the
// original slices) in descending final-score order together with those final,
// decayed scores.
//
// sigma controls how aggressively overlaps are penalised; the OpenCV default is
// 0.5, and non-positive values are treated as 0.5. boxes and scores must have
// the same length; it panics otherwise.
func SoftNMSBoxes(boxes []cv.Rect, scores []float64, scoreThreshold, sigma float64) (indices []int, keptScores []float64) {
	if len(boxes) != len(scores) {
		panic("objdetect: SoftNMSBoxes boxes and scores length mismatch")
	}
	if sigma <= 0 {
		sigma = 0.5
	}
	n := len(boxes)
	cur := make([]float64, n)
	copy(cur, scores)
	done := make([]bool, n)

	for {
		// Pick the highest-scoring not-yet-selected box.
		best := -1
		var bestScore float64
		for i := 0; i < n; i++ {
			if done[i] {
				continue
			}
			if best == -1 || cur[i] > bestScore {
				best = i
				bestScore = cur[i]
			}
		}
		if best == -1 {
			break
		}
		done[best] = true
		if bestScore >= scoreThreshold {
			indices = append(indices, best)
			keptScores = append(keptScores, bestScore)
		}
		// Decay every remaining box by its overlap with the chosen one.
		for i := 0; i < n; i++ {
			if done[i] {
				continue
			}
			iou := RectIoU(boxes[best], boxes[i])
			cur[i] *= math.Exp(-(iou * iou) / sigma)
		}
	}
	return indices, keptScores
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

// minInt2 is a second min helper (minInt already exists in cascade.go) kept
// separate to avoid touching that file.
func minInt2(a, b int) int {
	if a < b {
		return a
	}
	return b
}
