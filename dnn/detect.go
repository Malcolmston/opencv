package dnn

import (
	"fmt"
	"math"
	"sort"
)

// Box is an axis-aligned rectangle given by its top-left corner and size, the
// [x, y, width, height] convention used by OpenCV detection heads.
type Box struct {
	X, Y, W, H float64
}

// area returns the box area, clamped to zero for degenerate boxes.
func (b Box) area() float64 {
	if b.W <= 0 || b.H <= 0 {
		return 0
	}
	return b.W * b.H
}

// IoU returns the intersection-over-union overlap of b and other in [0,1]. It is
// zero when the boxes do not overlap or either is degenerate.
func (b Box) IoU(other Box) float64 {
	ax2, ay2 := b.X+b.W, b.Y+b.H
	bx2, by2 := other.X+other.W, other.Y+other.H
	ix1 := math.Max(b.X, other.X)
	iy1 := math.Max(b.Y, other.Y)
	ix2 := math.Min(ax2, bx2)
	iy2 := math.Min(ay2, by2)
	iw := ix2 - ix1
	ih := iy2 - iy1
	if iw <= 0 || ih <= 0 {
		return 0
	}
	inter := iw * ih
	union := b.area() + other.area() - inter
	if union <= 0 {
		return 0
	}
	return inter / union
}

// NMSBoxes performs greedy non-maximum suppression, the standard filter applied
// to a detector's raw boxes. Candidates whose score is below scoreThreshold are
// discarded; the rest are considered from highest score to lowest, each kept
// box suppressing every later box that overlaps it by more than nmsThreshold
// intersection-over-union. It returns the indices of the surviving boxes in
// descending-score order, mirroring cv::dnn::NMSBoxes.
//
// boxes and scores must have equal length. nmsThreshold is an IoU in [0,1].
func NMSBoxes(boxes []Box, scores []float64, scoreThreshold, nmsThreshold float64) []int {
	if len(boxes) != len(scores) {
		panic(fmt.Sprintf("dnn: NMSBoxes has %d boxes but %d scores", len(boxes), len(scores)))
	}
	// Collect candidates above the score threshold.
	cand := make([]int, 0, len(scores))
	for i, s := range scores {
		if s >= scoreThreshold {
			cand = append(cand, i)
		}
	}
	// Sort by score descending; ties broken by original index for determinism.
	sort.SliceStable(cand, func(a, b int) bool {
		if scores[cand[a]] != scores[cand[b]] {
			return scores[cand[a]] > scores[cand[b]]
		}
		return cand[a] < cand[b]
	})

	suppressed := make([]bool, len(cand))
	keep := make([]int, 0, len(cand))
	for i := 0; i < len(cand); i++ {
		if suppressed[i] {
			continue
		}
		ci := cand[i]
		keep = append(keep, ci)
		for j := i + 1; j < len(cand); j++ {
			if suppressed[j] {
				continue
			}
			if boxes[ci].IoU(boxes[cand[j]]) > nmsThreshold {
				suppressed[j] = true
			}
		}
	}
	return keep
}

// Classification pairs a class index with its score, as produced by
// [ClassifyTopK].
type Classification struct {
	// Index is the class index (position along the scored axis).
	Index int
	// Score is the value at that index.
	Score float64
}

// ClassifyTopK returns the k highest-scoring classes from a classifier output,
// sorted by descending score. The input must be a rank-1 [classes] tensor or a
// rank-2 [1, classes] tensor (a single sample). Ties are broken by ascending
// class index so the result is deterministic; k is clamped to the class count.
func ClassifyTopK(scores *Tensor, k int) []Classification {
	if scores == nil {
		panic("dnn: ClassifyTopK given a nil tensor")
	}
	var data []float64
	switch {
	case scores.Dims() == 1:
		data = scores.Data
	case scores.Dims() == 2 && scores.Shape[0] == 1:
		data = scores.Data
	default:
		panic(fmt.Sprintf("dnn: ClassifyTopK needs a [classes] or [1, classes] tensor, got %s", scores))
	}
	if k < 0 {
		panic(fmt.Sprintf("dnn: ClassifyTopK k must be >= 0, got %d", k))
	}
	all := make([]Classification, len(data))
	for i, v := range data {
		all[i] = Classification{Index: i, Score: v}
	}
	sort.SliceStable(all, func(a, b int) bool {
		if all[a].Score != all[b].Score {
			return all[a].Score > all[b].Score
		}
		return all[a].Index < all[b].Index
	})
	if k > len(all) {
		k = len(all)
	}
	return all[:k]
}
