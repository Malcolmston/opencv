package template2

import "math"

// SortByScore orders matches best-first in place: descending score when
// higherIsBetter is true, ascending otherwise. The sort is stable.
func SortByScore(matches []Match, higherIsBetter bool) {
	sortMatches(matches, higherIsBetter)
}

// FilterByScore returns the subset of matches whose score passes threshold: at
// least threshold when higherIsBetter is true, at most threshold otherwise. The
// input order is preserved.
func FilterByScore(matches []Match, threshold float64, higherIsBetter bool) []Match {
	var out []Match
	for _, m := range matches {
		if passesThreshold(m.Score, threshold, higherIsBetter) {
			out = append(out, m)
		}
	}
	return out
}

// NonMaxSuppression greedily prunes overlapping detections. It repeatedly takes
// the best remaining [Match] (the highest score when higherIsBetter is true,
// otherwise the lowest) and discards every other match whose
// intersection-over-union overlap with it is at least iouThreshold. The kept
// matches are returned best-first. The input slice is not modified.
func NonMaxSuppression(matches []Match, iouThreshold float64, higherIsBetter bool) []Match {
	if len(matches) == 0 {
		return nil
	}
	order := make([]Match, len(matches))
	copy(order, matches)
	sortMatches(order, higherIsBetter)

	suppressed := make([]bool, len(order))
	var kept []Match
	for i := range order {
		if suppressed[i] {
			continue
		}
		kept = append(kept, order[i])
		for j := i + 1; j < len(order); j++ {
			if suppressed[j] {
				continue
			}
			if order[i].IoU(order[j]) >= iouThreshold {
				suppressed[j] = true
			}
		}
	}
	return kept
}

// NonMaxSuppressionDistance greedily prunes detections whose centres lie close
// together. It repeatedly takes the best remaining [Match] and discards every
// other match whose centre is within minCenterDistance pixels (Euclidean) of
// it. This is useful when matches all share one template size, so centre
// distance is a simpler proxy than overlap. The kept matches are returned
// best-first; the input slice is not modified.
func NonMaxSuppressionDistance(matches []Match, minCenterDistance float64, higherIsBetter bool) []Match {
	if len(matches) == 0 {
		return nil
	}
	order := make([]Match, len(matches))
	copy(order, matches)
	sortMatches(order, higherIsBetter)

	suppressed := make([]bool, len(order))
	var kept []Match
	minSq := minCenterDistance * minCenterDistance
	for i := range order {
		if suppressed[i] {
			continue
		}
		kept = append(kept, order[i])
		cxi, cyi := order[i].CenterX(), order[i].CenterY()
		for j := i + 1; j < len(order); j++ {
			if suppressed[j] {
				continue
			}
			dx := order[j].CenterX() - cxi
			dy := order[j].CenterY() - cyi
			if dx*dx+dy*dy < minSq || math.Abs(dx*dx+dy*dy-minSq) < 1e-9 {
				suppressed[j] = true
			}
		}
	}
	return kept
}
