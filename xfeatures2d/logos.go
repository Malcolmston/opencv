package xfeatures2d

import (
	"math"
	"sort"
)

// LOGOSMatcher filters putative matches with Local Geometric Support (LOGOS), a
// port in the spirit of OpenCV's cv::xfeatures2d::matchLOGOS.
//
// LOGOS keeps a match only when its immediate neighbourhood of matches moves
// consistently under the local similarity transform implied by the matched
// keypoints' own scale and orientation. For a candidate match its NeighbourCount
// nearest neighbours (in the query image) are examined: each neighbour votes
// "consistent" when the query-side displacement, rotated by the candidate's
// orientation change and scaled by its scale change, predicts the train-side
// displacement to within Tolerance pixels. A match with at least MinSupport
// consistent neighbours survives. This uses each keypoint's Angle and Size, so
// feed keypoints from a descriptor that assigns them (for example [SURF] or
// [FREAK]).
type LOGOSMatcher struct {
	// NeighbourCount is how many nearest query-image neighbours are examined.
	NeighbourCount int
	// MinSupport is the minimum number of geometrically consistent neighbours.
	MinSupport int
	// Tolerance is the prediction error, in pixels, allowed for a neighbour to
	// count as consistent.
	Tolerance float64
}

// NewLOGOSMatcher returns a LOGOSMatcher with sensible defaults.
func NewLOGOSMatcher() *LOGOSMatcher {
	return &LOGOSMatcher{NeighbourCount: 8, MinSupport: 3, Tolerance: 6}
}

// Filter returns the subset of matches with enough local geometric support.
// kp1/kp2 are the keypoints the matches index; their Angle (degrees) and Size
// fields drive the local similarity transform.
func (l *LOGOSMatcher) Filter(kp1, kp2 []KeyPoint, matches []DMatch) []DMatch {
	if len(matches) == 0 {
		return nil
	}
	k := l.NeighbourCount
	if k < 1 {
		k = 1
	}
	var out []DMatch
	for i, mi := range matches {
		qi := kp1[mi.QueryIdx]
		ti := kp2[mi.TrainIdx]

		// The candidate's own local similarity transform (query -> train).
		dAngle := (ti.Angle - qi.Angle) * math.Pi / 180
		scaleRatio := 1.0
		if qi.Size > 0 && ti.Size > 0 {
			scaleRatio = ti.Size / qi.Size
		}
		ca := scaleRatio * math.Cos(dAngle)
		sa := scaleRatio * math.Sin(dAngle)

		// Nearest neighbours of the candidate in the query image.
		type nb struct {
			idx int
			d2  float64
		}
		nbs := make([]nb, 0, len(matches))
		for j, mj := range matches {
			if j == i {
				continue
			}
			qj := kp1[mj.QueryIdx]
			dx := float64(qj.Pt.X - qi.Pt.X)
			dy := float64(qj.Pt.Y - qi.Pt.Y)
			nbs = append(nbs, nb{j, dx*dx + dy*dy})
		}
		sort.Slice(nbs, func(a, b int) bool { return nbs[a].d2 < nbs[b].d2 })
		if len(nbs) > k {
			nbs = nbs[:k]
		}

		support := 0
		for _, n := range nbs {
			mj := matches[n.idx]
			qj := kp1[mj.QueryIdx]
			tj := kp2[mj.TrainIdx]
			qdx := float64(qj.Pt.X - qi.Pt.X)
			qdy := float64(qj.Pt.Y - qi.Pt.Y)
			// Predicted train-side displacement.
			predX := ca*qdx - sa*qdy
			predY := sa*qdx + ca*qdy
			tdx := float64(tj.Pt.X - ti.Pt.X)
			tdy := float64(tj.Pt.Y - ti.Pt.Y)
			if math.Hypot(predX-tdx, predY-tdy) <= l.Tolerance {
				support++
			}
		}
		if support >= l.MinSupport {
			out = append(out, mi)
		}
	}
	return out
}
