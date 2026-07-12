package objdetect

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// rectsSimilar reports whether two rectangles are close enough to belong to the
// same cluster, using OpenCV's relative-size criterion: their corner and size
// differences must all be within eps times their average side length.
func rectsSimilar(a, b cv.Rect, eps float64) bool {
	delta := eps * float64(minInt2(a.Width, b.Width)+minInt2(a.Height, b.Height)) * 0.5
	return math.Abs(float64(a.X-b.X)) <= delta &&
		math.Abs(float64(a.Y-b.Y)) <= delta &&
		math.Abs(float64(a.Width-b.Width)) <= delta &&
		math.Abs(float64(a.Height-b.Height)) <= delta
}

// clusterRects assigns every rectangle a cluster label using single-linkage
// agglomeration under rectsSimilar, returning the labels and the cluster count.
func clusterRects(rects []cv.Rect, eps float64) (labels []int, nClusters int) {
	n := len(rects)
	labels = make([]int, n)
	for i := range labels {
		labels[i] = -1
	}
	for i := 0; i < n; i++ {
		if labels[i] != -1 {
			continue
		}
		labels[i] = nClusters
		// Propagate the label transitively (single linkage).
		changed := true
		for changed {
			changed = false
			for j := 0; j < n; j++ {
				if labels[j] != -1 {
					continue
				}
				for k := 0; k < n; k++ {
					if labels[k] == nClusters && rectsSimilar(rects[j], rects[k], eps) {
						labels[j] = nClusters
						changed = true
						break
					}
				}
			}
		}
		nClusters++
	}
	return labels, nClusters
}

// GroupRectangles clusters similar rectangles and returns one averaged
// rectangle per cluster whose membership is at least minNeighbors, following the
// semantics of OpenCV's cv::groupRectangles. Two rectangles join the same
// cluster when their positions and sizes agree to within eps (relative to their
// side lengths); eps <= 0 is treated as the OpenCV default of 0.2.
//
// When minNeighbors is 0 the result is every input rectangle unchanged. A
// cluster of a single rectangle is dropped whenever minNeighbors > 1, matching
// OpenCV's behaviour of requiring corroborating overlaps.
func GroupRectangles(rects []cv.Rect, minNeighbors int, eps float64) []cv.Rect {
	if len(rects) == 0 {
		return nil
	}
	if minNeighbors <= 0 {
		out := make([]cv.Rect, len(rects))
		copy(out, rects)
		return out
	}
	if eps <= 0 {
		eps = 0.2
	}
	labels, nClusters := clusterRects(rects, eps)

	type acc struct{ x, y, w, h, count int }
	accs := make([]acc, nClusters)
	for i, l := range labels {
		accs[l].x += rects[i].X
		accs[l].y += rects[i].Y
		accs[l].w += rects[i].Width
		accs[l].h += rects[i].Height
		accs[l].count++
	}

	var out []cv.Rect
	for _, a := range accs {
		if a.count < minNeighbors {
			continue
		}
		out = append(out, cv.Rect{
			X:      a.x / a.count,
			Y:      a.y / a.count,
			Width:  a.w / a.count,
			Height: a.h / a.count,
		})
	}
	return out
}

// GroupRectanglesWeights clusters rectangles like [GroupRectangles] but also
// carries a per-rectangle score. Each surviving cluster's representative
// rectangle is the membership average and its returned weight is the maximum
// score among the cluster's members, mirroring OpenCV's groupRectangles overload
// that reports the strongest response of each merged group. rects and weights
// must have equal length; it panics otherwise.
func GroupRectanglesWeights(rects []cv.Rect, weights []float64, minNeighbors int, eps float64) ([]cv.Rect, []float64) {
	if len(rects) != len(weights) {
		panic("objdetect: GroupRectanglesWeights rects and weights length mismatch")
	}
	if len(rects) == 0 {
		return nil, nil
	}
	if minNeighbors <= 0 {
		outR := make([]cv.Rect, len(rects))
		outW := make([]float64, len(weights))
		copy(outR, rects)
		copy(outW, weights)
		return outR, outW
	}
	if eps <= 0 {
		eps = 0.2
	}
	labels, nClusters := clusterRects(rects, eps)

	type acc struct {
		x, y, w, h, count int
		maxW              float64
		hasW              bool
	}
	accs := make([]acc, nClusters)
	for i, l := range labels {
		accs[l].x += rects[i].X
		accs[l].y += rects[i].Y
		accs[l].w += rects[i].Width
		accs[l].h += rects[i].Height
		accs[l].count++
		if !accs[l].hasW || weights[i] > accs[l].maxW {
			accs[l].maxW = weights[i]
			accs[l].hasW = true
		}
	}

	var outR []cv.Rect
	var outW []float64
	for _, a := range accs {
		if a.count < minNeighbors {
			continue
		}
		outR = append(outR, cv.Rect{
			X:      a.x / a.count,
			Y:      a.y / a.count,
			Width:  a.w / a.count,
			Height: a.h / a.count,
		})
		outW = append(outW, a.maxW)
	}
	return outR, outW
}
