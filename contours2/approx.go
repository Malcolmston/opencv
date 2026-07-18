package contours2

import (
	cv "github.com/malcolmston/opencv"
)

// ApproxPolyDP approximates a polygonal curve with fewer vertices using the
// Ramer–Douglas–Peucker algorithm. epsilon is the maximum distance, in pixels,
// between the original curve and its approximation; larger values give coarser
// results. When closed is true the curve is treated as a closed polygon (the
// endpoints are joined before simplification), matching OpenCV's approxPolyDP.
//
// The returned slice is a new allocation; the input is not modified. epsilon
// must be non-negative or the function panics.
func ApproxPolyDP(curve []cv.Point, epsilon float64, closed bool) []cv.Point {
	if epsilon < 0 {
		panic("contours2: ApproxPolyDP requires epsilon >= 0")
	}
	n := len(curve)
	if n < 3 {
		out := make([]cv.Point, n)
		copy(out, curve)
		return out
	}

	if !closed {
		keep := make([]bool, n)
		keep[0] = true
		keep[n-1] = true
		contours2rdp(curve, 0, n-1, epsilon, keep)
		return contours2collect(curve, keep)
	}

	// Closed curve: split at the two mutually farthest points, then run the
	// open simplification on each half, as OpenCV does.
	start := 0
	maxD := -1.0
	for i := 1; i < n; i++ {
		d := float64((curve[i].X-curve[0].X)*(curve[i].X-curve[0].X) +
			(curve[i].Y-curve[0].Y)*(curve[i].Y-curve[0].Y))
		if d > maxD {
			maxD = d
			start = i
		}
	}
	end := 0
	maxD = -1.0
	for i := 0; i < n; i++ {
		d := float64((curve[i].X-curve[start].X)*(curve[i].X-curve[start].X) +
			(curve[i].Y-curve[start].Y)*(curve[i].Y-curve[start].Y))
		if d > maxD {
			maxD = d
			end = i
		}
	}

	// Build a rotated, duplicated index sequence from start around to start.
	seq := make([]cv.Point, 0, n+1)
	idxMap := make([]int, 0, n+1)
	for k := 0; k <= n; k++ {
		idx := (start + k) % n
		seq = append(seq, curve[idx])
		idxMap = append(idxMap, idx)
	}
	// Position of end within seq.
	endPos := (end - start + n) % n

	keep := make([]bool, len(seq))
	keep[0] = true
	keep[len(seq)-1] = true
	keep[endPos] = true
	contours2rdp(seq, 0, endPos, epsilon, keep)
	contours2rdp(seq, endPos, len(seq)-1, epsilon, keep)

	seen := make(map[int]bool)
	out := make([]cv.Point, 0, len(seq))
	for k := 0; k < len(seq)-1; k++ { // drop the duplicated closing point
		if keep[k] && !seen[idxMap[k]] {
			seen[idxMap[k]] = true
			out = append(out, seq[k])
		}
	}
	return out
}

// contours2rdp marks vertices to keep between indices first and last (inclusive)
// of pts according to the Douglas–Peucker recursion.
func contours2rdp(pts []cv.Point, first, last int, epsilon float64, keep []bool) {
	if last <= first+1 {
		return
	}
	maxDist := -1.0
	index := first
	a, b := pts[first], pts[last]
	for i := first + 1; i < last; i++ {
		d := contours2distToLine(pts[i], a, b)
		if d > maxDist {
			maxDist = d
			index = i
		}
	}
	if maxDist > epsilon {
		keep[index] = true
		contours2rdp(pts, first, index, epsilon, keep)
		contours2rdp(pts, index, last, epsilon, keep)
	}
}

// contours2collect returns the kept points of pts in order.
func contours2collect(pts []cv.Point, keep []bool) []cv.Point {
	out := make([]cv.Point, 0, len(pts))
	for i, k := range keep {
		if k {
			out = append(out, pts[i])
		}
	}
	return out
}
