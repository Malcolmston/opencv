package shapefit

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// PointsFromMat extracts the coordinates of the foreground pixels of a binary
// or grayscale image as a point set. A pixel counts as foreground when its
// first-channel sample is strictly greater than threshold. The returned points
// are in scan order (row by row, left to right), so the result is deterministic.
func PointsFromMat(m *cv.Mat, threshold uint8) []cv.Point2f {
	if m == nil || m.Empty() {
		return nil
	}
	var pts []cv.Point2f
	for y := 0; y < m.Rows; y++ {
		for x := 0; x < m.Cols; x++ {
			if m.At(y, x, 0) > threshold {
				pts = append(pts, cv.Point2f{X: float64(x), Y: float64(y)})
			}
		}
	}
	return pts
}

// PointsToMat rasterizes a point set into a new single-channel Mat of the given
// size. Points that fall inside the image set their pixel to 255; points
// outside are ignored. This is the inverse companion of [PointsFromMat], useful
// for visualizing or re-processing a fitted point set with the parent library.
func PointsToMat(pts []cv.Point2f, rows, cols int) *cv.Mat {
	m := cv.NewMat(rows, cols, 1)
	for _, p := range pts {
		x := int(math.Round(p.X))
		y := int(math.Round(p.Y))
		if x >= 0 && x < cols && y >= 0 && y < rows {
			m.Set(y, x, 0, 255)
		}
	}
	return m
}

// Centroid returns the arithmetic mean of the point set. It returns the zero
// point for an empty set.
func Centroid(pts []cv.Point2f) cv.Point2f {
	if len(pts) == 0 {
		return cv.Point2f{}
	}
	var sx, sy float64
	for _, p := range pts {
		sx += p.X
		sy += p.Y
	}
	n := float64(len(pts))
	return cv.Point2f{X: sx / n, Y: sy / n}
}

// BoundingBox returns the axis-aligned bounding box of the point set as its
// minimum and maximum corners. Both corners are the zero point for an empty set.
func BoundingBox(pts []cv.Point2f) (min, max cv.Point2f) {
	if len(pts) == 0 {
		return cv.Point2f{}, cv.Point2f{}
	}
	min = pts[0]
	max = pts[0]
	for _, p := range pts[1:] {
		if p.X < min.X {
			min.X = p.X
		}
		if p.Y < min.Y {
			min.Y = p.Y
		}
		if p.X > max.X {
			max.X = p.X
		}
		if p.Y > max.Y {
			max.Y = p.Y
		}
	}
	return min, max
}

// EstimateOrientations returns, for each point, the local edge orientation in
// radians in the range [0, π), estimated by fitting a total-least-squares line
// to the point and its k nearest neighbours. Isolated points (fewer than two
// neighbours) are assigned orientation 0. The result is used by the
// [GeneralizedHough] transform to index its R-table, but is exported because it
// is a generally useful primitive for unoriented edge point sets.
func EstimateOrientations(pts []cv.Point2f, k int) []float64 {
	n := len(pts)
	out := make([]float64, n)
	if n == 0 {
		return out
	}
	if k < 1 {
		k = 1
	}
	for i := range pts {
		neigh := shapefitKNearest(pts, i, k)
		if len(neigh) < 1 {
			continue
		}
		local := make([]cv.Point2f, 0, len(neigh)+1)
		local = append(local, pts[i])
		for _, j := range neigh {
			local = append(local, pts[j])
		}
		l, err := FitLine(local)
		if err != nil {
			continue
		}
		// Orientation of the line is perpendicular to its normal (a, b).
		ang := math.Atan2(-l.A, l.B)
		if ang < 0 {
			ang += math.Pi
		}
		if ang >= math.Pi {
			ang -= math.Pi
		}
		out[i] = ang
	}
	return out
}

// shapefitKNearest returns the indices of the k points nearest to pts[idx],
// excluding idx itself, ordered by increasing distance.
func shapefitKNearest(pts []cv.Point2f, idx, k int) []int {
	type nd struct {
		i int
		d float64
	}
	cand := make([]nd, 0, len(pts))
	p := pts[idx]
	for j := range pts {
		if j == idx {
			continue
		}
		dx := pts[j].X - p.X
		dy := pts[j].Y - p.Y
		cand = append(cand, nd{j, dx*dx + dy*dy})
	}
	// Partial selection sort of the k smallest (k is small).
	if k > len(cand) {
		k = len(cand)
	}
	for a := 0; a < k; a++ {
		best := a
		for b := a + 1; b < len(cand); b++ {
			if cand[b].d < cand[best].d {
				best = b
			}
		}
		cand[a], cand[best] = cand[best], cand[a]
	}
	out := make([]int, k)
	for a := 0; a < k; a++ {
		out[a] = cand[a].i
	}
	return out
}
