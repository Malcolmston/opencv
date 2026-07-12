package shape

import (
	"math"
	"sort"

	cv "github.com/malcolmston/opencv"
)

// Defect describes one convexity defect of a contour: a stretch of the contour
// that dips inward, away from its convex hull. Start and End are the hull points
// bounding the defect (with StartIndex and EndIndex their positions in the
// contour), Far is the contour point deepest inside the hull (FarIndex its
// position), and Depth is Far's perpendicular distance from the hull edge
// Start–End, in pixels.
type Defect struct {
	StartIndex int
	EndIndex   int
	FarIndex   int
	Start      cv.Point
	End        cv.Point
	Far        cv.Point
	Depth      float64
}

// ConvexHullIndices returns the convex hull of pts as indices into pts, in the
// order the hull vertices appear when walking the contour (ascending index).
// This is the index form that [ConvexityDefects] consumes, analogous to
// OpenCV's convexHull with returnPoints = false.
//
// Duplicate and collinear-interior points are dropped, so the result holds only
// true hull vertices. Fewer than three input points return their own indices.
func ConvexHullIndices(pts []cv.Point) []int {
	n := len(pts)
	if n < 3 {
		idx := make([]int, n)
		for i := range idx {
			idx[i] = i
		}
		return idx
	}
	hull := cv.ConvexHull(pts)
	// Map each hull vertex back to an input index. Match by coordinate, choosing
	// the first unused input point with those coordinates.
	used := make([]bool, n)
	var idx []int
	for _, hp := range hull {
		for i := 0; i < n; i++ {
			if !used[i] && pts[i].X == hp.X && pts[i].Y == hp.Y {
				used[i] = true
				idx = append(idx, i)
				break
			}
		}
	}
	sort.Ints(idx)
	return idx
}

// ConvexityDefects computes the convexity defects of a contour relative to its
// convex hull. hull holds indices into contour identifying the hull vertices
// (as produced by [ConvexHullIndices]); the indices are taken in contour order.
// For each pair of consecutive hull vertices the contour points lying between
// them are scanned, and the point farthest inside the hull edge — if any lies
// strictly inside — is reported as a [Defect].
//
// The returned defects are ordered by their starting hull vertex. A convex
// contour, or one with fewer than four points or fewer than three hull vertices,
// has no defects and yields nil.
func ConvexityDefects(contour []cv.Point, hull []int) []Defect {
	n := len(contour)
	if n < 4 || len(hull) < 3 {
		return nil
	}
	// Work on hull indices in ascending contour order so consecutive pairs bound
	// a forward arc of the contour.
	h := make([]int, len(hull))
	copy(h, hull)
	sort.Ints(h)

	var defects []Defect
	m := len(h)
	for k := 0; k < m; k++ {
		startIdx := h[k]
		endIdx := h[(k+1)%m]
		a := contour[startIdx]
		b := contour[endIdx]

		// Walk the contour from startIdx to endIdx (forward, wrapping).
		var farIdx int = -1
		var maxDist float64
		for i := (startIdx + 1) % n; i != endIdx; i = (i + 1) % n {
			d := pointLineDistance(contour[i], a, b)
			if d > maxDist {
				maxDist = d
				farIdx = i
			}
		}
		if farIdx >= 0 && maxDist > epsGeom {
			defects = append(defects, Defect{
				StartIndex: startIdx,
				EndIndex:   endIdx,
				FarIndex:   farIdx,
				Start:      a,
				End:        b,
				Far:        contour[farIdx],
				Depth:      maxDist,
			})
		}
	}
	return defects
}

// pointLineDistance returns the perpendicular distance from p to the line
// through a and b (or the distance to a when a == b).
func pointLineDistance(p, a, b cv.Point) float64 {
	dx := float64(b.X - a.X)
	dy := float64(b.Y - a.Y)
	length := math.Hypot(dx, dy)
	if length < epsGeom {
		return math.Hypot(float64(p.X-a.X), float64(p.Y-a.Y))
	}
	num := math.Abs(dy*float64(p.X-a.X) - dx*float64(p.Y-a.Y))
	return num / length
}
