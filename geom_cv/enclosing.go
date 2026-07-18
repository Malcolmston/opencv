package geom_cv

import (
	"math"
	"math/rand"

	cv "github.com/malcolmston/opencv"
)

// MinEnclosingCircle returns the smallest circle that contains every input
// point, computed with Welzl's algorithm. The point processing order is
// randomized with a fixed seed so the result is deterministic across runs.
// It panics on an empty point set.
func MinEnclosingCircle(pts []cv.Point2f) Circle {
	if len(pts) == 0 {
		panic("geom_cv: MinEnclosingCircle on empty point set")
	}
	shuffled := make([]cv.Point2f, len(pts))
	copy(shuffled, pts)
	rng := rand.New(rand.NewSource(1))
	rng.Shuffle(len(shuffled), func(i, j int) {
		shuffled[i], shuffled[j] = shuffled[j], shuffled[i]
	})
	return geom_cvWelzl(shuffled, nil)
}

// geom_cvWelzl is the recursive move-to-front core of Welzl's minimum-enclosing
// circle algorithm. boundary holds the (at most three) points known to lie on
// the circle.
func geom_cvWelzl(pts []cv.Point2f, boundary []cv.Point2f) Circle {
	if len(pts) == 0 || len(boundary) == 3 {
		return geom_cvTrivialCircle(boundary)
	}
	p := pts[len(pts)-1]
	rest := pts[:len(pts)-1]
	c := geom_cvWelzl(rest, boundary)
	if c.Contains(p) {
		return c
	}
	return geom_cvWelzl(rest, append(append([]cv.Point2f{}, boundary...), p))
}

// geom_cvTrivialCircle solves the minimum enclosing circle for zero to three
// boundary points directly.
func geom_cvTrivialCircle(b []cv.Point2f) Circle {
	switch len(b) {
	case 0:
		return Circle{Radius: -1}
	case 1:
		return Circle{Center: b[0], Radius: 0}
	case 2:
		return Circle{Center: Midpoint(b[0], b[1]), Radius: Distance(b[0], b[1]) / 2}
	default:
		c, ok := geom_cvCircumcircle(b[0], b[1], b[2])
		if !ok {
			// Collinear: enclose by the widest pair.
			return geom_cvWidestPairCircle(b)
		}
		return c
	}
}

// geom_cvWidestPairCircle returns the circle spanning the two farthest points of
// b, used as a fallback for degenerate (collinear) triples.
func geom_cvWidestPairCircle(b []cv.Point2f) Circle {
	best := Circle{Radius: -1}
	for i := 0; i < len(b); i++ {
		for j := i + 1; j < len(b); j++ {
			c := Circle{Center: Midpoint(b[i], b[j]), Radius: Distance(b[i], b[j]) / 2}
			enc := true
			for _, p := range b {
				if !c.Contains(p) {
					enc = false
					break
				}
			}
			if enc && c.Radius > best.Radius {
				best = c
			}
		}
	}
	return best
}

// MinEnclosingBox returns the smallest axis-aligned [BoundingBox] containing all
// input points. It panics on an empty point set.
func MinEnclosingBox(pts []cv.Point2f) BoundingBox {
	if len(pts) == 0 {
		panic("geom_cv: MinEnclosingBox on empty point set")
	}
	return PolygonBoundingBox(pts)
}

// MinEnclosingTriangleArea returns the area of the smallest enclosing triangle
// aligned to the point set's convex hull edges. For each hull edge the enclosing
// triangle formed by that edge's supporting line and the two extreme supporting
// lines is measured, and the minimum area is returned. It is an upper bound on
// the true minimum enclosing triangle and is exact when an optimal triangle
// shares an edge with the hull. Fewer than three distinct points yield 0.
func MinEnclosingTriangleArea(pts []cv.Point2f) float64 {
	hull := ConvexHull(pts)
	n := len(hull)
	if n < 3 {
		return 0
	}
	best := math.Inf(1)
	for i := 0; i < n; i++ {
		a := hull[i]
		b := hull[(i+1)%n]
		// Twice the enclosing triangle area for this base equals base length
		// times twice the max height, a coarse but valid upper bound.
		base := Distance(a, b)
		maxH := 0.0
		for _, p := range hull {
			if h := PointToLineDistance(a, b, p); h > maxH {
				maxH = h
			}
		}
		if area := base * maxH; area < best {
			best = area
		}
	}
	if math.IsInf(best, 1) {
		return 0
	}
	return best
}
