package shape

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// epsGeom is a small tolerance used by the enclosing-shape geometry to absorb
// floating-point round-off in point-in-circle and degeneracy tests.
const epsGeom = 1e-9

// fpoint is an internal fractional 2-D point.
type fpoint struct{ x, y float64 }

// circle is an internal circle with a fractional centre and radius.
type circle struct {
	cx, cy float64
	r      float64
}

// MinEnclosingCircle returns the smallest circle that encloses every point in
// pts, computed with Welzl's move-to-front algorithm. It returns the circle's
// centre (cx, cy) in fractional pixel coordinates and its radius.
//
// The algorithm is expected linear time. To keep results reproducible the input
// order is fixed with a deterministic permutation instead of a random one, so
// the same point set always yields the same circle.
//
// A single point yields a circle of radius zero centred on it; two points yield
// the circle whose diameter is the segment between them. It panics on an empty
// point set.
func MinEnclosingCircle(pts []cv.Point) (cx, cy, radius float64) {
	if len(pts) == 0 {
		panic("shape: MinEnclosingCircle on empty point set")
	}
	fp := make([]fpoint, len(pts))
	for i, p := range pts {
		fp[i] = fpoint{float64(p.X), float64(p.Y)}
	}
	// Deterministic permutation (a fixed-seed linear congruential shuffle) so
	// the algorithm keeps Welzl's good expected running time on adversarial
	// inputs without introducing non-determinism.
	shuffleDeterministic(fp)
	c := welzl(fp)
	return c.cx, c.cy, c.r
}

// shuffleDeterministic performs a Fisher–Yates shuffle driven by a fixed-seed
// LCG, giving a reproducible permutation.
func shuffleDeterministic(p []fpoint) {
	var state uint64 = 0x9e3779b97f4a7c15
	next := func(n int) int {
		state = state*6364136223846793005 + 1442695040888963407
		return int((state >> 33) % uint64(n))
	}
	for i := len(p) - 1; i > 0; i-- {
		j := next(i + 1)
		p[i], p[j] = p[j], p[i]
	}
}

// welzl computes the minimal enclosing circle of p using the iterative
// incremental (move-to-front) formulation of Welzl's algorithm.
func welzl(p []fpoint) circle {
	c := circle{cx: p[0].x, cy: p[0].y, r: 0}
	for i := 1; i < len(p); i++ {
		if inCircle(c, p[i]) {
			continue
		}
		c = circleWith1(p, i)
	}
	return c
}

// circleWith1 returns the minimal circle enclosing p[0..i] with p[i] on its
// boundary.
func circleWith1(p []fpoint, i int) circle {
	c := circle{cx: p[i].x, cy: p[i].y, r: 0}
	for j := 0; j < i; j++ {
		if inCircle(c, p[j]) {
			continue
		}
		c = circleWith2(p, i, j)
	}
	return c
}

// circleWith2 returns the minimal circle enclosing p[0..j] with p[i] and p[j]
// on its boundary.
func circleWith2(p []fpoint, i, j int) circle {
	c := circleFrom2(p[i], p[j])
	for k := 0; k < j; k++ {
		if inCircle(c, p[k]) {
			continue
		}
		c = circleFrom3(p[i], p[j], p[k])
	}
	return c
}

// inCircle reports whether q lies inside or on circle c (within tolerance).
func inCircle(c circle, q fpoint) bool {
	dx := q.x - c.cx
	dy := q.y - c.cy
	return dx*dx+dy*dy <= c.r*c.r+epsGeom*(1+c.r*c.r)
}

// circleFrom2 builds the circle whose diameter is the segment a–b.
func circleFrom2(a, b fpoint) circle {
	cx := (a.x + b.x) / 2
	cy := (a.y + b.y) / 2
	r := math.Hypot(a.x-cx, a.y-cy)
	return circle{cx: cx, cy: cy, r: r}
}

// circleFrom3 builds the circumcircle of the (assumed non-collinear) triangle
// a, b, c. If the points are nearly collinear it falls back to the smallest
// two-point circle among the pairs.
func circleFrom3(a, b, c fpoint) circle {
	ax, ay := a.x, a.y
	bx, by := b.x, b.y
	cx, cy := c.x, c.y
	d := 2 * (ax*(by-cy) + bx*(cy-ay) + cx*(ay-by))
	if math.Abs(d) < epsGeom {
		// Degenerate: pick the largest-diameter pair.
		best := circleFrom2(a, b)
		if cc := circleFrom2(a, c); cc.r > best.r {
			best = cc
		}
		if cc := circleFrom2(b, c); cc.r > best.r {
			best = cc
		}
		return best
	}
	a2 := ax*ax + ay*ay
	b2 := bx*bx + by*by
	c2 := cx*cx + cy*cy
	ux := (a2*(by-cy) + b2*(cy-ay) + c2*(ay-by)) / d
	uy := (a2*(cx-bx) + b2*(ax-cx) + c2*(bx-ax)) / d
	r := math.Hypot(ax-ux, ay-uy)
	return circle{cx: ux, cy: uy, r: r}
}

// MinEnclosingTriangle returns a minimal-area triangle that encloses every
// point in pts, together with its three vertices in fractional pixel
// coordinates. The area is the triangle's area.
//
// The triangle is found over the convex hull of the input: each hull edge is
// tried as a flush side of the triangle (the minimal enclosing triangle always
// has at least one side flush with a hull edge), and for that side the two
// remaining supporting lines are optimised to minimise the triangle's area. The
// smallest triangle over all hull edges is returned.
//
// Fewer than three distinct points cannot bound an area: the returned area is
// zero and the vertices collapse onto the input. It panics on an empty point
// set.
func MinEnclosingTriangle(pts []cv.Point) (area float64, tri [3][2]float64) {
	if len(pts) == 0 {
		panic("shape: MinEnclosingTriangle on empty point set")
	}
	hullPts := cv.ConvexHull(pts)
	h := make([]fpoint, len(hullPts))
	for i, p := range hullPts {
		h[i] = fpoint{float64(p.X), float64(p.Y)}
	}
	if len(h) < 3 {
		// Degenerate hull (point or segment): no enclosed area.
		var t [3][2]float64
		for i := range t {
			p := h[i%len(h)]
			t[i] = [2]float64{p.x, p.y}
		}
		return 0, t
	}
	if len(h) == 3 {
		a := triangleArea(h[0], h[1], h[2])
		return a, [3][2]float64{
			{h[0].x, h[0].y}, {h[1].x, h[1].y}, {h[2].x, h[2].y},
		}
	}

	// Centroid, used to orient outward normals.
	var gx, gy float64
	for _, p := range h {
		gx += p.x
		gy += p.y
	}
	gx /= float64(len(h))
	gy /= float64(len(h))

	bestArea := math.Inf(1)
	var bestTri [3][2]float64
	n := len(h)
	for i := 0; i < n; i++ {
		a := h[i]
		b := h[(i+1)%n]
		dx, dy := b.x-a.x, b.y-a.y
		if math.Hypot(dx, dy) < epsGeom {
			continue
		}
		// Outward normal of this edge: the perpendicular that points away from
		// the hull centroid.
		phi0 := outwardNormalAngle(dx, dy, (a.x+b.x)/2, (a.y+b.y)/2, gx, gy)
		ar, t := optimiseTriangle(h, phi0)
		if ar < bestArea {
			bestArea = ar
			bestTri = t
		}
	}
	return bestArea, bestTri
}

// outwardNormalAngle returns the angle of the edge normal (dx,dy is the edge
// direction) that points from the edge midpoint (mx,my) away from the hull
// centroid (gx,gy).
func outwardNormalAngle(dx, dy, mx, my, gx, gy float64) float64 {
	nx, ny := dy, -dx
	// Flip so the normal points away from the centroid.
	if nx*(mx-gx)+ny*(my-gy) < 0 {
		nx, ny = -nx, -ny
	}
	return math.Atan2(ny, nx)
}

// optimiseTriangle fixes one triangle side flush with the supporting line whose
// outward normal has angle phi0, then searches the two remaining supporting-line
// normal angles to minimise the enclosing triangle's area. It returns the best
// area and its vertices.
func optimiseTriangle(h []fpoint, phi0 float64) (float64, [3][2]float64) {
	// The two free normals must lie strictly between phi0 and phi0+2π to bound a
	// triangle; sample that open interval on a coarse grid, then refine the best
	// sample with a shrinking local search.
	const twoPi = 2 * math.Pi
	const margin = 1e-3
	lo := phi0 + margin
	hi := phi0 + twoPi - margin

	bestArea := math.Inf(1)
	var bestT [3][2]float64
	eval := func(p1, p2 float64) {
		ok, ar, t := triangleFromNormals(h, phi0, p1, p2)
		if ok && ar < bestArea {
			bestArea = ar
			bestT = t
		}
	}

	const grid = 48
	for i := 1; i < grid; i++ {
		p1 := lo + (hi-lo)*float64(i)/float64(grid)
		for j := i + 1; j < grid; j++ {
			p2 := lo + (hi-lo)*float64(j)/float64(grid)
			eval(p1, p2)
		}
	}
	if math.IsInf(bestArea, 1) {
		return bestArea, bestT
	}

	// Local refinement by coordinate descent with a shrinking step.
	// Recover the current best angles by re-searching around the coarse optimum.
	b1, b2 := bestNormals(h, phi0, lo, hi, grid)
	step := (hi - lo) / float64(grid)
	for iter := 0; iter < 40 && step > 1e-12; iter++ {
		improved := false
		for _, d := range [4][2]float64{{step, 0}, {-step, 0}, {0, step}, {0, -step}} {
			c1 := b1 + d[0]
			c2 := b2 + d[1]
			if c1 <= lo || c1 >= hi || c2 <= lo || c2 >= hi {
				continue
			}
			if ok, ar, t := triangleFromNormals(h, phi0, c1, c2); ok && ar < bestArea-1e-15 {
				bestArea = ar
				bestT = t
				b1, b2 = c1, c2
				improved = true
			}
		}
		if !improved {
			step /= 2
		}
	}
	return bestArea, bestT
}

// bestNormals repeats the coarse grid search and returns the (p1,p2) angle pair
// that produced the smallest area, used to seed local refinement.
func bestNormals(h []fpoint, phi0, lo, hi float64, grid int) (float64, float64) {
	best := math.Inf(1)
	var bp1, bp2 float64
	for i := 1; i < grid; i++ {
		p1 := lo + (hi-lo)*float64(i)/float64(grid)
		for j := i + 1; j < grid; j++ {
			p2 := lo + (hi-lo)*float64(j)/float64(grid)
			if ok, ar, _ := triangleFromNormals(h, phi0, p1, p2); ok && ar < best {
				best = ar
				bp1, bp2 = p1, p2
			}
		}
	}
	return bp1, bp2
}

// triangleFromNormals builds the triangle whose three sides are the supporting
// lines of the hull with outward-normal angles phi0, phi1 and phi2. It reports
// whether the three lines bound a finite triangle, and if so its area and
// vertices.
func triangleFromNormals(h []fpoint, phi0, phi1, phi2 float64) (bool, float64, [3][2]float64) {
	// The three supporting half-planes bound a finite enclosing triangle only if
	// their outward normals positively span the plane, i.e. every angular gap
	// between consecutive normals is below π. Otherwise the intersection is
	// unbounded and the pairwise vertices form a spurious (often degenerate)
	// triangle.
	if !normalsSpan(phi0, phi1, phi2) {
		return false, 0, [3][2]float64{}
	}
	n0 := fpoint{math.Cos(phi0), math.Sin(phi0)}
	n1 := fpoint{math.Cos(phi1), math.Sin(phi1)}
	n2 := fpoint{math.Cos(phi2), math.Sin(phi2)}
	h0 := support(h, n0)
	h1 := support(h, n1)
	h2 := support(h, n2)
	v01, ok01 := intersectLines(n0, h0, n1, h1)
	v12, ok12 := intersectLines(n1, h1, n2, h2)
	v20, ok20 := intersectLines(n2, h2, n0, h0)
	if !ok01 || !ok12 || !ok20 {
		return false, 0, [3][2]float64{}
	}
	area := triangleArea(v01, v12, v20)
	if !(area < math.Inf(1)) || area <= 0 {
		return false, 0, [3][2]float64{}
	}
	return true, area, [3][2]float64{
		{v01.x, v01.y}, {v12.x, v12.y}, {v20.x, v20.y},
	}
}

// support returns the supporting offset h = max over hull vertices of n·v, so
// that the half-plane n·x ≤ h contains the hull and its boundary line touches it.
func support(hull []fpoint, n fpoint) float64 {
	best := math.Inf(-1)
	for _, v := range hull {
		if d := n.x*v.x + n.y*v.y; d > best {
			best = d
		}
	}
	return best
}

// intersectLines intersects the lines n1·x = c1 and n2·x = c2, returning false
// when they are (nearly) parallel.
func intersectLines(n1 fpoint, c1 float64, n2 fpoint, c2 float64) (fpoint, bool) {
	det := n1.x*n2.y - n1.y*n2.x
	if math.Abs(det) < 1e-12 {
		return fpoint{}, false
	}
	x := (c1*n2.y - c2*n1.y) / det
	y := (n1.x*c2 - n2.x*c1) / det
	return fpoint{x, y}, true
}

// normalsSpan reports whether three direction angles positively span the plane:
// sorted around the circle, every gap between consecutive directions (including
// the wrap-around gap) is strictly below π. This is the condition for the three
// supporting half-planes to bound a finite triangle.
func normalsSpan(a, b, c float64) bool {
	const twoPi = 2 * math.Pi
	norm := func(x float64) float64 {
		x = math.Mod(x, twoPi)
		if x < 0 {
			x += twoPi
		}
		return x
	}
	xs := []float64{norm(a), norm(b), norm(c)}
	// Sort three values.
	if xs[0] > xs[1] {
		xs[0], xs[1] = xs[1], xs[0]
	}
	if xs[1] > xs[2] {
		xs[1], xs[2] = xs[2], xs[1]
	}
	if xs[0] > xs[1] {
		xs[0], xs[1] = xs[1], xs[0]
	}
	g0 := xs[1] - xs[0]
	g1 := xs[2] - xs[1]
	g2 := twoPi - (xs[2] - xs[0])
	const lim = math.Pi - 1e-7
	return g0 < lim && g1 < lim && g2 < lim
}

// triangleArea returns the (unsigned) area of triangle a,b,c.
func triangleArea(a, b, c fpoint) float64 {
	return math.Abs((b.x-a.x)*(c.y-a.y)-(c.x-a.x)*(b.y-a.y)) / 2
}
