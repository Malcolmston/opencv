package shapefit

import (
	"math"
	"sort"

	cv "github.com/malcolmston/opencv"
)

// EllipseDetection is an ellipse found by [DetectEllipses] together with the
// accumulator support that produced it.
type EllipseDetection struct {
	// Ellipse is the detected ellipse.
	Ellipse Ellipse
	// Votes is the number of points that supported the minor-axis estimate.
	Votes int
}

// DetectEllipses detects ellipses in a point set using the Xie–Ji method. Every
// pair of points is hypothesized as the endpoints of a major axis, fixing the
// center, orientation and semi-major length; the remaining points then vote for
// a semi-minor length via a one-dimensional accumulator. A pair whose peak
// exceeds minVotes yields a detection. Detections are filtered so their
// semi-major length lies in [minMajor, maxMajor] (pass maxMajor ≤ 0 to disable
// the upper bound) and overlapping detections sharing a center within
// minCenterDist are suppressed in favor of the higher-voted one.
//
// The algorithm is O(N²·log) in the number of points and is the heaviest
// routine in the package; keep the input point count modest.
func DetectEllipses(pts []cv.Point2f, minVotes int, minMajor, maxMajor, minCenterDist float64) []EllipseDetection {
	n := len(pts)
	if n < 5 || minVotes < 1 {
		return nil
	}
	var dets []EllipseDetection
	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			p1 := pts[i]
			p2 := pts[j]
			dx := p2.X - p1.X
			dy := p2.Y - p1.Y
			a := math.Hypot(dx, dy) / 2 // semi-major candidate
			if a < minMajor || (maxMajor > 0 && a > maxMajor) {
				continue
			}
			cx := (p1.X + p2.X) / 2
			cy := (p1.Y + p2.Y) / 2
			alpha := math.Atan2(dy, dx)
			// Accumulate candidate semi-minor lengths in integer-pixel bins.
			maxB := int(a) + 1
			acc := make([]int, maxB+1)
			a2 := a * a
			for k := 0; k < n; k++ {
				if k == i || k == j {
					continue
				}
				p3 := pts[k]
				ddx := p3.X - cx
				ddy := p3.Y - cy
				d := math.Hypot(ddx, ddy)
				if d < shapefitEps || d > a {
					continue
				}
				f := math.Hypot(p3.X-p2.X, p3.Y-p2.Y)
				// cos(tau) from the triangle center–p3–p2.
				cosTau := (a2 + d*d - f*f) / (2 * a * d)
				if cosTau < -1 {
					cosTau = -1
				} else if cosTau > 1 {
					cosTau = 1
				}
				cos2 := cosTau * cosTau
				denom := a2 - d*d*cos2
				if denom < shapefitEps {
					continue
				}
				b2 := (a2 * d * d * (1 - cos2)) / denom
				if b2 <= 0 {
					continue
				}
				b := math.Sqrt(b2)
				bi := int(math.Round(b))
				if bi >= 0 && bi <= maxB {
					acc[bi]++
				}
			}
			// Find the strongest semi-minor bin.
			bestBin, bestVotes := 0, 0
			for bi, v := range acc {
				if v > bestVotes {
					bestVotes = v
					bestBin = bi
				}
			}
			if bestVotes < minVotes || bestBin == 0 {
				continue
			}
			semiMinor := float64(bestBin)
			if semiMinor > a {
				semiMinor = a
			}
			dets = append(dets, EllipseDetection{
				Ellipse: Ellipse{
					Center:    cv.Point2f{X: cx, Y: cy},
					SemiMajor: a,
					SemiMinor: semiMinor,
					Angle:     shapefitWrapPi(alpha),
				},
				Votes: bestVotes,
			})
		}
	}
	sort.SliceStable(dets, func(a, b int) bool { return dets[a].Votes > dets[b].Votes })
	// Suppress near-duplicate centers.
	var out []EllipseDetection
	for _, d := range dets {
		keep := true
		for _, k := range out {
			if math.Hypot(d.Ellipse.Center.X-k.Ellipse.Center.X, d.Ellipse.Center.Y-k.Ellipse.Center.Y) < minCenterDist {
				keep = false
				break
			}
		}
		if keep {
			out = append(out, d)
		}
	}
	return out
}

// RectangleDetection is a rectangle recovered from a set of corner points by
// [DetectRectangles].
type RectangleDetection struct {
	// Corners are the four rectangle corners in order around the perimeter.
	Corners [4]cv.Point2f
	// Center is the rectangle center (the shared midpoint of the diagonals).
	Center cv.Point2f
}

// RotatedRect converts the detection to a
// [github.com/malcolmston/opencv.RotatedRect] with its angle in degrees, for
// interoperation with the parent library's drawing and geometry routines.
func (r RectangleDetection) RotatedRect() cv.RotatedRect {
	c := r.Corners
	// Two adjacent edges give width, height and angle.
	w := math.Hypot(c[1].X-c[0].X, c[1].Y-c[0].Y)
	h := math.Hypot(c[3].X-c[0].X, c[3].Y-c[0].Y)
	ang := math.Atan2(c[1].Y-c[0].Y, c[1].X-c[0].X)
	return cv.RotatedRect{
		CenterX: r.Center.X,
		CenterY: r.Center.Y,
		Width:   w,
		Height:  h,
		Angle:   ang * 180 / math.Pi,
	}
}

// DetectRectangles finds rectangles among a set of corner points. It uses the
// property that the two diagonals of a rectangle share the same midpoint and
// the same length: corner pairs are grouped by (rounded midpoint, rounded
// half-diagonal), and any two pairs in the same group whose four endpoints form
// right angles (within tol) constitute a rectangle. tol is the tolerance, in
// pixels, applied to the midpoint/length quantization and the right-angle test.
// Detected rectangles are returned with corners ordered around the perimeter.
func DetectRectangles(corners []cv.Point2f, tol float64) []RectangleDetection {
	n := len(corners)
	if n < 4 {
		return nil
	}
	if tol < 1 {
		tol = 1
	}
	type pair struct {
		i, j int
	}
	type key struct {
		mx, my, d int
	}
	groups := make(map[key][]pair)
	q := func(v float64) int { return int(math.Round(v / tol)) }
	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			mx := (corners[i].X + corners[j].X) / 2
			my := (corners[i].Y + corners[j].Y) / 2
			d := math.Hypot(corners[i].X-corners[j].X, corners[i].Y-corners[j].Y)
			k := key{q(mx), q(my), q(d)}
			groups[k] = append(groups[k], pair{i, j})
		}
	}
	seen := make(map[[4]int]bool)
	var out []RectangleDetection
	// Deterministic iteration over groups.
	keys := make([]key, 0, len(groups))
	for k := range groups {
		keys = append(keys, k)
	}
	sort.Slice(keys, func(a, b int) bool {
		if keys[a].mx != keys[b].mx {
			return keys[a].mx < keys[b].mx
		}
		if keys[a].my != keys[b].my {
			return keys[a].my < keys[b].my
		}
		return keys[a].d < keys[b].d
	})
	for _, k := range keys {
		ps := groups[k]
		for a := 0; a < len(ps); a++ {
			for b := a + 1; b < len(ps); b++ {
				idx := [4]int{ps[a].i, ps[a].j, ps[b].i, ps[b].j}
				// Order corners: diagonal1 endpoints i,j ; diagonal2 endpoints.
				A := corners[ps[a].i]
				C := corners[ps[a].j]
				B := corners[ps[b].i]
				Dd := corners[ps[b].j]
				// Rectangle perimeter order A, B, C, D.
				if !shapefitRightAngles(A, B, C, Dd, tol) {
					// Try swapping B and D.
					B, Dd = Dd, B
					if !shapefitRightAngles(A, B, C, Dd, tol) {
						continue
					}
				}
				canon := shapefitCanonKey(idx)
				if seen[canon] {
					continue
				}
				seen[canon] = true
				center := cv.Point2f{
					X: (A.X + C.X) / 2,
					Y: (A.Y + C.Y) / 2,
				}
				out = append(out, RectangleDetection{
					Corners: [4]cv.Point2f{A, B, C, Dd},
					Center:  center,
				})
			}
		}
	}
	return out
}

// shapefitRightAngles reports whether A, B, C, D (in perimeter order) form a
// rectangle: all four interior angles are right angles within tolerance.
func shapefitRightAngles(A, B, C, D cv.Point2f, tol float64) bool {
	verts := [4]cv.Point2f{A, B, C, D}
	for i := 0; i < 4; i++ {
		prev := verts[(i+3)%4]
		cur := verts[i]
		next := verts[(i+1)%4]
		e1x := prev.X - cur.X
		e1y := prev.Y - cur.Y
		e2x := next.X - cur.X
		e2y := next.Y - cur.Y
		dot := e1x*e2x + e1y*e2y
		l1 := math.Hypot(e1x, e1y)
		l2 := math.Hypot(e2x, e2y)
		if l1 < shapefitEps || l2 < shapefitEps {
			return false
		}
		// |cos| should be near zero; allow tol/min-side as angular slack.
		if math.Abs(dot)/(l1*l2) > tol/math.Min(l1, l2)+1e-3 {
			return false
		}
	}
	return true
}

// shapefitCanonKey returns a canonical (sorted) key for a set of four corner
// indices so each rectangle is reported once.
func shapefitCanonKey(idx [4]int) [4]int {
	s := idx
	sort.Ints(s[:])
	return s
}

// SymmetryAxis is a line of reflective symmetry through a point set, given by a
// point on the axis and its orientation.
type SymmetryAxis struct {
	// Point is a point on the symmetry axis (typically the point-set centroid).
	Point cv.Point2f
	// Angle is the axis orientation in radians in (-π/2, π/2].
	Angle float64
	// Score is the symmetry quality in [0, 1]: the fraction of points whose
	// mirror image has a near neighbor in the set.
	Score float64
}

// Reflect returns the reflection of p across the symmetry axis.
func (s SymmetryAxis) Reflect(p cv.Point2f) cv.Point2f {
	dx := math.Cos(s.Angle)
	dy := math.Sin(s.Angle)
	vx := p.X - s.Point.X
	vy := p.Y - s.Point.Y
	// v reflected about the unit direction (dx, dy): 2·(v·d)·d − v.
	d := vx*dx + vy*dy
	rx := 2*d*dx - vx
	ry := 2*d*dy - vy
	return cv.Point2f{X: s.Point.X + rx, Y: s.Point.Y + ry}
}

// DetectReflectionSymmetry searches for the strongest axis of reflective
// symmetry of a point set. It considers angleSteps candidate axes through the
// centroid spanning [0, π); for each, it reflects every point and scores the
// axis by the fraction of reflections that land within tol pixels of some
// original point. The best axis and true are returned, or the zero axis and
// false when the point set is too small. tol is the matching tolerance in
// pixels.
func DetectReflectionSymmetry(pts []cv.Point2f, angleSteps int, tol float64) (SymmetryAxis, bool) {
	if len(pts) < 2 {
		return SymmetryAxis{}, false
	}
	if angleSteps < 1 {
		angleSteps = 180
	}
	center := Centroid(pts)
	best := SymmetryAxis{Point: center}
	bestScore := -1.0
	for a := 0; a < angleSteps; a++ {
		ang := math.Pi * float64(a) / float64(angleSteps)
		axis := SymmetryAxis{Point: center, Angle: shapefitWrapPi(ang)}
		matched := 0
		for _, p := range pts {
			rp := axis.Reflect(p)
			if shapefitHasNeighbor(pts, rp, tol) {
				matched++
			}
		}
		score := float64(matched) / float64(len(pts))
		if score > bestScore {
			bestScore = score
			best = axis
			best.Score = score
		}
	}
	return best, true
}

// shapefitHasNeighbor reports whether any point of pts lies within tol of q.
func shapefitHasNeighbor(pts []cv.Point2f, q cv.Point2f, tol float64) bool {
	t2 := tol * tol
	for _, p := range pts {
		dx := p.X - q.X
		dy := p.Y - q.Y
		if dx*dx+dy*dy <= t2 {
			return true
		}
	}
	return false
}

// DetectRotationalSymmetry estimates the order of rotational symmetry of a
// point set about its centroid. For each candidate fold count from 2 to maxFold
// it rotates the set by 2π/fold and scores the match by the fraction of rotated
// points that land within tol of an original point. It returns the fold with
// the highest score and that score. A fold of 1 with score 1 is returned when
// no higher-order symmetry scores above 0.5, meaning only trivial symmetry was
// found. tol is the matching tolerance in pixels.
func DetectRotationalSymmetry(pts []cv.Point2f, maxFold int, tol float64) (fold int, score float64) {
	if len(pts) < 2 || maxFold < 2 {
		return 1, 1
	}
	center := Centroid(pts)
	const minScore = 0.5
	bestFold := 1
	bestScore := 0.0
	for f := 2; f <= maxFold; f++ {
		ang := 2 * math.Pi / float64(f)
		ca := math.Cos(ang)
		sa := math.Sin(ang)
		matched := 0
		for _, p := range pts {
			vx := p.X - center.X
			vy := p.Y - center.Y
			rx := vx*ca - vy*sa
			ry := vx*sa + vy*ca
			q := cv.Point2f{X: center.X + rx, Y: center.Y + ry}
			if shapefitHasNeighbor(pts, q, tol) {
				matched++
			}
		}
		s := float64(matched) / float64(len(pts))
		// Use >= so that, among folds achieving the same (maximal) score, the
		// largest fold wins — the highest order of symmetry present.
		if s >= bestScore && s >= minScore {
			bestScore = s
			bestFold = f
		}
	}
	if bestFold == 1 {
		return 1, 1
	}
	return bestFold, bestScore
}
