package geom_cv

import (
	cv "github.com/malcolmston/opencv"
)

// geom_cvClipHalfPlane clips the polygon poly to the half-plane where the affine
// function f is non-negative, using one pass of the Sutherland–Hodgman rule. f
// must be affine in its argument so that the boundary crossing is found by
// linear interpolation. The returned slice is a new allocation.
func geom_cvClipHalfPlane(poly []cv.Point2f, f func(cv.Point2f) float64) []cv.Point2f {
	n := len(poly)
	if n == 0 {
		return nil
	}
	var out []cv.Point2f
	for i := 0; i < n; i++ {
		cur := poly[i]
		nxt := poly[(i+1)%n]
		fc := f(cur)
		fn := f(nxt)
		curIn := fc >= -geom_cvEps
		nxtIn := fn >= -geom_cvEps
		if curIn {
			out = append(out, cur)
		}
		if curIn != nxtIn {
			t := fc / (fc - fn)
			out = append(out, Lerp(cur, nxt, t))
		}
	}
	return out
}

// ClipPolygon clips the subject polygon against the convex clip polygon with the
// Sutherland–Hodgman algorithm and returns the intersection polygon in the same
// winding as subject. The clip polygon must be convex; its winding may be either
// orientation. An empty result (no overlap) is returned as an empty slice. The
// inputs are not modified.
func ClipPolygon(subject, clip []cv.Point2f) []cv.Point2f {
	if len(subject) < 3 || len(clip) < 3 {
		return nil
	}
	// Normalize the clip polygon to counter-clockwise so "inside" is the left
	// side of every directed edge.
	c := EnsureCounterClockwise(clip)
	out := make([]cv.Point2f, len(subject))
	copy(out, subject)
	m := len(c)
	for i := 0; i < m && len(out) > 0; i++ {
		a := c[i]
		b := c[(i+1)%m]
		out = geom_cvClipHalfPlane(out, func(p cv.Point2f) float64 {
			return Cross(Sub(b, a), Sub(p, a))
		})
	}
	if out == nil {
		return []cv.Point2f{}
	}
	return out
}

// PolygonIntersectionArea returns the area of the overlap between the subject
// polygon and the convex clip polygon. The clip polygon must be convex; the
// subject may be any simple polygon that lies within the clip's convex decision
// (it is clipped by each clip edge in turn). Non-overlapping polygons yield 0.
func PolygonIntersectionArea(subject, clip []cv.Point2f) float64 {
	return PolygonArea(ClipPolygon(subject, clip))
}

// ClipSegmentToBox clips the segment ab to the axis-aligned box using the
// Liang–Barsky algorithm. It returns the clipped segment and true when any part
// of the segment lies inside the box, or the zero segment and false when the
// segment is entirely outside.
func ClipSegmentToBox(a, b cv.Point2f, box BoundingBox) (Segment, bool) {
	d := Sub(b, a)
	t0, t1 := 0.0, 1.0
	// Each clip returns false when the segment is fully rejected.
	clip := func(p, q float64) bool {
		if p == 0 {
			return q >= 0 // Parallel: inside only if q >= 0.
		}
		r := q / p
		if p < 0 {
			if r > t1 {
				return false
			}
			if r > t0 {
				t0 = r
			}
		} else {
			if r < t0 {
				return false
			}
			if r < t1 {
				t1 = r
			}
		}
		return true
	}
	if clip(-d.X, a.X-box.Min.X) && clip(d.X, box.Max.X-a.X) &&
		clip(-d.Y, a.Y-box.Min.Y) && clip(d.Y, box.Max.Y-a.Y) {
		if t1 < t0 {
			return Segment{}, false
		}
		return Segment{A: Add(a, Scale(d, t0)), B: Add(a, Scale(d, t1))}, true
	}
	return Segment{}, false
}
