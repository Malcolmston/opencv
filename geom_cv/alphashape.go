package geom_cv

import (
	cv "github.com/malcolmston/opencv"
)

// AlphaComplexTriangles returns the triangles of the Delaunay triangulation
// whose circumradius is at most alpha. This is the 2-simplex part of the alpha
// complex: as alpha grows from 0 to infinity the result grows from nothing up to
// the full Delaunay triangulation (equivalently the convex hull interior). A
// non-positive alpha yields no triangles. The input slice is not modified.
func AlphaComplexTriangles(pts []cv.Point2f, alpha float64) []Triangle {
	if alpha <= 0 {
		return nil
	}
	var out []Triangle
	for _, t := range DelaunayTriangulation(pts) {
		if circ, ok := Circumcircle(t.A, t.B, t.C); ok && circ.Radius <= alpha+geom_cvEps {
			out = append(out, t)
		}
	}
	return out
}

// AlphaShapeEdges returns the boundary edges of the alpha shape of the point set
// for the given alpha radius. It keeps the Delaunay triangles whose circumradius
// is at most alpha (see [AlphaComplexTriangles]) and reports the edges that
// belong to exactly one kept triangle, which form the outline of the shape. With
// a sufficiently large alpha the result is the convex hull boundary; with a
// small alpha the outline follows concavities in the point cloud. The edges are
// returned in a deterministic order. The input slice is not modified.
func AlphaShapeEdges(pts []cv.Point2f, alpha float64) []Segment {
	tris := AlphaComplexTriangles(pts, alpha)
	if len(tris) == 0 {
		return nil
	}
	type edgeInfo struct {
		count int
		seg   Segment
	}
	edges := map[geom_cvEdgeKey]*edgeInfo{}
	add := func(a, b cv.Point2f) {
		k := geom_cvMakeEdgeKey(a, b)
		if e, ok := edges[k]; ok {
			e.count++
		} else {
			edges[k] = &edgeInfo{count: 1, seg: Segment{A: a, B: b}}
		}
	}
	// Emit edges in triangle order for a deterministic boundary ordering.
	var order []geom_cvEdgeKey
	seen := map[geom_cvEdgeKey]bool{}
	record := func(a, b cv.Point2f) {
		k := geom_cvMakeEdgeKey(a, b)
		if !seen[k] {
			seen[k] = true
			order = append(order, k)
		}
	}
	for _, t := range tris {
		add(t.A, t.B)
		add(t.B, t.C)
		add(t.C, t.A)
		record(t.A, t.B)
		record(t.B, t.C)
		record(t.C, t.A)
	}
	var out []Segment
	for _, k := range order {
		if e := edges[k]; e.count == 1 {
			out = append(out, e.seg)
		}
	}
	return out
}
