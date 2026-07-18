package geom_cv

import (
	cv "github.com/malcolmston/opencv"
)

// VoronoiEdge is a bounded edge of a Voronoi diagram, connecting the
// circumcenters of two adjacent Delaunay triangles.
type VoronoiEdge struct {
	// A and B are the edge endpoints (two triangle circumcenters).
	A, B cv.Point2f
}

// VoronoiCell is the region of the plane closer to Site than to any other site,
// clipped to a bounding box. Vertices lists the cell polygon in order.
type VoronoiCell struct {
	// Site is the generating point the cell belongs to.
	Site cv.Point2f
	// Vertices are the cell polygon corners in order (counter-clockwise).
	Vertices []cv.Point2f
}

// geom_cvPtKey is a comparable key for a point, used to match shared Delaunay
// edges by their endpoints.
type geom_cvPtKey struct {
	x, y float64
}

func geom_cvKeyOf(p cv.Point2f) geom_cvPtKey { return geom_cvPtKey{p.X, p.Y} }

// geom_cvEdgeKey identifies an undirected edge by its two (sorted) endpoint
// keys.
type geom_cvEdgeKey struct {
	a, b geom_cvPtKey
}

func geom_cvMakeEdgeKey(p, q cv.Point2f) geom_cvEdgeKey {
	kp, kq := geom_cvKeyOf(p), geom_cvKeyOf(q)
	if kp.x > kq.x || (kp.x == kq.x && kp.y > kq.y) {
		kp, kq = kq, kp
	}
	return geom_cvEdgeKey{kp, kq}
}

// VoronoiEdges returns the finite (bounded) edges of the Voronoi diagram of the
// input sites, obtained as the dual of the Delaunay triangulation: for every
// Delaunay edge shared by two triangles, the two triangles' circumcenters are
// joined. Unbounded edges of the diagram (those on the convex hull) are omitted
// because they extend to infinity; use [VoronoiCells] for box-clipped regions
// instead. The input slice is not modified.
func VoronoiEdges(sites []cv.Point2f) []VoronoiEdge {
	tris := DelaunayTriangulation(sites)
	if len(tris) == 0 {
		return nil
	}
	centers := make([]cv.Point2f, len(tris))
	for i, t := range tris {
		if c, ok := Circumcenter(t.A, t.B, t.C); ok {
			centers[i] = c
		}
	}
	adj := map[geom_cvEdgeKey][]int{}
	for i, t := range tris {
		adj[geom_cvMakeEdgeKey(t.A, t.B)] = append(adj[geom_cvMakeEdgeKey(t.A, t.B)], i)
		adj[geom_cvMakeEdgeKey(t.B, t.C)] = append(adj[geom_cvMakeEdgeKey(t.B, t.C)], i)
		adj[geom_cvMakeEdgeKey(t.C, t.A)] = append(adj[geom_cvMakeEdgeKey(t.C, t.A)], i)
	}
	// Deterministic iteration: walk triangles and their edges in order,
	// emitting each shared edge once.
	seen := map[geom_cvEdgeKey]bool{}
	var out []VoronoiEdge
	emit := func(p, q cv.Point2f) {
		k := geom_cvMakeEdgeKey(p, q)
		if seen[k] {
			return
		}
		ts := adj[k]
		if len(ts) == 2 {
			out = append(out, VoronoiEdge{A: centers[ts[0]], B: centers[ts[1]]})
			seen[k] = true
		}
	}
	for _, t := range tris {
		emit(t.A, t.B)
		emit(t.B, t.C)
		emit(t.C, t.A)
	}
	return out
}

// VoronoiCells returns the Voronoi region of every site, clipped to the given
// bounding box. Each cell is computed exactly by starting from the box and
// successively clipping it with the perpendicular-bisector half-plane between
// the site and every other site, so a cell is always a convex polygon in
// counter-clockwise order. Duplicate sites produce empty cells. The input slice
// is not modified.
func VoronoiCells(sites []cv.Point2f, box BoundingBox) []VoronoiCell {
	cells := make([]VoronoiCell, len(sites))
	boxPoly := box.Corners()
	for i, s := range sites {
		poly := make([]cv.Point2f, len(boxPoly))
		copy(poly, boxPoly)
		for j, o := range sites {
			if i == j || (o.X == s.X && o.Y == s.Y) {
				continue
			}
			// Keep the half-plane of points at least as close to s as to o:
			// f(p) = |p-o|^2 - |p-s|^2 >= 0, which is affine in p.
			ox, oy, sx, sy := o.X, o.Y, s.X, s.Y
			ax := 2 * (sx - ox)
			ay := 2 * (sy - oy)
			c := ox*ox + oy*oy - sx*sx - sy*sy
			poly = geom_cvClipHalfPlane(poly, func(p cv.Point2f) float64 {
				return ax*p.X + ay*p.Y + c
			})
			if len(poly) == 0 {
				break
			}
		}
		cells[i] = VoronoiCell{Site: s, Vertices: poly}
	}
	return cells
}

// NearestSite returns the index of the site closest to the query point q, or -1
// when sites is empty. Ties are broken toward the lowest index, making the
// result deterministic.
func NearestSite(sites []cv.Point2f, q cv.Point2f) int {
	best := -1
	bestD := 0.0
	for i, s := range sites {
		d := DistanceSquared(s, q)
		if best == -1 || d < bestD {
			best = i
			bestD = d
		}
	}
	return best
}
