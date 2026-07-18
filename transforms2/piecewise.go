package transforms2

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// transforms2tri is a triangle given by three vertex indices.
type transforms2tri struct{ a, b, c int }

// transforms2edge is an undirected edge given by two vertex indices.
type transforms2edge struct{ a, b int }

// transforms2circumContains reports whether point p lies inside (or on) the
// circumscribed circle of triangle (pa, pb, pc). Degenerate triangles return
// false.
func transforms2circumContains(pa, pb, pc, p cv.Point2f) bool {
	ax, ay := pa.X, pa.Y
	bx, by := pb.X, pb.Y
	cx, cy := pc.X, pc.Y
	d := 2 * (ax*(by-cy) + bx*(cy-ay) + cx*(ay-by))
	if math.Abs(d) < 1e-12 {
		return false
	}
	a2 := ax*ax + ay*ay
	b2 := bx*bx + by*by
	c2 := cx*cx + cy*cy
	ux := (a2*(by-cy) + b2*(cy-ay) + c2*(ay-by)) / d
	uy := (a2*(cx-bx) + b2*(ax-cx) + c2*(bx-ax)) / d
	r2 := (ax-ux)*(ax-ux) + (ay-uy)*(ay-uy)
	dist2 := (p.X-ux)*(p.X-ux) + (p.Y-uy)*(p.Y-uy)
	return dist2 <= r2*(1+1e-9)
}

// DelaunayTriangulation computes a Delaunay triangulation of the given points
// using the Bowyer-Watson algorithm and returns the triangles as triples of
// indices into pts. It returns nil if fewer than three points are supplied.
// Duplicate or exactly collinear point sets may yield an empty triangulation.
func DelaunayTriangulation(pts []cv.Point2f) [][3]int {
	n := len(pts)
	if n < 3 {
		return nil
	}
	// Bounding box for the super-triangle.
	minX, minY := math.Inf(1), math.Inf(1)
	maxX, maxY := math.Inf(-1), math.Inf(-1)
	for _, p := range pts {
		minX = math.Min(minX, p.X)
		minY = math.Min(minY, p.Y)
		maxX = math.Max(maxX, p.X)
		maxY = math.Max(maxY, p.Y)
	}
	dx := maxX - minX
	dy := maxY - minY
	dmax := math.Max(dx, dy)
	if dmax == 0 {
		dmax = 1
	}
	midX := (minX + maxX) / 2
	midY := (minY + maxY) / 2
	verts := make([]cv.Point2f, n+3)
	copy(verts, pts)
	verts[n] = cv.Point2f{X: midX - 20*dmax, Y: midY - dmax}
	verts[n+1] = cv.Point2f{X: midX, Y: midY + 20*dmax}
	verts[n+2] = cv.Point2f{X: midX + 20*dmax, Y: midY - dmax}

	tris := []transforms2tri{{n, n + 1, n + 2}}
	for i := 0; i < n; i++ {
		p := verts[i]
		var bad []int
		for ti, t := range tris {
			if transforms2circumContains(verts[t.a], verts[t.b], verts[t.c], p) {
				bad = append(bad, ti)
			}
		}
		// Collect boundary edges: those belonging to exactly one bad triangle.
		edgeCount := map[transforms2edge]int{}
		norm := func(a, b int) transforms2edge {
			if a > b {
				a, b = b, a
			}
			return transforms2edge{a, b}
		}
		for _, ti := range bad {
			t := tris[ti]
			edgeCount[norm(t.a, t.b)]++
			edgeCount[norm(t.b, t.c)]++
			edgeCount[norm(t.c, t.a)]++
		}
		// Remove bad triangles (iterate high to low index).
		for j := len(bad) - 1; j >= 0; j-- {
			ti := bad[j]
			tris[ti] = tris[len(tris)-1]
			tris = tris[:len(tris)-1]
		}
		// Re-triangulate the hole.
		for e, cnt := range edgeCount {
			if cnt == 1 {
				tris = append(tris, transforms2tri{e.a, e.b, i})
			}
		}
	}
	// Drop triangles that touch the super-triangle.
	var out [][3]int
	for _, t := range tris {
		if t.a >= n || t.b >= n || t.c >= n {
			continue
		}
		out = append(out, [3]int{t.a, t.b, t.c})
	}
	return out
}

// transforms2bary returns the barycentric coordinates of (px, py) with respect
// to the triangle (t0, t1, t2). The boolean reports whether the triangle is
// non-degenerate.
func transforms2bary(t0, t1, t2 cv.Point2f, px, py float64) (float64, float64, float64, bool) {
	v0x, v0y := t1.X-t0.X, t1.Y-t0.Y
	v1x, v1y := t2.X-t0.X, t2.Y-t0.Y
	v2x, v2y := px-t0.X, py-t0.Y
	den := v0x*v1y - v1x*v0y
	if math.Abs(den) < 1e-12 {
		return 0, 0, 0, false
	}
	b := (v2x*v1y - v1x*v2y) / den
	c := (v0x*v2y - v2x*v0y) / den
	a := 1 - b - c
	return a, b, c, true
}

// WarpTriangle warps the source triangle srcTri onto the destination triangle
// dstTri, compositing the result into dst in place (overwriting the covered
// pixels). src and dst must have the same number of channels. Pixels of the
// destination triangle are sampled from src through the affine map that takes
// dstTri to srcTri, using the chosen interpolation and border handling.
func WarpTriangle(src, dst *cv.Mat, srcTri, dstTri [3]cv.Point2f, interp Interpolation, border BorderMode, fill float64) {
	if src.Channels != dst.Channels {
		panic("transforms2: WarpTriangle channel mismatch")
	}
	m := GetAffineTransform([3]cv.Point2f{dstTri[0], dstTri[1], dstTri[2]}, [3]cv.Point2f{srcTri[0], srcTri[1], srcTri[2]})
	minX := int(math.Floor(math.Min(dstTri[0].X, math.Min(dstTri[1].X, dstTri[2].X))))
	maxX := int(math.Ceil(math.Max(dstTri[0].X, math.Max(dstTri[1].X, dstTri[2].X))))
	minY := int(math.Floor(math.Min(dstTri[0].Y, math.Min(dstTri[1].Y, dstTri[2].Y))))
	maxY := int(math.Ceil(math.Max(dstTri[0].Y, math.Max(dstTri[1].Y, dstTri[2].Y))))
	if minX < 0 {
		minX = 0
	}
	if minY < 0 {
		minY = 0
	}
	if maxX >= dst.Cols {
		maxX = dst.Cols - 1
	}
	if maxY >= dst.Rows {
		maxY = dst.Rows - 1
	}
	for y := minY; y <= maxY; y++ {
		for x := minX; x <= maxX; x++ {
			a, b, c, ok := transforms2bary(dstTri[0], dstTri[1], dstTri[2], float64(x), float64(y))
			if !ok || a < -1e-9 || b < -1e-9 || c < -1e-9 {
				continue
			}
			sx, sy := ApplyAffine(m, float64(x), float64(y))
			di := (y*dst.Cols + x) * dst.Channels
			for ch := 0; ch < dst.Channels; ch++ {
				dst.Data[di+ch] = transforms2clampByte(SampleChannel(src, sx, sy, ch, interp, border, fill))
			}
		}
	}
}

// PiecewiseAffineWarp warps src by moving the control points srcPts to dstPts
// under a piecewise-affine deformation. The triangulation tris (triples of
// indices into the point slices) partitions the destination; if it is nil a
// Delaunay triangulation of dstPts is computed. The output has the given width
// and height, with uncovered pixels set to fill. srcPts and dstPts must have
// the same length. It panics otherwise.
func PiecewiseAffineWarp(src *cv.Mat, srcPts, dstPts []cv.Point2f, tris [][3]int, width, height int, interp Interpolation, border BorderMode, fill float64) *cv.Mat {
	if len(srcPts) != len(dstPts) {
		panic("transforms2: PiecewiseAffineWarp point count mismatch")
	}
	if width <= 0 || height <= 0 {
		panic("transforms2: PiecewiseAffineWarp requires positive size")
	}
	if tris == nil {
		tris = DelaunayTriangulation(dstPts)
	}
	dst := cv.NewMat(height, width, src.Channels)
	if fill != 0 {
		for i := range dst.Data {
			dst.Data[i] = transforms2clampByte(fill)
		}
	}
	for _, t := range tris {
		srcTri := [3]cv.Point2f{srcPts[t[0]], srcPts[t[1]], srcPts[t[2]]}
		dstTri := [3]cv.Point2f{dstPts[t[0]], dstPts[t[1]], dstPts[t[2]]}
		WarpTriangle(src, dst, srcTri, dstTri, interp, border, fill)
	}
	return dst
}
