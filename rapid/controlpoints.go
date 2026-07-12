package rapid

import (
	"math"
	"sort"
)

// contourEdge is a silhouette edge with its projected endpoints (a2d, b2d) and
// its originating 3D model endpoints (a3d, b3d).
type contourEdge struct {
	a2d, b2d Point2f
	a3d, b3d [3]float64
}

// edgeKey normalises an undirected edge (i, j) so the two orderings hash equal.
func edgeKey(i, j int) [2]int {
	if i < j {
		return [2]int{i, j}
	}
	return [2]int{j, i}
}

// faceVisible reports whether a triangle is front-facing (its outward normal
// points toward the camera at the origin). v0, v1, v2 are camera-space
// vertices.
func faceVisible(v0, v1, v2 [3]float64) bool {
	e1 := [3]float64{v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]}
	e2 := [3]float64{v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]}
	n := [3]float64{
		e1[1]*e2[2] - e1[2]*e2[1],
		e1[2]*e2[0] - e1[0]*e2[2],
		e1[0]*e2[1] - e1[1]*e2[0],
	}
	cen := [3]float64{(v0[0] + v1[0] + v2[0]) / 3, (v0[1] + v1[1] + v2[1]) / 3, (v0[2] + v1[2] + v2[2]) / 3}
	return n[0]*cen[0]+n[1]*cen[1]+n[2]*cen[2] < 0
}

// silhouetteEdges returns the contour edges of the mesh at the given pose: edges
// adjacent to exactly one visible (front-facing) triangle. Vertices behind the
// camera are ignored.
func silhouetteEdges(mesh *Mesh, r [3][3]float64, t [3]float64, kk [4]float64) []contourEdge {
	camVerts := make([][3]float64, len(mesh.Vertices))
	imgVerts := make([]Point2f, len(mesh.Vertices))
	inFront := make([]bool, len(mesh.Vertices))
	for i, v := range mesh.Vertices {
		p, xc, ok := project(v, r, t, kk)
		camVerts[i] = xc
		imgVerts[i] = p
		inFront[i] = ok
	}

	// visibleCount[edge] counts adjacent front-facing triangles.
	visibleCount := make(map[[2]int]int)
	edgeVerts := make(map[[2]int][2]int)
	for _, tri := range mesh.Tris {
		a, b, c := tri[0], tri[1], tri[2]
		vis := faceVisible(camVerts[a], camVerts[b], camVerts[c])
		for _, e := range [][2]int{{a, b}, {b, c}, {c, a}} {
			k := edgeKey(e[0], e[1])
			edgeVerts[k] = k
			if vis {
				visibleCount[k]++
			}
		}
	}

	var edges []contourEdge
	for k, cnt := range visibleCount {
		if cnt != 1 {
			continue
		}
		i, j := k[0], k[1]
		if !inFront[i] || !inFront[j] {
			continue
		}
		edges = append(edges, contourEdge{
			a2d: imgVerts[i], b2d: imgVerts[j],
			a3d: mesh.Vertices[i], b3d: mesh.Vertices[j],
		})
	}
	// Deterministic order for reproducible sampling.
	sort.Slice(edges, func(a, b int) bool {
		if edges[a].a2d.X != edges[b].a2d.X {
			return edges[a].a2d.X < edges[b].a2d.X
		}
		return edges[a].a2d.Y < edges[b].a2d.Y
	})
	return edges
}

// ExtractControlPoints projects the mesh at the given pose, extracts its
// silhouette (contour) edges, and samples up to num control points spread along
// them in proportion to each edge's projected length. length is the search
// half-window that will later be used by [ExtractLineBundle]; it is accepted for
// signature compatibility with OpenCV and reserved for clamping points away from
// the image border. width and height are the target image dimensions.
//
// Each returned [ControlPoint] carries its projected 2D location, the 3D model
// point it came from, and the unit contour normal along which the edge search is
// performed.
func ExtractControlPoints(num, length int, mesh *Mesh, pose Pose, k [3][3]float64, width, height int) []ControlPoint {
	if num <= 0 {
		return nil
	}
	r := rodrigues(pose.Rvec)
	kk := intrinsics(k)
	edges := silhouetteEdges(mesh, r, pose.Tvec, kk)
	if len(edges) == 0 {
		return nil
	}

	lengths := make([]float64, len(edges))
	var total float64
	for i, e := range edges {
		lengths[i] = e.b2d.sub(e.a2d).norm()
		total += lengths[i]
	}
	if total < 1e-9 {
		return nil
	}

	var cps []ControlPoint
	for i, e := range edges {
		n := int(math.Round(float64(num) * lengths[i] / total))
		if n < 1 {
			n = 1
		}
		dir := e.b2d.sub(e.a2d)
		dl := dir.norm()
		if dl < 1e-9 {
			continue
		}
		normal := Point2f{X: -dir.Y / dl, Y: dir.X / dl}
		for s := 0; s < n; s++ {
			// Sample at interior parameters to avoid clustering at shared vertices.
			tt := (float64(s) + 0.5) / float64(n)
			img := Point2f{X: e.a2d.X + tt*dir.X, Y: e.a2d.Y + tt*dir.Y}
			// Keep a length-sized search margin inside the image border.
			margin := float64(length)
			if img.X < margin || img.Y < margin || img.X > float64(width-1)-margin || img.Y > float64(height-1)-margin {
				continue
			}
			obj := [3]float64{
				e.a3d[0] + tt*(e.b3d[0]-e.a3d[0]),
				e.a3d[1] + tt*(e.b3d[1]-e.a3d[1]),
				e.a3d[2] + tt*(e.b3d[2]-e.a3d[2]),
			}
			cps = append(cps, ControlPoint{Image: img, Object: obj, Normal: normal})
		}
	}
	return cps
}
