package rapid

import cv "github.com/malcolmston/opencv"

// toPoint rounds a Point2f to the nearest integer image coordinate.
func toPoint(p Point2f) cv.Point {
	return cv.Point{X: int(p.X + 0.5), Y: int(p.Y + 0.5)}
}

// DrawWireframe draws the mesh edges onto img using the projected vertex
// positions pts2d and the triangle list tris. When cullBackface is true, only
// edges belonging to front-facing triangles (in image winding) are drawn, which
// approximates hidden-line removal for convex meshes.
func DrawWireframe(img *cv.Mat, pts2d []Point2f, tris [][3]int, color cv.Scalar, cullBackface bool) {
	drawn := make(map[[2]int]bool)
	for _, tri := range tris {
		a, b, c := tri[0], tri[1], tri[2]
		if a < 0 || b < 0 || c < 0 || a >= len(pts2d) || b >= len(pts2d) || c >= len(pts2d) {
			continue
		}
		if cullBackface && signedArea(pts2d[a], pts2d[b], pts2d[c]) <= 0 {
			continue
		}
		for _, e := range [][2]int{{a, b}, {b, c}, {c, a}} {
			key := e
			if key[0] > key[1] {
				key[0], key[1] = key[1], key[0]
			}
			if drawn[key] {
				continue
			}
			drawn[key] = true
			cv.Line(img, toPoint(pts2d[e[0]]), toPoint(pts2d[e[1]]), color, 1)
		}
	}
}

// signedArea returns twice the signed area of triangle (a, b, c) in image
// coordinates; positive for counter-clockwise winding.
func signedArea(a, b, c Point2f) float64 {
	return (b.X-a.X)*(c.Y-a.Y) - (c.X-a.X)*(b.Y-a.Y)
}

// DrawSearchLines draws the search line of every control point onto img by
// connecting the first and last recorded sample location for each row of
// srcLocations (as produced by [ExtractLineBundle]).
func DrawSearchLines(img *cv.Mat, srcLocations [][]cv.Point, color cv.Scalar) {
	for _, locs := range srcLocations {
		if len(locs) < 2 {
			continue
		}
		cv.Line(img, locs[0], locs[len(locs)-1], color, 1)
	}
}

// DrawCorrespondencies marks the found edge column on each row of the bundle
// image produced by [ExtractLineBundle]. cols holds the found column per row (as
// returned by [FindCorrespondencies]). If colors is non-nil it supplies a
// per-row marker colour; otherwise a bright marker is used. The bundle is
// modified in place.
func DrawCorrespondencies(bundle *cv.Mat, cols []int, colors []cv.Scalar) {
	if bundle == nil {
		return
	}
	for i := 0; i < bundle.Rows && i < len(cols); i++ {
		c := cols[i]
		if c < 0 || c >= bundle.Cols {
			continue
		}
		var col cv.Scalar
		if colors != nil && i < len(colors) {
			col = colors[i]
		} else {
			col = cv.NewScalar(255, 255, 255, 255)
		}
		for ch := 0; ch < bundle.Channels; ch++ {
			bundle.Set(i, c, ch, clampToByte(col[ch]))
		}
	}
}
