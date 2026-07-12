package rgbd

import cv "github.com/malcolmston/opencv"

// pointAt back-projects pixel (u, v) of depth using K, reporting whether the
// pixel carries a valid (positive) depth.
func pointAt(depth *cv.FloatMat, k [3][3]float64, u, v int) ([3]float64, bool) {
	if u < 0 || u >= depth.Cols || v < 0 || v >= depth.Rows {
		return [3]float64{}, false
	}
	z := depth.At(v, u)
	if z <= 0 {
		return [3]float64{}, false
	}
	return backProject(u, v, z, k), true
}

// Compute3DNormals estimates a unit surface normal at every pixel of a depth
// map from the local tangent plane spanned by neighbouring back-projected
// points. For pixel (row v, column u) it forms the horizontal tangent from the
// left/right neighbours and the vertical tangent from the up/down neighbours,
// takes their cross product and normalises it. Each normal is oriented to face
// the camera (its dot product with the viewing ray is negative).
//
// The result is a dense, row-major slice of length depth.Rows*depth.Cols
// matching [DepthTo3D]. Pixels whose depth is invalid, or that lack enough
// valid neighbours to form two independent tangents, receive the zero vector
// {0, 0, 0}. It panics if depth is nil/empty or K has a zero focal length.
func Compute3DNormals(depth *cv.FloatMat, k [3][3]float64) [][3]float64 {
	if depth == nil || len(depth.Data) == 0 {
		panic("rgbd: Compute3DNormals given an empty depth map")
	}
	validK(k)
	out := make([][3]float64, depth.Rows*depth.Cols)
	for v := 0; v < depth.Rows; v++ {
		for u := 0; u < depth.Cols; u++ {
			center, ok := pointAt(depth, k, u, v)
			if !ok {
				continue
			}
			// Horizontal tangent: prefer a central difference, falling back to a
			// one-sided difference against the centre at borders / missing data.
			left, okL := pointAt(depth, k, u-1, v)
			right, okR := pointAt(depth, k, u+1, v)
			var tx [3]float64
			switch {
			case okL && okR:
				tx = sub3(right, left)
			case okR:
				tx = sub3(right, center)
			case okL:
				tx = sub3(center, left)
			default:
				continue
			}
			// Vertical tangent, likewise.
			up, okU := pointAt(depth, k, u, v-1)
			down, okD := pointAt(depth, k, u, v+1)
			var ty [3]float64
			switch {
			case okU && okD:
				ty = sub3(down, up)
			case okD:
				ty = sub3(down, center)
			case okU:
				ty = sub3(center, up)
			default:
				continue
			}
			n := cross3(tx, ty)
			if norm3(n) < 1e-12 {
				continue
			}
			n = normalize3(n)
			// Orient toward the camera at the origin: the viewing ray points from
			// the camera to the surface point, so a camera-facing normal has a
			// negative dot product with it.
			if dot3(n, center) > 0 {
				n = scale3(n, -1)
			}
			out[v*depth.Cols+u] = n
		}
	}
	return out
}
