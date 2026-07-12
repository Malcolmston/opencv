package rgbd

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// NormalsMethod selects the estimator used by [RgbdNormals].
type NormalsMethod int

const (
	// NormalsFALS is the Fast Approximate Least Squares method: it fits, in a
	// least-squares sense, the plane through the back-projected points of a
	// window and takes its normal. It is accurate on smooth surfaces.
	NormalsFALS NormalsMethod = iota
	// NormalsLINEMOD estimates the normal from the cross product of the local
	// depth gradients (the tangent vectors along the image axes). It is the
	// cheapest method and matches the LINE-MOD detector's normals.
	NormalsLINEMOD
	// NormalsSRI is the Spherical Range Image method: it differentiates depth in
	// the spherical (azimuth, elevation) parametrisation of the image, which
	// handles strong perspective and grazing angles more gracefully than the
	// Cartesian methods.
	NormalsSRI
)

// RgbdNormals estimates per-pixel surface normals from a depth map using one of
// three methods (FALS, LINE-MOD or SRI), following OpenCV's rgbd::RgbdNormals.
// It is configured once with the image size, intrinsics, window half-size and
// method, then applied to any number of same-sized depth maps.
//
// The zero value is not usable; construct one with [NewRgbdNormals].
type RgbdNormals struct {
	Rows   int
	Cols   int
	K      [3][3]float64
	Window int
	Method NormalsMethod
}

// NewRgbdNormals returns an [RgbdNormals] for rows×cols depth maps with the
// given intrinsics, neighbourhood half-size and method. It panics if the size or
// window is not positive or K has a zero focal length.
func NewRgbdNormals(rows, cols int, k [3][3]float64, window int, method NormalsMethod) *RgbdNormals {
	if rows <= 0 || cols <= 0 {
		panic("rgbd: NewRgbdNormals requires a positive image size")
	}
	if window <= 0 {
		panic("rgbd: NewRgbdNormals requires a positive window")
	}
	validK(k)
	return &RgbdNormals{Rows: rows, Cols: cols, K: k, Window: window, Method: method}
}

// Compute returns a dense, row-major slice of unit normals, one per pixel, in
// the same layout as [DepthTo3D]. Pixels without a usable estimate hold the zero
// vector {0,0,0}. Every normal is oriented to face the camera (its dot product
// with the viewing ray is negative). It panics if depth is nil or its size does
// not match the configured Rows×Cols.
func (rn *RgbdNormals) Compute(depth *cv.FloatMat) [][3]float64 {
	if depth == nil || depth.Rows != rn.Rows || depth.Cols != rn.Cols {
		panic("rgbd: RgbdNormals.Compute given a depth map of the wrong size")
	}
	switch rn.Method {
	case NormalsLINEMOD:
		return rn.computeLinemod(depth)
	case NormalsSRI:
		return rn.computeSRI(depth)
	default:
		return rn.computeFALS(depth)
	}
}

// orient flips n so it points toward the camera at the origin (dot with the
// view ray p is negative) and returns the unit vector, or the zero vector if n
// is degenerate.
func orient(n, p [3]float64) [3]float64 {
	if norm3(n) < 1e-12 {
		return [3]float64{}
	}
	n = normalize3(n)
	if dot3(n, p) > 0 {
		n = scale3(n, -1)
	}
	return n
}

// computeFALS fits the plane n·X = 1 to the back-projected points of the window
// by solving the 3×3 normal equations (Σ XXᵀ)·n = Σ X; the plane normal is n
// normalised. A near-singular scatter matrix (too few points, or a window that
// straddles the origin) yields the zero vector.
func (rn *RgbdNormals) computeFALS(depth *cv.FloatMat) [][3]float64 {
	win := rn.Window
	out := make([][3]float64, rn.Rows*rn.Cols)
	for v := 0; v < rn.Rows; v++ {
		for u := 0; u < rn.Cols; u++ {
			center, ok := pointAt(depth, rn.K, u, v)
			if !ok {
				continue
			}
			var m [3][3]float64
			var rhs [3]float64
			count := 0
			for dv := -win; dv <= win; dv++ {
				for du := -win; du <= win; du++ {
					p, okp := pointAt(depth, rn.K, u+du, v+dv)
					if !okp {
						continue
					}
					for i := 0; i < 3; i++ {
						for j := 0; j < 3; j++ {
							m[i][j] += p[i] * p[j]
						}
						rhs[i] += p[i]
					}
					count++
				}
			}
			if count < 3 {
				continue
			}
			n, ok := solve3(m, rhs)
			if !ok {
				continue
			}
			out[v*rn.Cols+u] = orient(n, center)
		}
	}
	return out
}

// computeLinemod estimates the normal as the cross product of the local depth
// gradients: the horizontal and vertical tangent vectors formed from
// neighbouring back-projected points. It is the same construction as the
// package's [Compute3DNormals] but keyed to the configured window step.
func (rn *RgbdNormals) computeLinemod(depth *cv.FloatMat) [][3]float64 {
	out := make([][3]float64, rn.Rows*rn.Cols)
	step := rn.Window
	for v := 0; v < rn.Rows; v++ {
		for u := 0; u < rn.Cols; u++ {
			center, ok := pointAt(depth, rn.K, u, v)
			if !ok {
				continue
			}
			left, okL := pointAt(depth, rn.K, u-step, v)
			right, okR := pointAt(depth, rn.K, u+step, v)
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
			up, okU := pointAt(depth, rn.K, u, v-step)
			down, okD := pointAt(depth, rn.K, u, v+step)
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
			out[v*rn.Cols+u] = orient(cross3(tx, ty), center)
		}
	}
	return out
}

// computeSRI estimates normals in the spherical range-image parametrisation.
// Each pixel is written as a range r along its ray direction d(u,v); the surface
// normal is d minus its projections onto the two ray-derivative directions
// weighted by the log-range gradients, which is the analytic normal of the range
// surface r(θ, φ). Gradients use central differences of ln r over the window
// step.
func (rn *RgbdNormals) computeSRI(depth *cv.FloatMat) [][3]float64 {
	out := make([][3]float64, rn.Rows*rn.Cols)
	step := rn.Window
	// ray returns the unit viewing direction through pixel (u,v) and its range r
	// (Euclidean distance to the measured point), or ok=false if invalid.
	ray := func(u, v int) (dir [3]float64, r float64, ok bool) {
		p, okp := pointAt(depth, rn.K, u, v)
		if !okp {
			return dir, 0, false
		}
		r = norm3(p)
		if r < 1e-9 {
			return dir, 0, false
		}
		return scale3(p, 1/r), r, true
	}
	inv2s := 1.0 / float64(2*step)
	for v := 0; v < rn.Rows; v++ {
		for u := 0; u < rn.Cols; u++ {
			d, r, ok := ray(u, v)
			if !ok {
				continue
			}
			dl, rl, okL := ray(u-step, v)
			dr, rr, okR := ray(u+step, v)
			du, ru, okU := ray(u, v-step)
			dd, rd, okD := ray(u, v+step)
			if !(okL && okR && okU && okD) {
				continue
			}
			// A surface point in spherical form is P = r·d(θ,φ). Its tangent along
			// each image axis is ∂P = (∂r)·d + r·(∂d): the range derivative scales
			// the ray while the ray derivative sweeps the view sphere. Central
			// differences give both.
			drU := (rr - rl) * inv2s
			drV := (rd - ru) * inv2s
			ddU := scale3(sub3(dr, dl), inv2s)
			ddV := scale3(sub3(dd, du), inv2s)
			pu := add3(scale3(d, drU), scale3(ddU, r))
			pv := add3(scale3(d, drV), scale3(ddV, r))
			out[v*rn.Cols+u] = orient(cross3(pu, pv), d)
		}
	}
	return out
}

// solve3 solves the symmetric 3×3 system m·x = b by Cramer's rule, reporting
// ok=false when m is (near) singular.
func solve3(m [3][3]float64, b [3]float64) (x [3]float64, ok bool) {
	d := det3(m)
	if math.Abs(d) < 1e-15 {
		return x, false
	}
	mx := m
	mx[0][0], mx[1][0], mx[2][0] = b[0], b[1], b[2]
	my := m
	my[0][1], my[1][1], my[2][1] = b[0], b[1], b[2]
	mz := m
	mz[0][2], mz[1][2], mz[2][2] = b[0], b[1], b[2]
	return [3]float64{det3(mx) / d, det3(my) / d, det3(mz) / d}, true
}
