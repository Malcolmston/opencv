package stitching

import "math"

// BundleAdjuster globally refines the [CameraParams] produced by an [Estimator].
// The estimator recovers each camera independently by chaining pairwise
// homographies, which lets small errors accumulate around the panorama; bundle
// adjustment instead minimises a single cost over all correspondences at once,
// jointly adjusting every focal length and rotation so the images agree
// everywhere. Both implementations optimise with Levenberg–Marquardt and a
// numerically-differentiated Jacobian.
//
// The implementations differ in what they minimise: [BundleAdjusterRay]
// minimises the angle between the back-projected viewing rays of matched points,
// and [BundleAdjusterReproj] minimises their reprojection error in pixels.
type BundleAdjuster interface {
	// Refine returns cameras that better satisfy the correspondences in matches.
	// The bool result is false when there is nothing to optimise.
	Refine(cameras []CameraParams, matches []MatchesInfo) ([]CameraParams, bool)
}

// paramsPerCamera is the number of optimised parameters per camera: the focal
// length plus a 3-vector axis-angle rotation.
const paramsPerCamera = 4

// packParams flattens the cameras into the optimisation vector [focal, rx, ry,
// rz] per camera.
func packParams(cams []CameraParams) []float64 {
	x := make([]float64, len(cams)*paramsPerCamera)
	for i, c := range cams {
		rx, ry, rz := rodriguesFromMat(c.rot())
		base := i * paramsPerCamera
		x[base+0] = c.Focal
		x[base+1] = rx
		x[base+2] = ry
		x[base+3] = rz
	}
	return x
}

// unpackParams rebuilds cameras from the optimisation vector, preserving the
// aspect, principal point and translation of the templates.
func unpackParams(x []float64, templates []CameraParams) []CameraParams {
	out := make([]CameraParams, len(templates))
	copy(out, templates)
	for i := range out {
		base := i * paramsPerCamera
		out[i].Focal = x[base+0]
		out[i].R = [9]float64(rodriguesToMat(x[base+1], x[base+2], x[base+3]))
	}
	return out
}

// backProjectRay returns the unit viewing ray, in world coordinates, of pixel
// (x, y) in the camera. The ray is R⁻¹·K⁻¹·[x y 1]ᵀ normalised.
func backProjectRay(cam CameraParams, x, y float64) (float64, float64, float64) {
	kinv, ok := cam.kMat().inv3()
	if !ok {
		return 0, 0, 1
	}
	cx, cy, cz := kinv.vec(x, y, 1)
	wx, wy, wz := cam.rot().transpose().vec(cx, cy, cz)
	n := math.Sqrt(wx*wx + wy*wy + wz*wz)
	if n < 1e-12 {
		return 0, 0, 1
	}
	return wx / n, wy / n, wz / n
}

// rayResiduals returns the stacked differences between the world-space viewing
// rays of every corresponding point pair.
func rayResiduals(cams []CameraParams, matches []MatchesInfo) []float64 {
	var r []float64
	for _, mi := range matches {
		if mi.Src >= len(cams) || mi.Dst >= len(cams) {
			continue
		}
		for k := range mi.SrcPoints {
			sx, sy, sz := backProjectRay(cams[mi.Src], float64(mi.SrcPoints[k].X), float64(mi.SrcPoints[k].Y))
			dx, dy, dz := backProjectRay(cams[mi.Dst], float64(mi.DstPoints[k].X), float64(mi.DstPoints[k].Y))
			r = append(r, sx-dx, sy-dy, sz-dz)
		}
	}
	return r
}

// projectPoint maps (x, y) from the source camera into the destination camera
// via H = K_d·R_d·R_sᵀ·K_s⁻¹.
func projectPoint(src, dst CameraParams, x, y float64) (float64, float64, bool) {
	ksInv, ok := src.kMat().inv3()
	if !ok {
		return 0, 0, false
	}
	h := dst.kMat().mul(dst.rot()).mul(src.rot().transpose()).mul(ksInv)
	px, py, pw := h.vec(x, y, 1)
	if math.Abs(pw) < 1e-12 {
		return 0, 0, false
	}
	return px / pw, py / pw, true
}

// reprojResiduals returns the stacked forward and backward reprojection errors,
// in pixels, of every corresponding point pair.
func reprojResiduals(cams []CameraParams, matches []MatchesInfo) []float64 {
	var r []float64
	for _, mi := range matches {
		if mi.Src >= len(cams) || mi.Dst >= len(cams) {
			continue
		}
		for k := range mi.SrcPoints {
			sx, sy := float64(mi.SrcPoints[k].X), float64(mi.SrcPoints[k].Y)
			dx, dy := float64(mi.DstPoints[k].X), float64(mi.DstPoints[k].Y)
			if px, py, ok := projectPoint(cams[mi.Src], cams[mi.Dst], sx, sy); ok {
				r = append(r, px-dx, py-dy)
			}
			if px, py, ok := projectPoint(cams[mi.Dst], cams[mi.Src], dx, dy); ok {
				r = append(r, px-sx, py-sy)
			}
		}
	}
	return r
}

// BundleAdjusterRay refines the cameras by minimising the discrepancy between the
// back-projected viewing rays of matched points. Because rays are independent of
// image scale, it is numerically well-behaved and is the default global refiner
// in OpenCV's panorama pipeline.
type BundleAdjusterRay struct {
	// MaxIterations caps the Levenberg–Marquardt iterations. Zero selects a
	// sensible default.
	MaxIterations int
	// TermEps stops the optimisation when the relative cost improvement drops
	// below this value. Zero selects a sensible default.
	TermEps float64
}

// Refine implements [BundleAdjuster] using the ray-divergence cost.
func (ba BundleAdjusterRay) Refine(cameras []CameraParams, matches []MatchesInfo) ([]CameraParams, bool) {
	return levenbergMarquardt(cameras, matches, rayResiduals, ba.MaxIterations, ba.TermEps)
}

// BundleAdjusterReproj refines the cameras by minimising the reprojection error,
// in pixels, of matched points projected from each image into the other. It
// optimises the same quantity the panorama is ultimately judged by, at the cost
// of being slightly more sensitive to the initial focal estimate than
// [BundleAdjusterRay].
type BundleAdjusterReproj struct {
	// MaxIterations caps the Levenberg–Marquardt iterations. Zero selects a
	// sensible default.
	MaxIterations int
	// TermEps stops the optimisation when the relative cost improvement drops
	// below this value. Zero selects a sensible default.
	TermEps float64
}

// Refine implements [BundleAdjuster] using the reprojection-error cost.
func (ba BundleAdjusterReproj) Refine(cameras []CameraParams, matches []MatchesInfo) ([]CameraParams, bool) {
	return levenbergMarquardt(cameras, matches, reprojResiduals, ba.MaxIterations, ba.TermEps)
}

// levenbergMarquardt minimises the sum of squared residuals produced by resid
// over the camera parameters. It uses a forward-difference Jacobian and damped
// normal equations, accepting a step only when it lowers the cost. It returns the
// refined cameras and whether any optimisation was performed.
func levenbergMarquardt(cams []CameraParams, matches []MatchesInfo, resid func([]CameraParams, []MatchesInfo) []float64, maxIter int, eps float64) ([]CameraParams, bool) {
	if maxIter <= 0 {
		maxIter = 100
	}
	if eps <= 0 {
		eps = 1e-9
	}
	templates := make([]CameraParams, len(cams))
	copy(templates, cams)
	x := packParams(cams)
	p := len(x)

	eval := func(params []float64) []float64 {
		return resid(unpackParams(params, templates), matches)
	}
	r := eval(x)
	if len(r) == 0 {
		return cams, false
	}
	cost := dotf(r, r)
	lambda := 1e-3

	for iter := 0; iter < maxIter; iter++ {
		// Forward-difference Jacobian columns.
		cols := make([][]float64, p)
		jtr := make([]float64, p)
		for k := 0; k < p; k++ {
			h := 1e-6 * (1 + math.Abs(x[k]))
			saved := x[k]
			x[k] = saved + h
			rp := eval(x)
			x[k] = saved
			col := make([]float64, len(r))
			for m := 0; m < len(r) && m < len(rp); m++ {
				col[m] = (rp[m] - r[m]) / h
				jtr[k] += col[m] * r[m]
			}
			cols[k] = col
		}
		// Normal matrix JᵀJ.
		jtj := make([][]float64, p)
		for a := 0; a < p; a++ {
			jtj[a] = make([]float64, p)
		}
		for a := 0; a < p; a++ {
			for b := a; b < p; b++ {
				s := dotf(cols[a], cols[b])
				jtj[a][b] = s
				jtj[b][a] = s
			}
		}

		improved := false
		for try := 0; try < 8; try++ {
			aug := make([][]float64, p)
			for a := 0; a < p; a++ {
				aug[a] = append([]float64(nil), jtj[a]...)
				aug[a][a] += lambda * (jtj[a][a] + 1e-6)
			}
			neg := make([]float64, p)
			for a := 0; a < p; a++ {
				neg[a] = -jtr[a]
			}
			dx, ok := solveDense(aug, neg)
			if !ok {
				lambda *= 10
				continue
			}
			xNew := make([]float64, p)
			for a := 0; a < p; a++ {
				xNew[a] = x[a] + dx[a]
			}
			rNew := eval(xNew)
			costNew := dotf(rNew, rNew)
			if costNew < cost {
				rel := (cost - costNew) / (cost + 1e-30)
				copy(x, xNew)
				r = rNew
				cost = costNew
				lambda = math.Max(lambda*0.5, 1e-12)
				improved = true
				if rel < eps {
					return unpackParams(x, templates), true
				}
				break
			}
			lambda *= 10
			if lambda > 1e12 {
				break
			}
		}
		if !improved {
			break
		}
	}
	return unpackParams(x, templates), true
}

// dotf returns the dot product of two equal-length vectors.
func dotf(a, b []float64) float64 {
	var s float64
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	for i := 0; i < n; i++ {
		s += a[i] * b[i]
	}
	return s
}
