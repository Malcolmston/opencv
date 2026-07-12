package rgbd

import "math"

// ICPPointToPlane aligns the src point cloud to the dst point cloud with
// point-to-plane iterative closest point. Unlike the point-to-point [ICP],
// which minimises the distance between paired points, this minimises the
// distance from each transformed src point to the tangent plane of its matched
// dst point (the plane through that point with the supplied normal). The
// point-to-plane metric converges faster and more accurately on locally planar
// surfaces, which is why it underlies most depth-camera odometry.
//
// dstNormals must hold one unit normal per dst point (same length as dst). Each
// iteration re-pairs every src point with its nearest dst point, linearises the
// point-to-plane error in the incremental motion twist (ω, ν) and solves the
// 6×6 Gauss–Newton system, composing the increment into the running estimate.
// It returns the [Pose] mapping src toward dst and the final root-mean-square
// point-to-plane residual. It panics if any cloud is empty, dstNormals does not
// match dst in length, or iters is negative.
func ICPPointToPlane(src, dst [][3]float64, dstNormals [][3]float64, iters int) (Pose, float64) {
	if len(src) == 0 || len(dst) == 0 {
		panic("rgbd: ICPPointToPlane requires non-empty point clouds")
	}
	if len(dstNormals) != len(dst) {
		panic("rgbd: ICPPointToPlane requires one normal per dst point")
	}
	if iters < 0 {
		panic("rgbd: ICPPointToPlane requires a non-negative iteration count")
	}
	if iters == 0 {
		iters = 1
	}
	pose := IdentityPose()
	var residual float64
	for it := 0; it < iters; it++ {
		var a [6][6]float64
		var b [6]float64
		var sumSq float64
		var used int
		for _, sp := range src {
			p := pose.Apply(sp)
			j, _ := nearest(p, dst)
			n := dstNormals[j]
			if norm3(n) < 1e-9 {
				continue
			}
			diff := sub3(p, dst[j])
			r := dot3(n, diff)
			// Jacobian of r = n·(p + ω×p + ν − q) w.r.t. (ω, ν): the rotation part
			// is (p×n) and the translation part is n.
			pn := cross3(p, n)
			jac := [6]float64{pn[0], pn[1], pn[2], n[0], n[1], n[2]}
			accumulateNormal(&a, &b, jac, r, 1)
			sumSq += r * r
			used++
		}
		if used > 0 {
			residual = math.Sqrt(sumSq / float64(used))
		}
		xi, ok := solveGaussNewton(a, b)
		if !ok {
			break
		}
		delta := Pose{R: Rodrigues([3]float64{xi[0], xi[1], xi[2]}), T: [3]float64{xi[3], xi[4], xi[5]}}
		pose = delta.Compose(pose)
		if norm3([3]float64{xi[0], xi[1], xi[2]}) < 1e-10 && norm3([3]float64{xi[3], xi[4], xi[5]}) < 1e-10 {
			break
		}
	}
	// Recompute the residual against the final pose.
	var sumSq float64
	var used int
	for _, sp := range src {
		p := pose.Apply(sp)
		j, _ := nearest(p, dst)
		n := dstNormals[j]
		if norm3(n) < 1e-9 {
			continue
		}
		r := dot3(n, sub3(p, dst[j]))
		sumSq += r * r
		used++
	}
	if used > 0 {
		residual = math.Sqrt(sumSq / float64(used))
	}
	return pose, residual
}

// nearestColored returns the index in dst of the point minimising the joint
// distance ‖p−q‖² + colorWeight·(c−dstColor)², which blends geometric proximity
// with photometric similarity. dst must be non-empty.
func nearestColored(p [3]float64, c float64, dst [][3]float64, dstColor []float64, colorWeight float64) int {
	best := 0
	bestD := math.Inf(1)
	for i, q := range dst {
		dx := q[0] - p[0]
		dy := q[1] - p[1]
		dz := q[2] - p[2]
		dc := dstColor[i] - c
		d := dx*dx + dy*dy + dz*dz + colorWeight*dc*dc
		if d < bestD {
			bestD = d
			best = i
		}
	}
	return best
}

// ColoredICP aligns two coloured point clouds, using per-point colour (a scalar
// intensity in [0,1], say) to disambiguate correspondences that geometry alone
// leaves ambiguous. Each iteration matches every src point to the dst point that
// is closest in the joint position-plus-colour metric weighted by colorWeight,
// then solves the rigid transform of the matched pairs with the SVD-based Kabsch
// step, composing it into the running estimate.
//
// srcColor and dstColor must match their clouds in length. It returns the [Pose]
// mapping src toward dst and the final root-mean-square correspondence distance
// (geometric only). It panics if any cloud is empty, the colour slices do not
// match their clouds, colorWeight is negative, or iters is negative.
func ColoredICP(src, dst [][3]float64, srcColor, dstColor []float64, colorWeight float64, iters int) (Pose, float64) {
	if len(src) == 0 || len(dst) == 0 {
		panic("rgbd: ColoredICP requires non-empty point clouds")
	}
	if len(srcColor) != len(src) || len(dstColor) != len(dst) {
		panic("rgbd: ColoredICP requires one colour per point")
	}
	if colorWeight < 0 {
		panic("rgbd: ColoredICP requires a non-negative colour weight")
	}
	if iters < 0 {
		panic("rgbd: ColoredICP requires a non-negative iteration count")
	}
	if iters == 0 {
		iters = 1
	}
	pose := IdentityPose()
	matched := make([][3]float64, len(src))
	working := make([][3]float64, len(src))
	var err float64
	prevErr := math.Inf(1)
	for it := 0; it < iters; it++ {
		var sumSq float64
		for i, sp := range src {
			p := pose.Apply(sp)
			working[i] = p
			j := nearestColored(p, srcColor[i], dst, dstColor, colorWeight)
			matched[i] = dst[j]
			d := sub3(p, dst[j])
			sumSq += dot3(d, d)
		}
		err = math.Sqrt(sumSq / float64(len(src)))
		dr, dt := rigidFromCorrespondences(working, matched)
		delta := Pose{R: dr, T: dt}
		pose = delta.Compose(pose)
		if math.Abs(prevErr-err) < 1e-12 {
			break
		}
		prevErr = err
	}
	// Final geometric RMS against the converged pose.
	var sumSq float64
	for i, sp := range src {
		p := pose.Apply(sp)
		j := nearestColored(p, srcColor[i], dst, dstColor, colorWeight)
		d := sub3(p, dst[j])
		sumSq += dot3(d, d)
	}
	err = math.Sqrt(sumSq / float64(len(src)))
	return pose, err
}
