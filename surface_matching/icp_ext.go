package surface_matching

import "math"

// RegisterKD refines the alignment of model onto scene exactly as
// [ICP.Register] does, but accelerates the per-iteration nearest-neighbour
// search with a [KDTree3D] built once over the scene. The result is identical to
// the brute-force path; only the running time differs (O(log n) instead of
// O(n) per correspondence). It panics on empty clouds.
func (icp *ICP) RegisterKD(model, scene *PointCloud, init Pose3D) (Pose3D, float64) {
	if model == nil || model.Len() == 0 || scene == nil || scene.Len() == 0 {
		panic("surface_matching: ICP requires non-empty model and scene clouds")
	}
	tree := NewKDTree3D(scene.Points)
	iters := icp.MaxIterations
	if iters == 0 {
		iters = 1
	}

	pose := init
	if pose.R == (Mat3{}) {
		pose.R = identity3()
	}
	src := model.Points
	working := make([]Vec3, len(src))
	for i, p := range src {
		working[i] = pose.Apply(p)
	}
	matched := make([]Vec3, len(src))
	dists := make([]float64, len(src))
	prevErr := math.Inf(1)
	var err float64

	for it := 0; it < iters; it++ {
		for i, p := range working {
			j, d2 := tree.Nearest(p)
			matched[i] = scene.Points[j]
			dists[i] = d2
		}
		useSrc, useDst := rejectByMedian(working, matched, dists, icp.RejectionScale)

		var sumSq float64
		for _, d2 := range dists {
			sumSq += d2
		}
		err = math.Sqrt(sumSq / float64(len(dists)))

		dr, dt := rigidFromCorrespondences(useSrc, useDst)
		inc := Pose3D{R: dr, T: dt}
		pose = inc.Compose(pose)
		for i, p := range src {
			working[i] = pose.Apply(p)
		}
		if math.Abs(prevErr-err) < icp.Tolerance {
			break
		}
		prevErr = err
	}

	var s float64
	for _, p := range working {
		_, d2 := tree.Nearest(p)
		s += d2
	}
	err = math.Sqrt(s / float64(len(working)))
	pose.Residual = err
	return pose, err
}

// RegisterPointToPlane refines the alignment of model onto scene by minimising
// the point-to-plane error: the squared distance from each transformed model
// point to the tangent plane of its nearest scene point, using the scene's
// surface normals. Each iteration solves the 6×6 linearised (small-angle)
// least-squares system for an incremental rotation and translation and composes
// it onto the pose.
//
// Point-to-plane ICP slides freely along flat surfaces and typically converges
// in far fewer iterations than the point-to-point variant, which is why it is
// the standard refinement for range data. The scene must carry per-point
// normals; it panics otherwise or on empty clouds. The returned residual is the
// root-mean-square point-to-point correspondence distance, matching
// [ICP.Register].
func (icp *ICP) RegisterPointToPlane(model, scene *PointCloud, init Pose3D) (Pose3D, float64) {
	if model == nil || model.Len() == 0 || scene == nil || scene.Len() == 0 {
		panic("surface_matching: ICP requires non-empty model and scene clouds")
	}
	if len(scene.Normals) != len(scene.Points) {
		panic("surface_matching: RegisterPointToPlane requires per-point scene normals")
	}
	tree := NewKDTree3D(scene.Points)
	iters := icp.MaxIterations
	if iters == 0 {
		iters = 1
	}

	pose := init
	if pose.R == (Mat3{}) {
		pose.R = identity3()
	}
	src := model.Points
	working := make([]Vec3, len(src))
	for i, p := range src {
		working[i] = pose.Apply(p)
	}
	dists := make([]float64, len(src))
	nearIdx := make([]int, len(src))
	prevErr := math.Inf(1)
	var err float64

	for it := 0; it < iters; it++ {
		for i, p := range working {
			j, d2 := tree.Nearest(p)
			nearIdx[i] = j
			dists[i] = d2
		}
		thresh := math.Inf(1)
		if icp.RejectionScale > 0 && len(working) > 4 {
			thresh = medianSqThreshold(dists, icp.RejectionScale)
		}

		// Accumulate the 6×6 normal equations A·[ω;t] = b of the linearised
		// point-to-plane objective. For a source point s paired with scene point
		// d having unit normal n, the row is J = [s×n ; n] and the target is
		// (d − s)·n.
		var ata [6][6]float64
		var atb [6]float64
		var sumSq float64
		count := 0
		for i, s := range working {
			d2 := dists[i]
			sumSq += d2
			if d2 > thresh {
				continue
			}
			dp := scene.Points[nearIdx[i]]
			n := scene.Normals[nearIdx[i]]
			c := cross3(s, n)
			j := [6]float64{c[0], c[1], c[2], n[0], n[1], n[2]}
			e := dot3(sub3(dp, s), n)
			for a := 0; a < 6; a++ {
				for b := 0; b < 6; b++ {
					ata[a][b] += j[a] * j[b]
				}
				atb[a] += j[a] * e
			}
			count++
		}
		err = math.Sqrt(sumSq / float64(len(dists)))

		if count >= 6 {
			if x, ok := solve6x6(ata, atb); ok {
				omega := Vec3{x[0], x[1], x[2]}
				trans := Vec3{x[3], x[4], x[5]}
				dr := rotationFromOmega(omega)
				inc := Pose3D{R: dr, T: trans}
				pose = inc.Compose(pose)
				for i, p := range src {
					working[i] = pose.Apply(p)
				}
			}
		}
		if math.Abs(prevErr-err) < icp.Tolerance {
			break
		}
		prevErr = err
	}

	var s float64
	for _, p := range working {
		_, d2 := tree.Nearest(p)
		s += d2
	}
	err = math.Sqrt(s / float64(len(working)))
	pose.Residual = err
	return pose, err
}

// RegisterMultiScale runs a coarse-to-fine (pyramid) point-to-point ICP. It
// voxel-down-samples both clouds at a sequence of levels shrinking geometrically
// from coarse to fine, aligning at each level with [ICP.RegisterKD] and passing
// the result down as the initial pose for the next, finer level. The coarsest
// grid gives a wide, cheap basin of convergence that removes gross
// misalignment; the finest recovers detail.
//
// levels must be at least 1; level 1 reduces to a single full-resolution
// registration. The coarsest voxel size is 0.1× the model diameter, halving
// each level down to full resolution. When the scene carries per-point normals
// each level uses the wider-basin point-to-plane step ([ICP.RegisterPointToPlane]);
// otherwise it falls back to point-to-point ([ICP.RegisterKD]). It panics on
// empty clouds or levels < 1.
func (icp *ICP) RegisterMultiScale(model, scene *PointCloud, init Pose3D, levels int) (Pose3D, float64) {
	if model == nil || model.Len() == 0 || scene == nil || scene.Len() == 0 {
		panic("surface_matching: ICP requires non-empty model and scene clouds")
	}
	if levels < 1 {
		panic("surface_matching: RegisterMultiScale requires levels >= 1")
	}
	diam := model.Diameter()
	pose := init
	if pose.R == (Mat3{}) {
		pose.R = identity3()
	}
	var residual float64
	for lvl := levels - 1; lvl >= 0; lvl-- {
		var m, s *PointCloud
		if lvl == 0 || diam <= 0 {
			m, s = model, scene
		} else {
			voxel := 0.1 * diam * math.Pow(0.5, float64(levels-1-lvl))
			m = model.VoxelDownsample(voxel)
			s = scene.VoxelDownsample(voxel)
			if m.Len() < 3 || s.Len() < 3 {
				m, s = model, scene
			}
		}
		if len(s.Normals) == len(s.Points) && s.Len() > 0 {
			pose, residual = icp.RegisterPointToPlane(m, s, pose)
		} else {
			pose, residual = icp.RegisterKD(m, s, pose)
		}
	}
	pose.Residual = residual
	return pose, residual
}

// rejectByMedian returns the subset of paired source/destination points whose
// squared correspondence distance is within scale× the median, giving ICP
// robustness to partial overlap. A non-positive scale, or too few survivors,
// leaves the correspondences untouched.
func rejectByMedian(working, matched []Vec3, dists []float64, scale float64) ([]Vec3, []Vec3) {
	if scale <= 0 || len(working) <= 4 {
		return working, matched
	}
	thresh := medianSqThreshold(dists, scale)
	us := make([]Vec3, 0, len(working))
	ud := make([]Vec3, 0, len(working))
	for i := range working {
		if dists[i] <= thresh {
			us = append(us, working[i])
			ud = append(ud, matched[i])
		}
	}
	if len(us) >= 3 {
		return us, ud
	}
	return working, matched
}

// rotationFromOmega returns the rotation matrix whose axis-angle (exponential
// map) representation is the vector omega: its direction is the rotation axis
// and its magnitude the rotation angle in radians. A zero vector yields the
// identity.
func rotationFromOmega(omega Vec3) Mat3 {
	angle := norm3(omega)
	if angle < 1e-15 {
		return identity3()
	}
	return rotationAxisAngle(omega, angle)
}

// solve6x6 solves the linear system a·x = b for a fixed 6×6 matrix by Gaussian
// elimination with partial pivoting. It returns the solution and true, or the
// zero vector and false if the matrix is singular to working precision.
func solve6x6(a [6][6]float64, b [6]float64) ([6]float64, bool) {
	const n = 6
	var m [6][7]float64
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			m[i][j] = a[i][j]
		}
		m[i][n] = b[i]
	}
	for col := 0; col < n; col++ {
		piv := col
		best := math.Abs(m[col][col])
		for r := col + 1; r < n; r++ {
			if v := math.Abs(m[r][col]); v > best {
				best = v
				piv = r
			}
		}
		if best < 1e-14 {
			return [6]float64{}, false
		}
		m[col], m[piv] = m[piv], m[col]
		inv := 1 / m[col][col]
		for r := 0; r < n; r++ {
			if r == col {
				continue
			}
			f := m[r][col] * inv
			if f == 0 {
				continue
			}
			for c := col; c <= n; c++ {
				m[r][c] -= f * m[col][c]
			}
		}
	}
	var x [6]float64
	for i := 0; i < n; i++ {
		x[i] = m[i][n] / m[i][i]
	}
	return x, true
}
