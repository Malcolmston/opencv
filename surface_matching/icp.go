package surface_matching

import "math"

// ICP refines a rigid alignment with point-to-point iterative closest point.
// Given an initial model-to-scene [Pose3D], each iteration transforms the model
// by the current pose, pairs every transformed model point with its nearest
// scene point, solves the rigid transform that minimises the summed squared
// pairing distance (an SVD-based Kabsch step), and composes that increment onto
// the pose. Iteration stops after MaxIterations or once the root-mean-square
// pairing error stops improving by more than Tolerance.
//
// The zero value is not ready to use; construct one with [NewICP]. ICP performs
// a brute-force nearest-neighbour search (no KD-tree/FLANN acceleration), which
// suits the modest clouds typical of surface matching.
type ICP struct {
	MaxIterations int
	Tolerance     float64
	// RejectionScale rejects a correspondence whose distance exceeds this
	// multiple of the current median pairing distance before solving each step,
	// giving robustness to partial overlap and outliers. A value <= 0 disables
	// rejection.
	RejectionScale float64
}

// NewICP returns an ICP configured with the given iteration cap and convergence
// tolerance and a default outlier-rejection scale of 2.5×. It panics if
// maxIterations is negative or tolerance is negative.
func NewICP(maxIterations int, tolerance float64) *ICP {
	if maxIterations < 0 {
		panic("surface_matching: maxIterations must be non-negative")
	}
	if tolerance < 0 {
		panic("surface_matching: tolerance must be non-negative")
	}
	return &ICP{
		MaxIterations:  maxIterations,
		Tolerance:      tolerance,
		RejectionScale: 2.5,
	}
}

// Register refines the alignment of the model cloud onto the scene cloud
// starting from init and returns the refined model-to-scene pose together with
// the final root-mean-square correspondence distance (also stored in the
// returned pose's Residual). Only the point coordinates are used; normals are
// ignored by this point-to-point formulation. It panics if either cloud is
// empty.
func (icp *ICP) Register(model, scene *PointCloud, init Pose3D) (Pose3D, float64) {
	if model == nil || model.Len() == 0 || scene == nil || scene.Len() == 0 {
		panic("surface_matching: ICP requires non-empty model and scene clouds")
	}
	iters := icp.MaxIterations
	if iters == 0 {
		iters = 1
	}

	pose := init
	if pose.R == (Mat3{}) {
		pose.R = identity3()
	}

	src := model.Points
	dst := scene.Points

	working := make([]Vec3, len(src))
	for i, p := range src {
		working[i] = pose.Apply(p)
	}

	matched := make([]Vec3, len(src))
	dists := make([]float64, len(src))
	prevErr := math.Inf(1)
	var err float64

	for it := 0; it < iters; it++ {
		// Pair each working point with its nearest scene point.
		for i, p := range working {
			j, d2 := nearestPoint(p, dst)
			matched[i] = dst[j]
			dists[i] = d2
		}

		// Optional correspondence rejection based on the median distance.
		useSrc, useDst := working, matched
		if icp.RejectionScale > 0 && len(working) > 4 {
			thresh := medianSqThreshold(dists, icp.RejectionScale)
			us := make([]Vec3, 0, len(working))
			ud := make([]Vec3, 0, len(working))
			for i := range working {
				if dists[i] <= thresh {
					us = append(us, working[i])
					ud = append(ud, matched[i])
				}
			}
			if len(us) >= 3 {
				useSrc, useDst = us, ud
			} else {
				useSrc, useDst = working, matched
			}
		}

		// RMS error over all correspondences (measured before this step's update).
		var sumSq float64
		for _, d2 := range dists {
			sumSq += d2
		}
		err = math.Sqrt(sumSq / float64(len(dists)))

		// Incremental transform and composition onto the cumulative pose.
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

	// Final residual against the converged pose.
	var s float64
	for _, p := range working {
		_, d2 := nearestPoint(p, dst)
		s += d2
	}
	err = math.Sqrt(s / float64(len(working)))
	pose.Residual = err
	return pose, err
}

// nearestPoint returns the index in dst of the point closest to q and the
// squared distance to it. dst must be non-empty.
func nearestPoint(q Vec3, dst []Vec3) (int, float64) {
	best := 0
	bestD := math.Inf(1)
	for i, p := range dst {
		d := sqDist(p, q)
		if d < bestD {
			bestD = d
			best = i
		}
	}
	return best, bestD
}

// medianSqThreshold returns scale·median(dists), used to reject far-flung
// correspondences. dists holds squared distances; the median is taken on a copy
// so the caller's slice is untouched.
func medianSqThreshold(dists []float64, scale float64) float64 {
	cp := make([]float64, len(dists))
	copy(cp, dists)
	insertionSortFloat(cp)
	median := cp[len(cp)/2]
	return scale * median
}

// insertionSortFloat sorts a float slice ascending in place. It is used for the
// small correspondence-distance arrays where a full sort import is unnecessary.
func insertionSortFloat(a []float64) {
	for i := 1; i < len(a); i++ {
		v := a[i]
		j := i - 1
		for j >= 0 && a[j] > v {
			a[j+1] = a[j]
			j--
		}
		a[j+1] = v
	}
}

// rigidFromCorrespondences solves for the rotation R and translation t that
// best map the src points onto their paired dst points in the least-squares
// sense (the orthogonal Procrustes / Kabsch problem) using the local 3×3 SVD.
// A reflection, if the raw solution has one, is removed so det(R)=+1.
func rigidFromCorrespondences(src, dst []Vec3) (Mat3, Vec3) {
	n := len(src)
	var cs, cd Vec3
	for i := 0; i < n; i++ {
		cs = add3(cs, src[i])
		cd = add3(cd, dst[i])
	}
	inv := 1.0 / float64(n)
	cs = scale3(cs, inv)
	cd = scale3(cd, inv)

	var h Mat3
	for i := 0; i < n; i++ {
		a := sub3(src[i], cs)
		b := sub3(dst[i], cd)
		for r0 := 0; r0 < 3; r0++ {
			for c0 := 0; c0 < 3; c0++ {
				h[r0][c0] += a[r0] * b[c0]
			}
		}
	}
	u, _, v := svd3(h)
	r := mul3(v, transpose3(u))
	if det3(r) < 0 {
		for row := 0; row < 3; row++ {
			v[row][2] = -v[row][2]
		}
		r = mul3(v, transpose3(u))
	}
	t := sub3(cd, matVec3(r, cs))
	return r, t
}
