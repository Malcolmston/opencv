package rgbd

import "math"

// nearest returns the index in dst of the point closest to q, together with the
// squared distance. dst must be non-empty.
func nearest(q [3]float64, dst [][3]float64) (int, float64) {
	best := 0
	bestD := math.Inf(1)
	for i, p := range dst {
		dx := p[0] - q[0]
		dy := p[1] - q[1]
		dz := p[2] - q[2]
		d := dx*dx + dy*dy + dz*dz
		if d < bestD {
			bestD = d
			best = i
		}
	}
	return best, bestD
}

// rigidFromCorrespondences solves for the rotation R and translation t that
// best map the src points onto their paired dst points in the least-squares
// sense (the orthogonal Procrustes / Kabsch problem), using the local 3×3 SVD.
func rigidFromCorrespondences(src, dst [][3]float64) (r [3][3]float64, t [3]float64) {
	n := len(src)
	var cs, cd [3]float64
	for i := 0; i < n; i++ {
		cs = add3(cs, src[i])
		cd = add3(cd, dst[i])
	}
	inv := 1.0 / float64(n)
	cs = scale3(cs, inv)
	cd = scale3(cd, inv)

	// Cross-covariance H = Σ (src_i - cs)(dst_i - cd)ᵀ.
	var h [3][3]float64
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
	// R = V·Uᵀ, fixing a reflection so det(R) = +1.
	r = mul3(v, transpose3(u))
	if det3(r) < 0 {
		for row := 0; row < 3; row++ {
			v[row][2] = -v[row][2]
		}
		r = mul3(v, transpose3(u))
	}
	t = sub3(cd, matVec3(r, cs))
	return r, t
}

// ICP aligns the src point cloud to the dst point cloud with point-to-point
// iterative closest point. Each iteration pairs every src point (under the
// current estimate) with its nearest dst point, then solves the rigid transform
// that minimises the squared distance between the pairs via an SVD-based Kabsch
// step. The estimates compose across iterations.
//
// It returns the cumulative rotation R and translation t mapping the original
// src cloud toward dst (p' = R·p + t) and the final root-mean-square
// correspondence distance err. Iteration stops early once the error stops
// improving. It panics if either cloud is empty or iters is negative.
func ICP(src, dst [][3]float64, iters int) (r [3][3]float64, t [3]float64, err float64) {
	if len(src) == 0 || len(dst) == 0 {
		panic("rgbd: ICP requires non-empty point clouds")
	}
	if iters < 0 {
		panic("rgbd: ICP requires a non-negative iteration count")
	}
	if iters == 0 {
		iters = 1
	}
	// Identity initial transform.
	r = [3][3]float64{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}

	// working holds src under the current cumulative transform.
	working := make([][3]float64, len(src))
	copy(working, src)

	matched := make([][3]float64, len(src))
	prevErr := math.Inf(1)
	for it := 0; it < iters; it++ {
		var sumSq float64
		for i, p := range working {
			j, d2 := nearest(p, dst)
			matched[i] = dst[j]
			sumSq += d2
		}
		err = math.Sqrt(sumSq / float64(len(working)))
		// Solve the incremental transform from the current working set to its
		// matches and compose it into the cumulative estimate.
		dr, dt := rigidFromCorrespondences(working, matched)
		r = mul3(dr, r)
		t = add3(matVec3(dr, t), dt)
		for i, p := range src {
			working[i] = add3(matVec3(r, p), t)
		}
		if math.Abs(prevErr-err) < 1e-12 {
			break
		}
		prevErr = err
	}
	// Report the RMS correspondence error against the final transform, which is
	// one step ahead of the value measured at the top of the last iteration.
	var s float64
	for _, p := range working {
		_, d2 := nearest(p, dst)
		s += d2
	}
	err = math.Sqrt(s / float64(len(working)))
	return r, t, err
}
