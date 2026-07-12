package calib3d

import "math"

// DecomposeHomographyMat factorises a plane-induced image homography H into the
// rigid motions and plane normals consistent with it. K is the intrinsic matrix
// shared by the two views. It returns, in corresponding order, the candidate
// rotation matrices, unit-scaled translation vectors (t/d, the camera
// translation divided by the plane's distance d), and plane normals expressed in
// the first camera frame.
//
// The method is the classic Faugeras–Lustman analytical decomposition: the
// Euclidean homography He = K⁻¹·H·K is normalised by its middle singular value
// and its SVD yields a closed-form family of solutions. Physically implausible
// mirror solutions are removed by keeping only normals facing the camera
// (n_z ≥ 0) and near-duplicates are merged, leaving the (typically four)
// distinct candidates OpenCV also reports. Callers disambiguate the remaining
// solutions with a visibility (cheirality) test on observed points. ok is false
// when K is singular or H is degenerate.
func DecomposeHomographyMat(H [3][3]float64, K [3][3]float64) (rotations [][3][3]float64, translations [][3]float64, normals [][3]float64, ok bool) {
	kInv, okk := inv3(K)
	if !okk {
		return nil, nil, nil, false
	}
	// Euclidean homography.
	He := mul3(kInv, mul3(H, K))
	u, s, v := svd3(He)
	if s[1] < 1e-15 {
		return nil, nil, nil, false
	}
	// Normalise by the middle singular value.
	d1 := s[0] / s[1]
	d2 := 1.0
	d3 := s[2] / s[1]
	sgn := signf(det3(u)) * signf(det3(v))

	denom := d1*d1 - d3*d3
	if denom < 1e-18 {
		// Pure rotation (all singular values equal): single solution.
		R := scaleMat(mul3(u, transpose3(v)), sgn)
		return [][3][3]float64{R}, [][3]float64{{0, 0, 0}}, [][3]float64{{0, 0, 1}}, true
	}
	aux1 := math.Sqrt((d1*d1 - d2*d2) / denom)
	aux3 := math.Sqrt((d2*d2 - d3*d3) / denom)
	x1s := [4]float64{aux1, aux1, -aux1, -aux1}
	x3s := [4]float64{aux3, -aux3, aux3, -aux3}

	var rot [][3][3]float64
	var tr [][3]float64
	var nr [][3]float64

	add := func(R [3][3]float64, t, n [3]float64) {
		// Flip n to face the camera, flipping t with it to preserve t·nᵀ.
		if n[2] < 0 {
			n = scale3(n, -1)
			t = scale3(t, -1)
		}
		// Merge near-duplicates.
		for i := range rot {
			if matClose(rot[i], R, 1e-6) && vecClose(nr[i], n, 1e-6) {
				return
			}
		}
		rot = append(rot, R)
		tr = append(tr, t)
		nr = append(nr, n)
	}

	// Case d' = +d2 (proper motion).
	auxST := math.Sqrt((d1*d1-d2*d2)*(d2*d2-d3*d3)) / ((d1 + d3) * d2)
	cth := (d2*d2 + d1*d3) / ((d1 + d3) * d2)
	sths := [4]float64{auxST, -auxST, -auxST, auxST}
	for i := 0; i < 4; i++ {
		Rp := [3][3]float64{{cth, 0, -sths[i]}, {0, 1, 0}, {sths[i], 0, cth}}
		R := scaleMat(mul3(u, mul3(Rp, transpose3(v))), sgn)
		tp := [3]float64{x1s[i], 0, -x3s[i]}
		tp = scale3(tp, d1-d3)
		t := matVec3(u, tp)
		np := [3]float64{x1s[i], 0, x3s[i]}
		n := matVec3(v, np)
		add(R, t, n)
	}
	// Case d' = -d2 (reflection family).
	auxSP := math.Sqrt((d1*d1-d2*d2)*(d2*d2-d3*d3)) / ((d1 - d3) * d2)
	cph := (d1*d3 - d2*d2) / ((d1 - d3) * d2)
	sphs := [4]float64{auxSP, -auxSP, -auxSP, auxSP}
	for i := 0; i < 4; i++ {
		Rp := [3][3]float64{{cph, 0, sphs[i]}, {0, -1, 0}, {sphs[i], 0, -cph}}
		R := scaleMat(mul3(u, mul3(Rp, transpose3(v))), sgn)
		tp := [3]float64{x1s[i], 0, x3s[i]}
		tp = scale3(tp, d1+d3)
		t := matVec3(u, tp)
		np := [3]float64{x1s[i], 0, x3s[i]}
		n := matVec3(v, np)
		add(R, t, n)
	}
	return rot, tr, nr, true
}

// matClose reports whether two 3×3 matrices agree entrywise within tol.
func matClose(a, b [3][3]float64, tol float64) bool {
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			if math.Abs(a[i][j]-b[i][j]) > tol {
				return false
			}
		}
	}
	return true
}

// vecClose reports whether two 3-vectors agree componentwise within tol.
func vecClose(a, b [3]float64, tol float64) bool {
	return math.Abs(a[0]-b[0]) <= tol && math.Abs(a[1]-b[1]) <= tol && math.Abs(a[2]-b[2]) <= tol
}
