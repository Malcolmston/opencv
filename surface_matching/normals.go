package surface_matching

// This file provides non-mutating, k-d-tree-accelerated surface-normal
// estimators that complement the in-place [PointCloud.ComputeNormals]. They
// return a fresh cloud (leaving the input untouched), share their PCA core, and
// offer both k-nearest-neighbour and fixed-radius neighbourhoods with an
// optional orientation flip toward a viewpoint.

// NormalOrientation selects how estimated normals are oriented. PCA fixes a
// normal only up to sign; these policies resolve the ambiguity.
type NormalOrientation int

const (
	// OrientAsComputed leaves each normal with the raw sign returned by PCA.
	OrientAsComputed NormalOrientation = iota
	// OrientTowardViewpoint flips each normal to point toward the given
	// viewpoint, the correct choice for a surface scanned from that vantage.
	OrientTowardViewpoint
)

// ComputeNormalsPC3d estimates a unit surface normal at every point from the
// local tangent plane of its k nearest neighbours (the smallest-eigenvalue
// eigenvector of the neighbourhood covariance) and returns a new cloud with the
// same points and the estimated normals; the input is not modified. Neighbour
// search uses a [KDTree3D], so it scales far better than the brute-force
// [PointCloud.ComputeNormals]. When orient is [OrientTowardViewpoint] each
// normal is flipped to face viewpoint.
//
// This is the package's analogue of OpenCV's computeNormalsPC3d. It panics if k
// is below 2 or the cloud has fewer than k points.
func ComputeNormalsPC3d(pc *PointCloud, k int, orient NormalOrientation, viewpoint Vec3) *PointCloud {
	if k < 2 {
		panic("surface_matching: ComputeNormalsPC3d requires k >= 2")
	}
	n := len(pc.Points)
	if n < k {
		panic("surface_matching: ComputeNormalsPC3d requires at least k points")
	}
	tree := NewKDTree3D(pc.Points)
	normals := make([]Vec3, n)
	for i, p := range pc.Points {
		nbrs := tree.NearestK(p, k)
		idx := make([]int, len(nbrs))
		for j, nb := range nbrs {
			idx[j] = nb.Index
		}
		normals[i] = planeNormal(pc.Points, idx, p, orient, viewpoint)
	}
	out := &PointCloud{Points: make([]Vec3, n), Normals: normals}
	copy(out.Points, pc.Points)
	return out
}

// ComputeNormalsRadius estimates normals like [ComputeNormalsPC3d] but from all
// neighbours within the given radius rather than a fixed count, so the
// neighbourhood adapts to local point density. Points with fewer than three
// neighbours in radius (an ill-posed plane fit) keep a zero normal. It returns a
// new cloud and panics if radius is not positive.
func ComputeNormalsRadius(pc *PointCloud, radius float64, orient NormalOrientation, viewpoint Vec3) *PointCloud {
	if radius <= 0 {
		panic("surface_matching: ComputeNormalsRadius requires radius > 0")
	}
	n := len(pc.Points)
	tree := NewKDTree3D(pc.Points)
	normals := make([]Vec3, n)
	for i, p := range pc.Points {
		idx := tree.RadiusSearch(p, radius)
		if len(idx) < 3 {
			continue
		}
		normals[i] = planeNormal(pc.Points, idx, p, orient, viewpoint)
	}
	out := &PointCloud{Points: make([]Vec3, n), Normals: normals}
	copy(out.Points, pc.Points)
	return out
}

// planeNormal fits a plane to the neighbourhood points[idx] by PCA and returns
// its unit normal (the smallest-eigenvalue eigenvector of the covariance),
// oriented toward viewpoint when requested. p is the point the normal belongs
// to, used for the orientation test.
func planeNormal(points []Vec3, idx []int, p Vec3, orient NormalOrientation, viewpoint Vec3) Vec3 {
	var mean Vec3
	for _, j := range idx {
		mean = add3(mean, points[j])
	}
	mean = scale3(mean, 1/float64(len(idx)))
	var cov Mat3
	for _, j := range idx {
		d := sub3(points[j], mean)
		for r := 0; r < 3; r++ {
			for c := 0; c < 3; c++ {
				cov[r][c] += d[r] * d[c]
			}
		}
	}
	vals, vecs := jacobiEigenSym(cov)
	mi := 0
	for t := 1; t < 3; t++ {
		if vals[t] < vals[mi] {
			mi = t
		}
	}
	normal := normalize3(Vec3{vecs[0][mi], vecs[1][mi], vecs[2][mi]})
	if orient == OrientTowardViewpoint && dot3(normal, sub3(viewpoint, p)) < 0 {
		normal = scale3(normal, -1)
	}
	return normal
}
