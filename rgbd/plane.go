package rgbd

import (
	"math"
	"math/rand"
)

// Plane is an oriented plane in 3-D given in Hessian normal form: a unit Normal
// n and an offset D such that a point p lies on the plane when n·p + D = 0. The
// signed distance of any point from the plane is n·p + D. Inliers records how
// many cloud points were assigned to the plane by [PlaneSegmentation].
type Plane struct {
	Normal  [3]float64
	D       float64
	Inliers int
}

// Distance returns the absolute perpendicular distance from p to the plane.
func (pl Plane) Distance(p [3]float64) float64 {
	return math.Abs(dot3(pl.Normal, p) + pl.D)
}

// PlaneOptions configures [PlaneSegmentation].
type PlaneOptions struct {
	// DistanceThreshold is the maximum point-to-plane distance for a point to
	// count as an inlier, in the same units as the point coordinates.
	DistanceThreshold float64
	// Iterations is the number of random 3-point hypotheses tried per plane.
	Iterations int
	// MaxPlanes caps how many planes are extracted.
	MaxPlanes int
	// MinInliers is the minimum inlier count for a plane to be accepted. A value
	// of zero or less selects an automatic threshold of max(3, len(points)/10).
	MinInliers int
	// Seed seeds the deterministic random sampler.
	Seed int64
}

// DefaultPlaneOptions returns reasonable defaults: a 0.02-unit inlier band, 200
// hypotheses per plane, up to 4 planes, an automatic minimum inlier count and a
// fixed seed for deterministic results.
func DefaultPlaneOptions() PlaneOptions {
	return PlaneOptions{
		DistanceThreshold: 0.02,
		Iterations:        200,
		MaxPlanes:         4,
		MinInliers:        0,
		Seed:              1,
	}
}

// fitPlane3 fits the plane through three points, returning it and whether the
// points were non-collinear enough to define one.
func fitPlane3(a, b, c [3]float64) (Plane, bool) {
	n := cross3(sub3(b, a), sub3(c, a))
	if norm3(n) < 1e-12 {
		return Plane{}, false
	}
	n = normalize3(n)
	return Plane{Normal: n, D: -dot3(n, a)}, true
}

// refitPlane recomputes a best-fit plane through a set of points by taking the
// eigenvector of their covariance matrix with the smallest eigenvalue as the
// normal. It returns the input plane unchanged if there are too few points.
func refitPlane(points [][3]float64, idx []int) Plane {
	if len(idx) < 3 {
		return Plane{}
	}
	var c [3]float64
	for _, i := range points0(points, idx) {
		c = add3(c, i)
	}
	inv := 1.0 / float64(len(idx))
	c = scale3(c, inv)
	var cov [3][3]float64
	for _, p := range points0(points, idx) {
		d := sub3(p, c)
		for r := 0; r < 3; r++ {
			for k := 0; k < 3; k++ {
				cov[r][k] += d[r] * d[k]
			}
		}
	}
	vals, vecs := jacobiEigenSym(cov)
	best := 0
	for i := 1; i < 3; i++ {
		if vals[i] < vals[best] {
			best = i
		}
	}
	n := normalize3([3]float64{vecs[0][best], vecs[1][best], vecs[2][best]})
	return Plane{Normal: n, D: -dot3(n, c)}
}

// points0 gathers the points referenced by idx into a fresh slice.
func points0(points [][3]float64, idx []int) [][3]float64 {
	out := make([][3]float64, len(idx))
	for i, id := range idx {
		out[i] = points[id]
	}
	return out
}

// PlaneSegmentation extracts one or more dominant planes from a point cloud with
// sequential RANSAC. It repeatedly finds the plane supported by the most
// still-unassigned points, refines it from its inliers, records it and removes
// those points from consideration, stopping when the best plane has fewer than
// the minimum inliers or MaxPlanes have been found.
//
// It returns the planes in the order discovered (largest first) and a label
// slice of length len(points): labels[i] is the index into planes of the plane
// point i was assigned to, or -1 if it was left unassigned. Sampling is
// deterministic for a given Seed.
func PlaneSegmentation(points [][3]float64, opts PlaneOptions) ([]Plane, []int) {
	labels := make([]int, len(points))
	for i := range labels {
		labels[i] = -1
	}
	if len(points) < 3 {
		return nil, labels
	}
	if opts.Iterations <= 0 {
		opts.Iterations = 200
	}
	if opts.MaxPlanes <= 0 {
		opts.MaxPlanes = 1
	}
	if opts.DistanceThreshold <= 0 {
		opts.DistanceThreshold = 0.02
	}
	minInliers := opts.MinInliers
	if minInliers <= 0 {
		minInliers = len(points) / 10
		if minInliers < 3 {
			minInliers = 3
		}
	}

	rng := rand.New(rand.NewSource(opts.Seed))

	// pool holds the indices of points not yet assigned to a plane.
	pool := make([]int, len(points))
	for i := range pool {
		pool[i] = i
	}

	var planes []Plane
	for len(planes) < opts.MaxPlanes && len(pool) >= 3 {
		var bestPlane Plane
		bestCount := 0
		for it := 0; it < opts.Iterations; it++ {
			a, b, c := sample3(rng, len(pool))
			pl, ok := fitPlane3(points[pool[a]], points[pool[b]], points[pool[c]])
			if !ok {
				continue
			}
			count := 0
			for _, id := range pool {
				if pl.Distance(points[id]) <= opts.DistanceThreshold {
					count++
				}
			}
			if count > bestCount {
				bestCount = count
				bestPlane = pl
			}
		}
		if bestCount < minInliers {
			break
		}
		// Collect inliers of the best hypothesis and refine the plane from them.
		var inliers []int
		for _, id := range pool {
			if bestPlane.Distance(points[id]) <= opts.DistanceThreshold {
				inliers = append(inliers, id)
			}
		}
		refined := refitPlane(points, inliers)
		// Re-select inliers against the refined plane.
		inliers = inliers[:0]
		remaining := pool[:0]
		for _, id := range pool {
			if refined.Distance(points[id]) <= opts.DistanceThreshold {
				inliers = append(inliers, id)
			} else {
				remaining = append(remaining, id)
			}
		}
		if len(inliers) < minInliers {
			break
		}
		refined.Inliers = len(inliers)
		planeIdx := len(planes)
		for _, id := range inliers {
			labels[id] = planeIdx
		}
		planes = append(planes, refined)
		pool = append([]int(nil), remaining...)
	}
	return planes, labels
}

// sample3 draws three distinct indices in [0, n) from rng. It requires n >= 3.
func sample3(rng *rand.Rand, n int) (int, int, int) {
	a := rng.Intn(n)
	b := rng.Intn(n)
	for b == a {
		b = rng.Intn(n)
	}
	c := rng.Intn(n)
	for c == a || c == b {
		c = rng.Intn(n)
	}
	return a, b, c
}
