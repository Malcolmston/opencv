package stereo2

import (
	"errors"
	"math"
	"math/rand"
)

// Point3D is a point (or vector) in 3-D space using float64 coordinates.
type Point3D struct {
	// X is the coordinate along the horizontal (image column) axis.
	X float64
	// Y is the coordinate along the vertical (image row) axis.
	Y float64
	// Z is the coordinate along the optical (depth) axis.
	Z float64
}

// Add returns the vector sum p+q.
func (p Point3D) Add(q Point3D) Point3D { return Point3D{p.X + q.X, p.Y + q.Y, p.Z + q.Z} }

// Sub returns the vector difference p-q.
func (p Point3D) Sub(q Point3D) Point3D { return Point3D{p.X - q.X, p.Y - q.Y, p.Z - q.Z} }

// Scale returns p scaled by the scalar s.
func (p Point3D) Scale(s float64) Point3D { return Point3D{p.X * s, p.Y * s, p.Z * s} }

// Dot returns the dot product p·q.
func (p Point3D) Dot(q Point3D) float64 { return p.X*q.X + p.Y*q.Y + p.Z*q.Z }

// Cross returns the cross product p×q.
func (p Point3D) Cross(q Point3D) Point3D {
	return Point3D{
		X: p.Y*q.Z - p.Z*q.Y,
		Y: p.Z*q.X - p.X*q.Z,
		Z: p.X*q.Y - p.Y*q.X,
	}
}

// Norm returns the Euclidean length of p.
func (p Point3D) Norm() float64 { return math.Sqrt(p.Dot(p)) }

// Plane is an oriented plane in 3-D space, the set of points satisfying
// A*x + B*y + C*z + D = 0. Fitting routines return a plane whose normal
// (A, B, C) is a unit vector.
type Plane struct {
	// A is the x component of the plane normal.
	A float64
	// B is the y component of the plane normal.
	B float64
	// C is the z component of the plane normal.
	C float64
	// D is the signed offset; -D is the dot product of the normal with any point on the plane.
	D float64
}

// Normal returns the plane normal (A, B, C) as a [Point3D].
func (pl Plane) Normal() Point3D { return Point3D{pl.A, pl.B, pl.C} }

// DistanceTo returns the signed perpendicular distance from point p to the
// plane. Its sign indicates which side of the plane p lies on; its magnitude is
// the true distance when the normal is unit length, as produced by [FitPlane]
// and [FitPlaneRANSAC].
func (pl Plane) DistanceTo(p Point3D) float64 {
	n := math.Sqrt(pl.A*pl.A + pl.B*pl.B + pl.C*pl.C)
	if n == 0 {
		return math.NaN()
	}
	return (pl.A*p.X + pl.B*p.Y + pl.C*p.Z + pl.D) / n
}

// FitPlane fits the total-least-squares plane to a set of 3-D points: the plane
// through their centroid whose normal is the direction of least variance (the
// eigenvector of the point covariance with the smallest eigenvalue). Unlike an
// ordinary z = f(x,y) regression it handles planes of any orientation. It
// returns an error if fewer than three points are supplied or the points are
// collinear, which leaves the normal undetermined.
func FitPlane(points []Point3D) (Plane, error) {
	if len(points) < 3 {
		return Plane{}, errors.New("stereo2: FitPlane needs at least 3 points")
	}
	var cx, cy, cz float64
	for _, p := range points {
		cx += p.X
		cy += p.Y
		cz += p.Z
	}
	n := float64(len(points))
	cx, cy, cz = cx/n, cy/n, cz/n
	// Symmetric covariance matrix.
	var xx, xy, xz, yy, yz, zz float64
	for _, p := range points {
		dx, dy, dz := p.X-cx, p.Y-cy, p.Z-cz
		xx += dx * dx
		xy += dx * dy
		xz += dx * dz
		yy += dy * dy
		yz += dy * dz
		zz += dz * dz
	}
	cov := [3][3]float64{
		{xx, xy, xz},
		{xy, yy, yz},
		{xz, yz, zz},
	}
	vals, vecs := jacobiEigen3x3(cov)
	// Smallest eigenvalue index.
	small := 0
	if vals[1] < vals[small] {
		small = 1
	}
	if vals[2] < vals[small] {
		small = 2
	}
	nx, ny, nz := vecs[0][small], vecs[1][small], vecs[2][small]
	ln := math.Sqrt(nx*nx + ny*ny + nz*nz)
	if ln < 1e-12 {
		return Plane{}, errors.New("stereo2: FitPlane could not determine a normal (degenerate points)")
	}
	nx, ny, nz = nx/ln, ny/ln, nz/ln
	d := -(nx*cx + ny*cy + nz*cz)
	return Plane{A: nx, B: ny, C: nz, D: d}, nil
}

// FitPlaneRANSAC robustly fits a plane to points that may contain outliers. It
// runs iterations rounds of random 3-point sampling, scoring each candidate
// plane by the number of inliers within threshold perpendicular distance, then
// refits [FitPlane] to the largest inlier set. The seed makes the sampling
// deterministic. It returns the fitted plane and the indices of its inliers, or
// an error if fewer than three points are supplied.
func FitPlaneRANSAC(points []Point3D, iterations int, threshold float64, seed int64) (Plane, []int, error) {
	if len(points) < 3 {
		return Plane{}, nil, errors.New("stereo2: FitPlaneRANSAC needs at least 3 points")
	}
	if iterations <= 0 {
		iterations = 100
	}
	rng := rand.New(rand.NewSource(seed))
	bestCount := -1
	var bestPlane Plane
	for it := 0; it < iterations; it++ {
		i := rng.Intn(len(points))
		j := rng.Intn(len(points))
		k := rng.Intn(len(points))
		if i == j || j == k || i == k {
			continue
		}
		a, b, c := points[i], points[j], points[k]
		normal := b.Sub(a).Cross(c.Sub(a))
		ln := normal.Norm()
		if ln < 1e-12 {
			continue
		}
		normal = normal.Scale(1 / ln)
		d := -normal.Dot(a)
		cand := Plane{A: normal.X, B: normal.Y, C: normal.Z, D: d}
		count := 0
		for _, p := range points {
			if math.Abs(cand.DistanceTo(p)) <= threshold {
				count++
			}
		}
		if count > bestCount {
			bestCount = count
			bestPlane = cand
		}
	}
	if bestCount < 0 {
		return Plane{}, nil, errors.New("stereo2: FitPlaneRANSAC found no valid sample (degenerate points)")
	}
	// Collect inliers of the best model and refine.
	var inliers []int
	inPoints := make([]Point3D, 0, bestCount)
	for idx, p := range points {
		if math.Abs(bestPlane.DistanceTo(p)) <= threshold {
			inliers = append(inliers, idx)
			inPoints = append(inPoints, p)
		}
	}
	if len(inPoints) >= 3 {
		if refined, err := FitPlane(inPoints); err == nil {
			bestPlane = refined
		}
	}
	return bestPlane, inliers, nil
}

// jacobiEigen3x3 computes the eigenvalues and orthonormal eigenvectors of a 3x3
// real symmetric matrix using the classical cyclic Jacobi method. It returns the
// eigenvalues (vals[i]) and the matching eigenvectors as columns of vecs
// (vecs[row][i]).
func jacobiEigen3x3(m [3][3]float64) (vals [3]float64, vecs [3][3]float64) {
	a := m
	// v starts as identity and accumulates the rotations.
	v := [3][3]float64{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}
	for sweep := 0; sweep < 100; sweep++ {
		off := math.Abs(a[0][1]) + math.Abs(a[0][2]) + math.Abs(a[1][2])
		if off < 1e-18 {
			break
		}
		for _, pq := range [3][2]int{{0, 1}, {0, 2}, {1, 2}} {
			p, q := pq[0], pq[1]
			if math.Abs(a[p][q]) < 1e-300 {
				continue
			}
			theta := (a[q][q] - a[p][p]) / (2 * a[p][q])
			t := math.Copysign(1, theta) / (math.Abs(theta) + math.Sqrt(theta*theta+1))
			if theta == 0 {
				t = 1
			}
			c := 1 / math.Sqrt(t*t+1)
			s := t * c
			// Apply the Jacobi rotation J^T A J.
			app := a[p][p]
			aqq := a[q][q]
			apq := a[p][q]
			a[p][p] = c*c*app - 2*s*c*apq + s*s*aqq
			a[q][q] = s*s*app + 2*s*c*apq + c*c*aqq
			a[p][q] = 0
			a[q][p] = 0
			for i := 0; i < 3; i++ {
				if i != p && i != q {
					aip := a[i][p]
					aiq := a[i][q]
					a[i][p] = c*aip - s*aiq
					a[p][i] = a[i][p]
					a[i][q] = s*aip + c*aiq
					a[q][i] = a[i][q]
				}
			}
			for i := 0; i < 3; i++ {
				vip := v[i][p]
				viq := v[i][q]
				v[i][p] = c*vip - s*viq
				v[i][q] = s*vip + c*viq
			}
		}
	}
	vals = [3]float64{a[0][0], a[1][1], a[2][2]}
	vecs = v
	return vals, vecs
}
