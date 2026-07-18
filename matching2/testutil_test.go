package matching2

import (
	"math"
	"testing"

	"github.com/malcolmston/opencv/core"
)

// approx reports whether a and b are within tol.
func approx(a, b, tol float64) bool { return math.Abs(a-b) <= tol }

// testK is a fixed pinhole intrinsic matrix used across the geometry tests.
var testK = [3][3]float64{
	{800, 0, 320},
	{0, 800, 240},
	{0, 0, 1},
}

// skew returns the 3×3 skew-symmetric cross-product matrix [v]_x.
func skew(v [3]float64) [3][3]float64 {
	return [3][3]float64{
		{0, -v[2], v[1]},
		{v[2], 0, -v[0]},
		{-v[1], v[0], 0},
	}
}

// scene builds a deterministic two-view scene: a set of 3-D world points, the
// second camera's pose (R2, t2) relative to the first (which is [I|0]), and the
// projections of the points into both images under intrinsics testK.
func scene(t *testing.T) (world []core.Point3d, R2 [3][3]float64, t2 [3]float64, img1, img2 []core.Point2d) {
	t.Helper()
	// A 20° rotation about the y axis and a sideways-and-forward baseline.
	R2 = RodriguesToMatrix([3]float64{0, 20 * math.Pi / 180, 0})
	t2 = [3]float64{1.0, 0.15, 0.25}
	// A non-coplanar cloud in front of both cameras.
	coords := [][3]float64{
		{-1.0, -0.8, 5.0}, {0.9, -0.6, 6.0}, {-0.7, 0.9, 4.5}, {0.8, 0.7, 5.5},
		{-1.2, 0.1, 7.0}, {1.1, 0.2, 6.5}, {0.0, -1.0, 5.2}, {0.2, 1.0, 4.8},
		{-0.5, -0.3, 8.0}, {0.6, 0.4, 7.5}, {-0.9, 0.5, 6.2}, {0.3, -0.7, 5.9},
		{0.1, 0.0, 4.2}, {-0.3, 0.6, 9.0}, {0.7, -0.2, 6.8}, {-0.6, -0.9, 7.3},
	}
	R1 := Mat3Identity()
	t1 := [3]float64{0, 0, 0}
	for _, c := range coords {
		world = append(world, core.Point3d{X: c[0], Y: c[1], Z: c[2]})
	}
	img1 = ProjectPoints(world, R1, t1, testK)
	img2 = ProjectPoints(world, R2, t2, testK)
	return world, R2, t2, img1, img2
}

// frob returns the Frobenius distance between two 3×3 matrices after scaling
// each to unit Frobenius norm and aligning sign, so proportional matrices
// compare as equal.
func frob(a, b [3][3]float64) float64 {
	na := normalizeFrobenius(a)
	nb := normalizeFrobenius(b)
	var dPlus, dMinus float64
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			dp := na[i][j] - nb[i][j]
			dm := na[i][j] + nb[i][j]
			dPlus += dp * dp
			dMinus += dm * dm
		}
	}
	return math.Sqrt(math.Min(dPlus, dMinus))
}
