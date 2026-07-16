package rapid_test

import (
	"fmt"

	cv "github.com/malcolmston/opencv"
	"github.com/malcolmston/opencv/rapid"
)

// Example tracks a known cube from a perturbed initial pose. It renders the
// cube's true silhouette, runs a single RAPID iteration with Proceed, and shows
// that one Gauss-Newton step already pulls the pose closer to the truth.
func Example() {
	mesh := &rapid.Mesh{
		Vertices: [][3]float64{
			{-0.5, -0.5, -0.5}, {0.5, -0.5, -0.5}, {0.5, 0.5, -0.5}, {-0.5, 0.5, -0.5},
			{-0.5, -0.5, 0.5}, {0.5, -0.5, 0.5}, {0.5, 0.5, 0.5}, {-0.5, 0.5, 0.5},
		},
		Tris: [][3]int{
			{4, 5, 6}, {4, 6, 7}, {1, 0, 3}, {1, 3, 2},
			{1, 2, 6}, {1, 6, 5}, {0, 4, 7}, {0, 7, 3},
			{3, 7, 6}, {3, 6, 2}, {0, 1, 5}, {0, 5, 4},
		},
	}
	k := rapid.NewCamera(500, 500, 240, 240)
	truePose := rapid.Pose{Rvec: [3]float64{0.15, -0.2, 0.1}, Tvec: [3]float64{0.2, -0.1, 6}}

	// Render the true silhouette as a filled white blob on black.
	img := cv.NewMat(480, 480, 1)
	tp := rapid.ProjectVertices(mesh, truePose, k)
	for _, tri := range mesh.Tris {
		poly := []cv.Point{
			{X: int(tp[tri[0]].X + 0.5), Y: int(tp[tri[0]].Y + 0.5)},
			{X: int(tp[tri[1]].X + 0.5), Y: int(tp[tri[1]].Y + 0.5)},
			{X: int(tp[tri[2]].X + 0.5), Y: int(tp[tri[2]].Y + 0.5)},
		}
		cv.FillPoly(img, [][]cv.Point{poly}, cv.NewScalar(255))
	}

	// Start from a perturbed guess and take one RAPID step.
	init := rapid.Pose{Rvec: [3]float64{0.11, -0.15, 0.07}, Tvec: [3]float64{0, 0.05, 5.75}}
	before := poseError(rapid.ProjectVertices(mesh, init, k), tp)
	updated, ratio, _ := rapid.NewRapid(mesh).Proceed(img, 80, 20, k, init)
	after := poseError(rapid.ProjectVertices(mesh, updated, k), tp)

	fmt.Printf("matched=%v improved=%v\n", ratio > 0.5, after < before)
	// Output: matched=true improved=true
}

// poseError returns the mean squared reprojection error between two vertex sets.
func poseError(a, b []rapid.Point2f) float64 {
	var e float64
	for i := range a {
		dx, dy := a[i].X-b[i].X, a[i].Y-b[i].Y
		e += dx*dx + dy*dy
	}
	return e / float64(len(a))
}
