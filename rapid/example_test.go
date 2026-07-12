package rapid_test

import (
	"fmt"

	cv "github.com/malcolmston/opencv"
	"github.com/malcolmston/opencv/rapid"
)

// cube returns a unit cube with consistent outward winding, used by the
// examples.
func cube() *rapid.Mesh {
	return &rapid.Mesh{
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
}

// ExampleProjectVertices projects the cube's vertices through a camera and
// prints how many landed inside a 480×480 frame.
func ExampleProjectVertices() {
	k := rapid.NewCamera(500, 500, 240, 240)
	pose := rapid.Pose{Rvec: [3]float64{0, 0, 0}, Tvec: [3]float64{0, 0, 6}}
	pts := rapid.ProjectVertices(cube(), pose, k)
	inside := 0
	for _, p := range pts {
		if p.X >= 0 && p.X < 480 && p.Y >= 0 && p.Y < 480 {
			inside++
		}
	}
	fmt.Printf("%d vertices, %d inside frame\n", len(pts), inside)
	// Output: 8 vertices, 8 inside frame
}

// ExampleExtractControlPoints samples control points along the projected
// silhouette of the cube.
func ExampleExtractControlPoints() {
	k := rapid.NewCamera(500, 500, 240, 240)
	pose := rapid.Pose{Rvec: [3]float64{0.15, -0.2, 0.1}, Tvec: [3]float64{0, 0, 6}}
	cps := rapid.ExtractControlPoints(64, 15, cube(), pose, k, 480, 480)
	fmt.Println(len(cps) > 0)
	// Output: true
}

// ExampleRapid_Compute tracks the cube's pose from a perturbed initial guess by
// aligning the mesh silhouette to a rendered image, driving the reprojection
// error down to a couple of pixels.
func ExampleRapid_Compute() {
	mesh := cube()
	k := rapid.NewCamera(500, 500, 240, 240)
	truePose := rapid.Pose{Rvec: [3]float64{0.15, -0.2, 0.1}, Tvec: [3]float64{0.2, -0.1, 6}}

	// Render the true silhouette as a white blob on black. Filling every
	// triangle paints the union of the faces, i.e. the solid silhouette.
	img := cv.NewMat(480, 480, 1)
	pts := rapid.ProjectVertices(mesh, truePose, k)
	for _, tri := range mesh.Tris {
		poly := []cv.Point{roundPt(pts[tri[0]]), roundPt(pts[tri[1]]), roundPt(pts[tri[2]])}
		cv.FillPoly(img, [][]cv.Point{poly}, cv.NewScalar(255))
	}

	init := rapid.Pose{Rvec: [3]float64{0.11, -0.15, 0.07}, Tvec: [3]float64{0, 0.05, 5.75}}
	tracker := rapid.NewRapid(mesh)
	final, ratio := tracker.Compute(img, 80, 20, k, init, rapid.TermCriteria{MaxCount: 40, Epsilon: 1e-6})

	// Reprojection error between recovered and true pose.
	fp := rapid.ProjectVertices(mesh, final, k)
	tp := rapid.ProjectVertices(mesh, truePose, k)
	var err float64
	for i := range fp {
		dx, dy := fp[i].X-tp[i].X, fp[i].Y-tp[i].Y
		err += dx*dx + dy*dy
	}
	fmt.Printf("converged=%v matched=%v\n", err/float64(len(fp)) < 5, ratio > 0.5)
	// Output: converged=true matched=true
}

func roundPt(p rapid.Point2f) cv.Point {
	return cv.Point{X: int(p.X + 0.5), Y: int(p.Y + 0.5)}
}
