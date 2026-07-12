package surface_matching_test

import (
	"fmt"
	"math"
	"os"
	"path/filepath"

	sm "github.com/malcolmston/opencv/surface_matching"
)

// buildSurface returns a small deterministic oriented cloud shaped like a graph
// surface, used by the examples as a stand-in model object.
func buildSurface() *sm.PointCloud {
	pc := &sm.PointCloud{}
	for i := 0; i < 11; i++ {
		for j := 0; j < 11; j++ {
			x := -1 + 0.2*float64(i)
			y := -0.8 + 0.2*float64(j)
			z := 0.4*x*x + 0.15*y*y + 0.2*x*y + 0.1*x
			pc.Points = append(pc.Points, sm.Vec3{x, y, z})
			pc.Normals = append(pc.Normals, sm.Vec3{-(0.8*x + 0.2*y + 0.1), -(0.3*y + 0.2*x), 1})
		}
	}
	// Normalise the analytic normals.
	pc.ComputeNormals(10, sm.Vec3{0, 0, 20})
	return pc
}

// Example demonstrates the full train, match and refine pipeline: a model is
// hidden in a scene by a known rigid transform, PPF matching recovers it, and
// ICP polishes the alignment to a near-zero residual.
func Example() {
	model := buildSurface()

	// Synthesise a scene by applying a known rotation and translation.
	rot := sm.NewPose(rotAxis(sm.Vec3{1, 2, 3}, 0.5), sm.Vec3{0.3, -0.2, 0.15})
	scene := model.TransformPose(rot)

	det := sm.NewPPF3DDetector(0.05, 0.05)
	det.TrainModel(model)

	poses := det.Match(scene, 0.2, 0.05)
	top := poses[0]

	icp := sm.NewICP(50, 1e-10)
	refined, residual := icp.Register(model, scene, top)

	fmt.Printf("candidates: %v\n", len(poses) > 0)
	fmt.Printf("rotation recovered: %v\n", refined.AngleTo(rot) < 1e-3)
	fmt.Printf("translation recovered: %v\n", refined.TranslationTo(rot) < 1e-3)
	fmt.Printf("tight residual: %v\n", residual < 1e-6)
	// Output:
	// candidates: true
	// rotation recovered: true
	// translation recovered: true
	// tight residual: true
}

// ExamplePPF3DDetector_Match shows reading out the strongest candidate pose and
// applying it to map a model point into the scene frame.
func ExamplePPF3DDetector_Match() {
	model := buildSurface()
	scene := model.TransformPose(sm.NewPose(rotAxis(sm.Vec3{0, 0, 1}, 0.3), sm.Vec3{0.1, 0.2, 0}))

	det := sm.NewPPF3DDetector(0.05, 0.05)
	det.TrainModel(model)
	poses := det.Match(scene, 0.2, 0.05)

	best := poses[0]
	mapped := best.Apply(model.Points[0])
	fmt.Printf("have pose with votes>0: %v\n", best.Votes > 0)
	fmt.Printf("mapped point is finite: %v\n", isFinite(mapped))
	// Output:
	// have pose with votes>0: true
	// mapped point is finite: true
}

// ExampleLoadPLY writes a tiny ASCII PLY file and reads it back.
func ExampleLoadPLY() {
	dir, _ := os.MkdirTemp("", "ply")
	defer os.RemoveAll(dir)
	path := filepath.Join(dir, "cube.ply")
	content := "ply\nformat ascii 1.0\nelement vertex 2\n" +
		"property float x\nproperty float y\nproperty float z\n" +
		"property float nx\nproperty float ny\nproperty float nz\n" +
		"end_header\n0 0 0 0 0 1\n1 1 1 1 0 0\n"
	os.WriteFile(path, []byte(content), 0o644)

	pc, err := sm.LoadPLY(path)
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	fmt.Printf("points: %d, normals: %d\n", pc.Len(), len(pc.Normals))
	// Output:
	// points: 2, normals: 2
}

func rotAxis(axis sm.Vec3, angle float64) sm.Mat3 {
	// Rodrigues' rotation, duplicated here so the example depends only on the
	// package's exported surface.
	n := math.Sqrt(axis[0]*axis[0] + axis[1]*axis[1] + axis[2]*axis[2])
	x, y, z := axis[0]/n, axis[1]/n, axis[2]/n
	c := math.Cos(angle)
	s := math.Sin(angle)
	t := 1 - c
	return sm.Mat3{
		{c + x*x*t, x*y*t - z*s, x*z*t + y*s},
		{y*x*t + z*s, c + y*y*t, y*z*t - x*s},
		{z*x*t - y*s, z*y*t + x*s, c + z*z*t},
	}
}

func isFinite(v sm.Vec3) bool {
	for _, c := range v {
		if c != c || c > 1e308 || c < -1e308 {
			return false
		}
	}
	return true
}
