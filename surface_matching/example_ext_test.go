package surface_matching_test

import (
	"fmt"
	"os"
	"path/filepath"

	sm "github.com/malcolmston/opencv/surface_matching"
)

// ExampleKDTree3D builds a k-d tree over a cloud and runs the three query kinds.
func ExampleKDTree3D() {
	pts := []sm.Vec3{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 0}, {2, 2, 2}}
	tree := sm.NewKDTree3D(pts)

	idx, d2 := tree.Nearest(sm.Vec3{0.9, 0.1, 0})
	fmt.Printf("nearest index: %d (sqDist %.2f)\n", idx, d2)

	k := tree.NearestK(sm.Vec3{0.1, 0, 0}, 2)
	fmt.Printf("2-NN indices: %d %d\n", k[0].Index, k[1].Index)

	within := tree.RadiusSearch(sm.Vec3{0, 0, 0}, 1.1)
	fmt.Printf("within radius: %d points\n", len(within))
	// Output:
	// nearest index: 1 (sqDist 0.02)
	// 2-NN indices: 0 1
	// within radius: 3 points
}

// ExampleICP_RegisterPointToPlane refines a perturbed pose with point-to-plane
// ICP, which uses the scene normals and converges to a near-zero residual.
func ExampleICP_RegisterPointToPlane() {
	model := buildSurface()
	known := sm.NewPose(rotAxis(sm.Vec3{1, 2, 3}, 0.4), sm.Vec3{0.2, -0.1, 0.05})
	scene := model.TransformPose(known)

	init := sm.NewPose(rotAxis(sm.Vec3{0, 1, 0.3}, 0.47), sm.Vec3{0.24, -0.13, 0.08})
	icp := sm.NewICP(60, 1e-12)
	refined, residual := icp.RegisterPointToPlane(model, scene, init)

	fmt.Printf("rotation recovered: %v\n", refined.AngleTo(known) < 1e-3)
	fmt.Printf("tight residual: %v\n", residual < 1e-5)
	// Output:
	// rotation recovered: true
	// tight residual: true
}

// ExamplePPF3DDetector_MatchInstances detects two copies of a model placed in
// one scene.
func ExamplePPF3DDetector_MatchInstances() {
	model := buildSurface()
	diam := model.Diameter()

	a := model.TransformPose(sm.NewPose(rotAxis(sm.Vec3{1, 2, 3}, 0.4), sm.Vec3{0, 0, 0}))
	b := model.TransformPose(sm.NewPose(rotAxis(sm.Vec3{0, 1, 1}, -0.6), sm.Vec3{3 * diam, 0, 0}))
	scene := &sm.PointCloud{}
	scene.Points = append(append(scene.Points, a.Points...), b.Points...)
	scene.Normals = append(append(scene.Normals, a.Normals...), b.Normals...)

	det := sm.NewPPF3DDetector(0.04, 0.04)
	det.TrainModel(model)
	instances := det.MatchInstances(scene, 0.1, 0.04, 2)

	fmt.Printf("instances found: %v\n", len(instances) >= 2)
	fmt.Printf("separated: %v\n", instances[0].TranslationTo(instances[1]) > 0.5*diam)
	// Output:
	// instances found: true
	// separated: true
}

// ExampleScorePose verifies a hypothesised pose by its geometric inlier ratio.
func ExampleScorePose() {
	model := buildSurface()
	known := sm.NewPose(rotAxis(sm.Vec3{0, 0, 1}, 0.3), sm.Vec3{0.1, 0.2, 0})
	scene := model.TransformPose(known)

	good := sm.ScorePose(model, scene, known, 0.02*model.Diameter())
	wrong := sm.ScorePose(model, scene, sm.IdentityPose(), 0.02*model.Diameter())
	fmt.Printf("true pose is well supported: %v\n", good > 0.99)
	fmt.Printf("identity pose is not: %v\n", wrong < 0.5)
	// Output:
	// true pose is well supported: true
	// identity pose is not: true
}

// ExampleWritePLYBinary writes a cloud as binary PLY and reads it back with the
// format-detecting reader.
func ExampleWritePLYBinary() {
	model := buildSurface()
	dir, _ := os.MkdirTemp("", "smply")
	defer os.RemoveAll(dir)
	path := filepath.Join(dir, "model.ply")

	if err := sm.WritePLYBinary(path, model); err != nil {
		fmt.Println("write error:", err)
		return
	}
	pc, err := sm.ReadPLY(path)
	if err != nil {
		fmt.Println("read error:", err)
		return
	}
	fmt.Printf("round-tripped %d points with normals: %v\n", pc.Len(), len(pc.Normals) == pc.Len())
	// Output:
	// round-tripped 121 points with normals: true
}
