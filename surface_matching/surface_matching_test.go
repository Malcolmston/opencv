package surface_matching

import (
	"math"
	"strings"
	"testing"
)

// makeSurface builds a deterministic, asymmetric graph surface
//
//	z = a·x² + b·y² + c·x·y + d·x
//
// over an off-centre rectangular domain, with analytic outward normals. The
// linear and cross terms and the asymmetric domain break every rotational and
// reflective symmetry, so a rigid transform of the cloud has a unique recovering
// pose — exactly what a PPF match should find.
func makeSurface(nx, ny int) *PointCloud {
	const (
		a = 0.40
		b = 0.15
		c = 0.20
		d = 0.10
	)
	xs := linspace(-1.0, 1.0, nx)
	ys := linspace(-0.8, 1.2, ny)
	pc := &PointCloud{}
	for _, x := range xs {
		for _, y := range ys {
			z := a*x*x + b*y*y + c*x*y + d*x
			fx := 2*a*x + c*y + d
			fy := 2*b*y + c*x
			n := normalize3(Vec3{-fx, -fy, 1})
			pc.Points = append(pc.Points, Vec3{x, y, z})
			pc.Normals = append(pc.Normals, n)
		}
	}
	return pc
}

func linspace(lo, hi float64, n int) []float64 {
	out := make([]float64, n)
	if n == 1 {
		out[0] = lo
		return out
	}
	step := (hi - lo) / float64(n-1)
	for i := 0; i < n; i++ {
		out[i] = lo + step*float64(i)
	}
	return out
}

// knownTransform is the fixed rigid pose applied to a model to synthesise a
// scene in the tests.
func knownTransform() Pose3D {
	r := rotationAxisAngle(Vec3{1, 2, 3}, 0.5)
	return Pose3D{R: r, T: Vec3{0.3, -0.2, 0.15}}
}

func TestPPFMatchRecoversKnownPose(t *testing.T) {
	model := makeSurface(13, 13)
	known := knownTransform()
	scene := model.TransformPose(known)

	det := NewPPF3DDetector(0.04, 0.04)
	det.TrainModel(model)
	if !det.Trained() {
		t.Fatal("detector reports untrained after TrainModel")
	}
	if det.hashBuckets() == 0 {
		t.Fatal("empty hash table after training")
	}

	poses := det.Match(scene, 0.2, 0.04)
	if len(poses) == 0 {
		t.Fatal("Match returned no poses")
	}
	top := poses[0]

	angErr := top.AngleTo(known)
	transErr := top.TranslationTo(known)
	t.Logf("top pose votes=%d angleErr=%.4f rad transErr=%.4f (diam=%.3f)",
		top.Votes, angErr, transErr, model.Diameter())

	if angErr > 0.30 {
		t.Errorf("rotation error too large: %.4f rad", angErr)
	}
	if transErr > 0.30 {
		t.Errorf("translation error too large: %.4f", transErr)
	}

	// Poses must be sorted by descending votes.
	for i := 1; i < len(poses); i++ {
		if poses[i-1].Votes < poses[i].Votes {
			t.Fatalf("poses not sorted by votes at %d: %d < %d", i, poses[i-1].Votes, poses[i].Votes)
		}
	}
}

func TestPPFMatchDeterministic(t *testing.T) {
	model := makeSurface(11, 11)
	scene := model.TransformPose(knownTransform())

	run := func() Pose3D {
		det := NewPPF3DDetector(0.05, 0.05)
		det.TrainModel(model)
		return det.Match(scene, 0.25, 0.05)[0]
	}
	a := run()
	b := run()
	if a.Votes != b.Votes || a.T != b.T || a.R != b.R {
		t.Fatalf("Match is not deterministic:\n%+v\n%+v", a, b)
	}
}

func TestICPRefinesPerturbedPose(t *testing.T) {
	model := makeSurface(13, 13)
	known := knownTransform()
	scene := model.TransformPose(known)

	// Perturb the ground-truth pose to build a rough initial alignment.
	perturbR := mul3(rotationAxisAngle(Vec3{0, 1, 0.3}, 0.08), known.R)
	init := Pose3D{R: perturbR, T: add3(known.T, Vec3{0.05, -0.04, 0.03})}

	icp := NewICP(60, 1e-10)
	refined, residual := icp.Register(model, scene, init)
	t.Logf("ICP residual=%.3e angleErr=%.3e transErr=%.3e",
		residual, refined.AngleTo(known), refined.TranslationTo(known))

	if residual > 1e-4 {
		t.Errorf("ICP residual too large: %.3e", residual)
	}
	if refined.Residual != residual {
		t.Errorf("pose Residual %.3e != returned residual %.3e", refined.Residual, residual)
	}
	if refined.AngleTo(known) > 1e-2 {
		t.Errorf("ICP rotation not recovered: %.3e", refined.AngleTo(known))
	}
	if refined.TranslationTo(known) > 1e-2 {
		t.Errorf("ICP translation not recovered: %.3e", refined.TranslationTo(known))
	}
}

func TestPPFThenICP(t *testing.T) {
	model := makeSurface(13, 13)
	known := knownTransform()
	scene := model.TransformPose(known)

	det := NewPPF3DDetector(0.04, 0.04)
	det.TrainModel(model)
	poses := det.Match(scene, 0.2, 0.04)
	if len(poses) == 0 {
		t.Fatal("no poses to refine")
	}

	icp := NewICP(80, 1e-11)
	refined, residual := icp.Register(model, scene, poses[0])
	t.Logf("PPF+ICP residual=%.3e", residual)
	if residual > 1e-3 {
		t.Errorf("PPF+ICP residual too large: %.3e", residual)
	}
	if refined.AngleTo(known) > 5e-2 {
		t.Errorf("combined rotation error too large: %.3e", refined.AngleTo(known))
	}
}

func TestComputeNormalsMatchesAnalytic(t *testing.T) {
	// A plane z = 0 has constant normal +z; PCA normals must recover it (up to
	// sign, fixed by the viewpoint above the plane).
	pc := &PointCloud{}
	for i := 0; i < 6; i++ {
		for j := 0; j < 6; j++ {
			pc.Points = append(pc.Points, Vec3{float64(i) * 0.2, float64(j) * 0.2, 0})
		}
	}
	pc.ComputeNormals(8, Vec3{0.5, 0.5, 10})
	for i, n := range pc.Normals {
		if math.Abs(n[2]-1) > 1e-6 || math.Abs(n[0]) > 1e-6 || math.Abs(n[1]) > 1e-6 {
			t.Fatalf("normal %d = %v, want (0,0,1)", i, n)
		}
	}
}

func TestComputeNormalsOnCurvedSurface(t *testing.T) {
	model := makeSurface(15, 15)
	analytic := make([]Vec3, len(model.Normals))
	copy(analytic, model.Normals)

	est := model.Clone()
	est.ComputeNormals(12, Vec3{0, 0, 20})
	// Compare estimated to analytic normals over interior points where the
	// neighbourhood is symmetric enough for PCA to be accurate.
	var maxAng float64
	count := 0
	for i := range est.Normals {
		ang := angleBetween(est.Normals[i], analytic[i])
		if ang > maxAng {
			maxAng = ang
		}
		if ang < 0.2 {
			count++
		}
	}
	frac := float64(count) / float64(len(est.Normals))
	t.Logf("normal agreement: %.2f within 0.2 rad, max %.3f", frac, maxAng)
	if frac < 0.7 {
		t.Errorf("too few normals agree with analytic: %.2f", frac)
	}
}

func TestVoxelDownsampleReducesAndDeterministic(t *testing.T) {
	model := makeSurface(20, 20)
	a := model.VoxelDownsample(0.25)
	b := model.VoxelDownsample(0.25)
	if a.Len() >= model.Len() {
		t.Errorf("downsample did not reduce: %d -> %d", model.Len(), a.Len())
	}
	if a.Len() != b.Len() {
		t.Fatalf("downsample non-deterministic length: %d vs %d", a.Len(), b.Len())
	}
	for i := range a.Points {
		if a.Points[i] != b.Points[i] || a.Normals[i] != b.Normals[i] {
			t.Fatalf("downsample non-deterministic at %d", i)
		}
	}
	for _, n := range a.Normals {
		if math.Abs(norm3(n)-1) > 1e-9 {
			t.Errorf("downsampled normal not unit: |n|=%.6f", norm3(n))
		}
	}
}

func TestPoseHelpers(t *testing.T) {
	p := knownTransform()
	// Inverse composed with pose is identity.
	id := p.Inverse().Compose(p)
	if rotationAngle(id.R, identity3()) > 1e-12 || norm3(id.T) > 1e-12 {
		t.Errorf("inverse∘pose not identity: R angle %.3e, |T|=%.3e",
			rotationAngle(id.R, identity3()), norm3(id.T))
	}
	// Apply then inverse-apply round-trips a point.
	x := Vec3{0.7, -0.3, 1.1}
	back := p.Inverse().Apply(p.Apply(x))
	if norm3(sub3(back, x)) > 1e-12 {
		t.Errorf("round-trip failed: got %v want %v", back, x)
	}
	// Matrix bottom row and translation column.
	m := p.Matrix()
	if m[3][0] != 0 || m[3][1] != 0 || m[3][2] != 0 || m[3][3] != 1 {
		t.Errorf("homogeneous bottom row wrong: %v", m[3])
	}
	if m[0][3] != p.T[0] || m[1][3] != p.T[1] || m[2][3] != p.T[2] {
		t.Errorf("translation column wrong")
	}
	if !strings.Contains(p.String(), "Pose3D") {
		t.Errorf("String() unexpected: %q", p.String())
	}
}

func TestParsePLY(t *testing.T) {
	data := `ply
format ascii 1.0
comment made by test
element vertex 3
property float x
property float y
property float z
property float nx
property float ny
property float nz
element face 0
property list uchar int vertex_indices
end_header
0 0 0 0 0 1
1 0 0 0 0 1
0 1 0 0 0 2
`
	pc, err := parsePLY(strings.NewReader(data))
	if err != nil {
		t.Fatalf("parsePLY error: %v", err)
	}
	if pc.Len() != 3 {
		t.Fatalf("want 3 points, got %d", pc.Len())
	}
	if len(pc.Normals) != 3 {
		t.Fatalf("want 3 normals, got %d", len(pc.Normals))
	}
	// The third normal (0,0,2) must be normalised to (0,0,1).
	if math.Abs(norm3(pc.Normals[2])-1) > 1e-9 {
		t.Errorf("normal not normalised: %v", pc.Normals[2])
	}
	if pc.Points[2] != (Vec3{0, 1, 0}) {
		t.Errorf("third point wrong: %v", pc.Points[2])
	}
}

func TestParsePLYNoNormals(t *testing.T) {
	data := `ply
format ascii 1.0
element vertex 2
property float x
property float y
property float z
end_header
1 2 3
4 5 6
`
	pc, err := parsePLY(strings.NewReader(data))
	if err != nil {
		t.Fatalf("parsePLY error: %v", err)
	}
	if pc.Len() != 2 || len(pc.Normals) != 0 {
		t.Fatalf("unexpected cloud: %d points, %d normals", pc.Len(), len(pc.Normals))
	}
}

func TestParsePLYErrors(t *testing.T) {
	if _, err := parsePLY(strings.NewReader("not a ply\n")); err == nil {
		t.Error("expected error for non-PLY input")
	}
	binHeader := "ply\nformat binary_little_endian 1.0\nelement vertex 0\nend_header\n"
	if _, err := parsePLY(strings.NewReader(binHeader)); err == nil {
		t.Error("expected error for binary PLY")
	}
	noXYZ := "ply\nformat ascii 1.0\nelement vertex 0\nproperty float a\nend_header\n"
	if _, err := parsePLY(strings.NewReader(noXYZ)); err == nil {
		t.Error("expected error for missing x/y/z")
	}
}

func TestLinalgKernels(t *testing.T) {
	// SVD reconstructs a matrix: U·diag(S)·Vᵀ == A.
	a := Mat3{{2, -1, 0}, {0.5, 3, 1}, {1, 0, 2}}
	u, s, v := svd3(a)
	var recon Mat3
	sd := Mat3{{s[0], 0, 0}, {0, s[1], 0}, {0, 0, s[2]}}
	recon = mul3(u, mul3(sd, transpose3(v)))
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			if math.Abs(recon[i][j]-a[i][j]) > 1e-9 {
				t.Fatalf("SVD reconstruction mismatch at (%d,%d): %.6f vs %.6f", i, j, recon[i][j], a[i][j])
			}
		}
	}

	// Quaternion round-trip on a rotation matrix.
	r := rotationAxisAngle(Vec3{0.3, -1, 0.7}, 1.1)
	back := quatToMat(matToQuat(r))
	if rotationAngle(r, back) > 1e-9 {
		t.Errorf("quat round-trip angle error %.3e", rotationAngle(r, back))
	}

	// alignToXAxis sends the normal to +x and is a proper rotation.
	n := normalize3(Vec3{0.2, -0.5, 0.8})
	rm := alignToXAxis(n)
	got := matVec3(rm, n)
	if norm3(sub3(got, Vec3{1, 0, 0})) > 1e-9 {
		t.Errorf("alignToXAxis did not map normal to +x: %v", got)
	}
	if math.Abs(det3(rm)-1) > 1e-9 {
		t.Errorf("alignToXAxis not a proper rotation: det=%.6f", det3(rm))
	}
}

func TestClusterPosesMerges(t *testing.T) {
	base := knownTransform()
	// Two near-identical poses and one far-away pose.
	near := base
	near.R = mul3(rotationAxisAngle(Vec3{1, 0, 0}, 0.01), base.R)
	near.T = add3(base.T, Vec3{0.001, 0, 0})
	base.Votes = 10
	near.Votes = 6
	far := Pose3D{R: identity3(), T: Vec3{5, 5, 5}, Votes: 3}

	clusters := clusterPoses([]Pose3D{base, near, far}, 0.1, 0.1)
	if len(clusters) != 2 {
		t.Fatalf("want 2 clusters, got %d", len(clusters))
	}
	if clusters[0].Votes != 16 {
		t.Errorf("merged cluster votes = %d, want 16", clusters[0].Votes)
	}
}

func TestNewPPFPanics(t *testing.T) {
	mustPanic(t, func() { NewPPF3DDetector(0, 0.05) })
	mustPanic(t, func() { NewPPF3DDetector(0.05, 2) })
	mustPanic(t, func() { NewPPF3DDetectorAngles(0.05, 0.05, 1) })
	det := NewPPF3DDetector(0.05, 0.05)
	mustPanic(t, func() { det.Match(makeSurface(3, 3), 0.5, 0.5) }) // untrained
	mustPanic(t, func() { det.TrainModel(&PointCloud{}) })          // too few points
}

func TestICPPanics(t *testing.T) {
	mustPanic(t, func() { NewICP(-1, 0.1) })
	mustPanic(t, func() { NewICP(10, -1) })
	icp := NewICP(10, 1e-9)
	mustPanic(t, func() { icp.Register(&PointCloud{}, makeSurface(3, 3), IdentityPose()) })
}

func mustPanic(t *testing.T, f func()) {
	t.Helper()
	defer func() {
		if recover() == nil {
			t.Error("expected panic, got none")
		}
	}()
	f()
}
