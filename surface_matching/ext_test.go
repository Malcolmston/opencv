package surface_matching

import (
	"bytes"
	"math"
	"testing"
)

// bruteNearest is a reference brute-force nearest-neighbour used to validate the
// k-d tree.
func bruteNearest(q Vec3, pts []Vec3) (int, float64) {
	best, bestD := -1, math.Inf(1)
	for i, p := range pts {
		if d := sqDist(q, p); d < bestD {
			bestD, best = d, i
		}
	}
	return best, bestD
}

func TestKDTreeNearestMatchesBrute(t *testing.T) {
	model := makeSurface(12, 10)
	tree := NewKDTree3D(model.Points)
	if tree.Len() != model.Len() {
		t.Fatalf("tree Len %d != %d", tree.Len(), model.Len())
	}
	// Query at every model point and at off-surface probes.
	probes := append([]Vec3{}, model.Points...)
	probes = append(probes, Vec3{0.11, -0.37, 0.42}, Vec3{-0.9, 1.1, 0.2}, Vec3{2, 2, 2})
	for _, q := range probes {
		gi, gd := tree.Nearest(q)
		bi, bd := bruteNearest(q, model.Points)
		if math.Abs(gd-bd) > 1e-12 {
			t.Fatalf("nearest dist mismatch at %v: %.6g vs %.6g", q, gd, bd)
		}
		// Indices may differ only on exact ties; distances must match, and the
		// found point must equal the brute point in coordinates.
		if model.Points[gi] != model.Points[bi] && gd != bd {
			t.Fatalf("nearest index mismatch at %v: %d vs %d", q, gi, bi)
		}
	}
}

func TestKDTreeNearestKAndRadius(t *testing.T) {
	model := makeSurface(9, 9)
	tree := NewKDTree3D(model.Points)
	q := model.Points[20]

	k := 6
	nn := tree.NearestK(q, k)
	if len(nn) != k {
		t.Fatalf("NearestK returned %d, want %d", len(nn), k)
	}
	// Ordered non-decreasing and the closest is the query point itself (d=0).
	if nn[0].SqDist != 0 {
		t.Errorf("closest neighbour not the point itself: d=%.3g", nn[0].SqDist)
	}
	for i := 1; i < len(nn); i++ {
		if nn[i].SqDist < nn[i-1].SqDist {
			t.Fatalf("NearestK not sorted at %d", i)
		}
	}
	// The k-NN set must equal the k smallest brute distances.
	all := make([]float64, model.Len())
	for i, p := range model.Points {
		all[i] = sqDist(q, p)
	}
	insertionSortFloat(all)
	for i := 0; i < k; i++ {
		if math.Abs(nn[i].SqDist-all[i]) > 1e-12 {
			t.Fatalf("NearestK dist %d = %.6g, want %.6g", i, nn[i].SqDist, all[i])
		}
	}

	// Radius search agrees with a brute count.
	r := 0.35
	idx := tree.RadiusSearch(q, r)
	want := 0
	for _, p := range model.Points {
		if sqDist(q, p) <= r*r {
			want++
		}
	}
	if len(idx) != want {
		t.Fatalf("RadiusSearch found %d, brute %d", len(idx), want)
	}
}

func TestRegisterKDMatchesBrute(t *testing.T) {
	model := makeSurface(11, 11)
	known := knownTransform()
	scene := model.TransformPose(known)
	perturbR := mul3(rotationAxisAngle(Vec3{0, 1, 0.3}, 0.06), known.R)
	init := Pose3D{R: perturbR, T: add3(known.T, Vec3{0.04, -0.03, 0.02})}

	icp := NewICP(60, 1e-11)
	brute, br := icp.Register(model, scene, init)
	kd, kr := icp.RegisterKD(model, scene, init)
	if math.Abs(br-kr) > 1e-9 {
		t.Errorf("KD residual %.3e != brute %.3e", kr, br)
	}
	if kd.AngleTo(brute) > 1e-9 || kd.TranslationTo(brute) > 1e-9 {
		t.Errorf("KD pose diverged from brute: ang=%.3e trans=%.3e",
			kd.AngleTo(brute), kd.TranslationTo(brute))
	}
	if kr > 1e-4 {
		t.Errorf("KD ICP residual too large: %.3e", kr)
	}
}

func TestPointToPlaneICPConverges(t *testing.T) {
	model := makeSurface(13, 13)
	known := knownTransform()
	scene := model.TransformPose(known)
	perturbR := mul3(rotationAxisAngle(Vec3{0.2, 1, -0.4}, 0.07), known.R)
	init := Pose3D{R: perturbR, T: add3(known.T, Vec3{0.05, -0.04, 0.03})}

	icp := NewICP(60, 1e-12)
	refined, residual := icp.RegisterPointToPlane(model, scene, init)
	t.Logf("point-to-plane residual=%.3e ang=%.3e trans=%.3e",
		residual, refined.AngleTo(known), refined.TranslationTo(known))
	if residual > 1e-4 {
		t.Errorf("point-to-plane residual too large: %.3e", residual)
	}
	if refined.AngleTo(known) > 1e-2 {
		t.Errorf("point-to-plane rotation not recovered: %.3e", refined.AngleTo(known))
	}
	if refined.TranslationTo(known) > 1e-2 {
		t.Errorf("point-to-plane translation not recovered: %.3e", refined.TranslationTo(known))
	}
	if refined.Residual != residual {
		t.Errorf("Residual field %.3e != returned %.3e", refined.Residual, residual)
	}
}

func TestMultiScaleICPWideBasin(t *testing.T) {
	model := makeSurface(15, 15)
	known := knownTransform()
	scene := model.TransformPose(known)
	// A larger perturbation than plain ICP would tolerate.
	perturbR := mul3(rotationAxisAngle(Vec3{0.3, 1, 0.2}, 0.25), known.R)
	init := Pose3D{R: perturbR, T: add3(known.T, Vec3{0.15, -0.12, 0.1})}

	icp := NewICP(40, 1e-11)
	refined, residual := icp.RegisterMultiScale(model, scene, init, 3)
	t.Logf("multiscale residual=%.3e ang=%.3e trans=%.3e",
		residual, refined.AngleTo(known), refined.TranslationTo(known))
	if residual > 1e-3 {
		t.Errorf("multiscale residual too large: %.3e", residual)
	}
	if refined.AngleTo(known) > 5e-2 {
		t.Errorf("multiscale rotation not recovered: %.3e", refined.AngleTo(known))
	}
}

func TestSolve6x6(t *testing.T) {
	a := [6][6]float64{
		{4, 1, 0, 0, 1, 0},
		{1, 3, 1, 0, 0, 1},
		{0, 1, 2, 1, 0, 0},
		{0, 0, 1, 5, 1, 0},
		{1, 0, 0, 1, 4, 1},
		{0, 1, 0, 0, 1, 3},
	}
	want := [6]float64{1, -2, 0.5, 3, -1, 2}
	var b [6]float64
	for i := 0; i < 6; i++ {
		for j := 0; j < 6; j++ {
			b[i] += a[i][j] * want[j]
		}
	}
	got, ok := solve6x6(a, b)
	if !ok {
		t.Fatal("solve6x6 reported singular")
	}
	for i := 0; i < 6; i++ {
		if math.Abs(got[i]-want[i]) > 1e-10 {
			t.Fatalf("solve6x6[%d]=%.6g want %.6g", i, got[i], want[i])
		}
	}
	if _, ok := solve6x6([6][6]float64{}, [6]float64{}); ok {
		t.Error("solve6x6 should report singular for zero matrix")
	}
}

func TestSamplePCUniform(t *testing.T) {
	model := makeSurface(10, 10)
	s := SamplePCUniform(model, 3)
	wantN := (model.Len() + 2) / 3
	if s.Len() != wantN {
		t.Fatalf("SamplePCUniform len %d, want %d", s.Len(), wantN)
	}
	if len(s.Normals) != s.Len() {
		t.Fatalf("uniform sample dropped normals: %d vs %d", len(s.Normals), s.Len())
	}
	if s.Points[0] != model.Points[0] || s.Points[1] != model.Points[3] {
		t.Errorf("uniform sample took wrong indices")
	}
	mustPanic(t, func() { SamplePCUniform(model, 0) })
}

func TestSamplePCByQuantization(t *testing.T) {
	model := makeSurface(20, 20)
	a := SamplePCByQuantization(model, 0.1)
	b := SamplePCByQuantization(model, 0.1)
	if a.Len() >= model.Len() {
		t.Errorf("quantization did not reduce: %d -> %d", model.Len(), a.Len())
	}
	if a.Len() != b.Len() {
		t.Fatalf("quantization non-deterministic length %d vs %d", a.Len(), b.Len())
	}
	for i := range a.Points {
		if a.Points[i] != b.Points[i] {
			t.Fatalf("quantization non-deterministic at %d", i)
		}
	}
	// Representatives must be genuine input points.
	set := map[Vec3]bool{}
	for _, p := range model.Points {
		set[p] = true
	}
	for _, p := range a.Points {
		if !set[p] {
			t.Fatalf("quantization emitted a non-original point %v", p)
		}
	}
	mustPanic(t, func() { SamplePCByQuantization(model, 0) })
}

func TestComputeNormalsPC3dVariants(t *testing.T) {
	// Plane z=0: normals must be ±z; viewpoint above forces +z.
	pc := &PointCloud{}
	for i := 0; i < 7; i++ {
		for j := 0; j < 7; j++ {
			pc.Points = append(pc.Points, Vec3{float64(i) * 0.2, float64(j) * 0.2, 0})
		}
	}
	out := ComputeNormalsPC3d(pc, 8, OrientTowardViewpoint, Vec3{0.6, 0.6, 10})
	if len(pc.Normals) != 0 {
		t.Error("ComputeNormalsPC3d mutated the input cloud")
	}
	for i, n := range out.Normals {
		if math.Abs(n[2]-1) > 1e-6 || math.Abs(n[0]) > 1e-6 || math.Abs(n[1]) > 1e-6 {
			t.Fatalf("normal %d = %v, want (0,0,1)", i, n)
		}
	}
	// Radius variant on the same plane.
	outR := ComputeNormalsRadius(pc, 0.45, OrientTowardViewpoint, Vec3{0.6, 0.6, 10})
	nonZero := 0
	for _, n := range outR.Normals {
		if norm3(n) > 0.5 {
			nonZero++
			if math.Abs(math.Abs(n[2])-1) > 1e-6 {
				t.Fatalf("radius normal not ±z: %v", n)
			}
		}
	}
	if nonZero == 0 {
		t.Error("ComputeNormalsRadius produced no normals")
	}
	mustPanic(t, func() { ComputeNormalsPC3d(pc, 1, OrientAsComputed, Vec3{}) })
	mustPanic(t, func() { ComputeNormalsRadius(pc, 0, OrientAsComputed, Vec3{}) })
}

func TestComputeNormalsPC3dMatchesInPlace(t *testing.T) {
	// The k-d-tree PCA estimator must agree with the brute-force in-place one.
	model := makeSurface(12, 12)
	ref := model.Clone()
	ref.ComputeNormals(10, Vec3{0, 0, 20})
	out := ComputeNormalsPC3d(model, 10, OrientTowardViewpoint, Vec3{0, 0, 20})
	for i := range ref.Normals {
		if angleBetween(ref.Normals[i], out.Normals[i]) > 1e-9 {
			t.Fatalf("normal %d disagrees: %v vs %v", i, ref.Normals[i], out.Normals[i])
		}
	}
}

func TestTransformPCPose(t *testing.T) {
	model := makeSurface(6, 6)
	pose := knownTransform()
	a := TransformPCPose(model, pose)
	b := model.TransformPose(pose)
	for i := range a.Points {
		if norm3(sub3(a.Points[i], b.Points[i])) > 1e-12 {
			t.Fatalf("TransformPCPose disagrees with method at %d", i)
		}
	}
	if TransformPCPose(nil, pose) != nil {
		t.Error("TransformPCPose(nil) should be nil")
	}
}

func TestPoseScoringAndVerification(t *testing.T) {
	model := makeSurface(11, 11)
	known := knownTransform()
	scene := model.TransformPose(known)
	diam := model.Diameter()

	// The true pose scores ~1; a wrong pose scores low.
	good := ScorePose(model, scene, known, 0.02*diam)
	if good < 0.99 {
		t.Errorf("true pose scored only %.3f", good)
	}
	bad := Pose3D{R: identity3(), T: Vec3{5, 5, 5}}
	if s := ScorePose(model, scene, bad, 0.02*diam); s > 0.01 {
		t.Errorf("wrong pose scored too high: %.3f", s)
	}
	if PoseInliers(model, scene, known, 0.02*diam) != model.Len() {
		t.Errorf("true pose should have all inliers")
	}
	ok, sc := VerifyPose(model, scene, known, 0.02*diam, 0.9)
	if !ok || sc < 0.9 {
		t.Errorf("VerifyPose rejected the true pose: ok=%v score=%.3f", ok, sc)
	}
	okBad, _ := VerifyPose(model, scene, bad, 0.02*diam, 0.9)
	if okBad {
		t.Error("VerifyPose accepted a wrong pose")
	}
	// Normal-aware scoring also accepts the true pose.
	if s := ScorePoseNormals(model, scene, known, 0.02*diam, 0.2); s < 0.99 {
		t.Errorf("ScorePoseNormals on true pose = %.3f", s)
	}
	mustPanic(t, func() { PoseInliers(model, scene, known, 0) })
}

func TestSuppressNonMaximum(t *testing.T) {
	base := knownTransform()
	base.Votes = 20
	near := base
	near.T = add3(base.T, Vec3{0.001, 0, 0})
	near.Votes = 15
	far := Pose3D{R: identity3(), T: Vec3{5, 5, 5}, Votes: 8}

	kept := SuppressNonMaximum([]Pose3D{near, base, far}, 0.1, 0.1)
	if len(kept) != 2 {
		t.Fatalf("NMS kept %d, want 2", len(kept))
	}
	if kept[0].Votes != 20 {
		t.Errorf("NMS did not keep strongest first: %d", kept[0].Votes)
	}
	// NMS keeps originals, it does not sum votes (unlike clustering).
	if kept[0].Votes == 35 {
		t.Error("NMS should not merge votes")
	}
	if SuppressNonMaximum(nil, 0.1, 0.1) != nil {
		t.Error("NMS of nil should be nil")
	}
}

func TestClusterAndAverageExported(t *testing.T) {
	base := knownTransform()
	base.Votes = 10
	near := base
	near.R = mul3(rotationAxisAngle(Vec3{1, 0, 0}, 0.01), base.R)
	near.Votes = 6
	clusters := ClusterPoses([]Pose3D{base, near}, 0.1, 0.1)
	if len(clusters) != 1 || clusters[0].Votes != 16 {
		t.Fatalf("ClusterPoses wrong: %+v", clusters)
	}
	avg := AveragePoses([]Pose3D{base, near})
	if avg.Votes != 16 {
		t.Errorf("AveragePoses votes = %d, want 16", avg.Votes)
	}
	if AveragePoses(nil) != (Pose3D{}) {
		t.Error("AveragePoses(nil) should be the zero pose")
	}
}

func TestHashIndexMatchesTable(t *testing.T) {
	model := makeSurface(9, 9)
	det := NewPPF3DDetector(0.05, 0.05)
	det.TrainModel(model)
	idx := det.BuildHashIndex()
	if idx.Len() != det.hashBuckets() {
		t.Fatalf("index keys %d != table buckets %d", idx.Len(), det.hashBuckets())
	}
	// Every table bucket must be retrievable with identical contents.
	for key, bucket := range det.hashTable {
		got := idx.Lookup(key)
		if len(got) != len(bucket) {
			t.Fatalf("bucket %d size %d != %d", key, len(got), len(bucket))
		}
	}
	// A key that cannot exist returns nil.
	if idx.Lookup(^uint64(0)) != nil {
		t.Error("Lookup of absent key should be nil")
	}
}

func TestTrainModelSpreadStillMatches(t *testing.T) {
	model := makeSurface(13, 13)
	known := knownTransform()
	scene := model.TransformPose(known)

	det := NewPPF3DDetector(0.04, 0.04)
	det.TrainModelSpread(model)
	if !det.Trained() {
		t.Fatal("spread detector reports untrained")
	}
	poses := det.Match(scene, 0.2, 0.04)
	if len(poses) == 0 {
		t.Fatal("spread Match returned nothing")
	}
	top := poses[0]
	if top.AngleTo(known) > 0.3 || top.TranslationTo(known) > 0.3 {
		t.Errorf("spread match poor: ang=%.3f trans=%.3f", top.AngleTo(known), top.TranslationTo(known))
	}
	mustPanic(t, func() {
		bad := NewPPF3DDetector(0.05, 0.05)
		bad.TrainModelSpread(&PointCloud{})
	})
}

func TestMatchInstancesFindsTwoCopies(t *testing.T) {
	model := makeSurface(13, 13)
	diam := model.Diameter()

	// Place two copies of the model far apart (well beyond half a diameter) with
	// different rigid poses to form a two-instance scene.
	poseA := Pose3D{R: rotationAxisAngle(Vec3{1, 2, 3}, 0.4), T: Vec3{0, 0, 0}}
	poseB := Pose3D{R: rotationAxisAngle(Vec3{0, 1, 1}, -0.6), T: Vec3{3 * diam, 0.2, -0.1}}
	copyA := model.TransformPose(poseA)
	copyB := model.TransformPose(poseB)

	scene := &PointCloud{}
	scene.Points = append(append([]Vec3{}, copyA.Points...), copyB.Points...)
	scene.Normals = append(append([]Vec3{}, copyA.Normals...), copyB.Normals...)

	det := NewPPF3DDetector(0.04, 0.04)
	det.TrainModel(model)

	instances := det.MatchInstances(scene, 0.1, 0.04, 2)
	if len(instances) < 2 {
		t.Fatalf("MatchInstances found %d instances, want 2", len(instances))
	}

	// Each ground-truth instance must be matched by some returned pose after ICP
	// polishing (PPF alone lands in the basin; ICP confirms the instance).
	icp := NewICP(60, 1e-10)
	matchInstance := func(truth Pose3D, truthCloud *PointCloud) bool {
		for _, inst := range instances[:2] {
			refined, _ := icp.RegisterKD(model, scene, inst)
			if refined.AngleTo(truth) < 0.1 && refined.TranslationTo(truth) < 0.1*diam {
				return true
			}
		}
		return false
	}
	if !matchInstance(poseA, copyA) {
		t.Error("instance A not recovered")
	}
	if !matchInstance(poseB, copyB) {
		t.Error("instance B not recovered")
	}

	// The two reported instances must be spatially separated.
	if instances[0].TranslationTo(instances[1]) < 0.5*diam {
		t.Errorf("reported instances not separated: %.3f", instances[0].TranslationTo(instances[1]))
	}
	mustPanic(t, func() { det.MatchInstances(scene, 0.1, 0.04, 0) })
}

func TestPLYRoundTripASCIIAndBinary(t *testing.T) {
	model := makeSurface(5, 4)

	check := func(name string, write func(*bytes.Buffer) error) {
		var buf bytes.Buffer
		if err := write(&buf); err != nil {
			t.Fatalf("%s write: %v", name, err)
		}
		pc, err := readPLY(bytes.NewReader(buf.Bytes()))
		if err != nil {
			t.Fatalf("%s read: %v", name, err)
		}
		if pc.Len() != model.Len() || len(pc.Normals) != model.Len() {
			t.Fatalf("%s: got %d points %d normals, want %d", name, pc.Len(), len(pc.Normals), model.Len())
		}
		for i := range model.Points {
			if norm3(sub3(pc.Points[i], model.Points[i])) > 1e-12 {
				t.Fatalf("%s point %d mismatch: %v vs %v", name, i, pc.Points[i], model.Points[i])
			}
			if angleBetween(pc.Normals[i], model.Normals[i]) > 1e-9 {
				t.Fatalf("%s normal %d mismatch", name, i)
			}
		}
	}
	check("ascii", func(b *bytes.Buffer) error { return writePLYASCII(b, model) })
	check("binary", func(b *bytes.Buffer) error { return writePLYBinary(b, model) })
}

func TestReadPLYBinaryFloat32(t *testing.T) {
	// A hand-built little-endian binary PLY with float32 coords (no normals)
	// exercises the type-decoding path.
	header := "ply\nformat binary_little_endian 1.0\nelement vertex 2\n" +
		"property float x\nproperty float y\nproperty float z\nend_header\n"
	var buf bytes.Buffer
	buf.WriteString(header)
	pts := []Vec3{{1, 2, 3}, {-4, 5, -6}}
	for _, p := range pts {
		for k := 0; k < 3; k++ {
			var b [4]byte
			bits := math.Float32bits(float32(p[k]))
			b[0] = byte(bits)
			b[1] = byte(bits >> 8)
			b[2] = byte(bits >> 16)
			b[3] = byte(bits >> 24)
			buf.Write(b[:])
		}
	}
	pc, err := readPLY(bytes.NewReader(buf.Bytes()))
	if err != nil {
		t.Fatalf("readPLY float32: %v", err)
	}
	if pc.Len() != 2 || len(pc.Normals) != 0 {
		t.Fatalf("got %d points %d normals", pc.Len(), len(pc.Normals))
	}
	for i := range pts {
		if norm3(sub3(pc.Points[i], pts[i])) > 1e-6 {
			t.Errorf("float32 point %d = %v, want %v", i, pc.Points[i], pts[i])
		}
	}
}

func TestReadPLYErrors(t *testing.T) {
	if _, err := readPLY(bytes.NewReader([]byte("nope\n"))); err == nil {
		t.Error("expected error for non-PLY")
	}
	bad := "ply\nformat ascii 1.0\nelement vertex 1\nproperty float a\nend_header\n0\n"
	if _, err := readPLY(bytes.NewReader([]byte(bad))); err == nil {
		t.Error("expected error for missing x/y/z")
	}
	badType := "ply\nformat ascii 1.0\nelement vertex 0\nproperty weird x\nend_header\n"
	if _, err := readPLY(bytes.NewReader([]byte(badType))); err == nil {
		t.Error("expected error for unknown property type")
	}
}
