# surface_matching — standard-library-only implementation of 3-D surface matching by Point-Pair Features

[![Go Reference](https://pkg.go.dev/badge/github.com/malcolmston/opencv/surface_matching.svg)](https://pkg.go.dev/github.com/malcolmston/opencv/surface_matching)

Package surface_matching is a standard-library-only implementation of 3-D
surface matching by Point-Pair Features (PPF), mirroring the algorithm of
OpenCV's contrib surface_matching module (Drost, Ulrich, Navab & Ilic, "Model
Globally, Match Locally: Efficient and Robust 3D Object Recognition", CVPR
2010).

It depends only on the Go standard library and, nominally, the root cv package
(github.com/malcolmston/opencv). It uses no cgo, no third-party code, and no
sibling cv/* subpackage: because the module works with 3-D point clouds rather
than images, every numerical kernel it needs — 3×3 matrix algebra, a
symmetric Jacobi eigensolver, a 3×3 SVD and quaternion averaging — is
implemented locally in linalg.go.

## Install

```sh
go get github.com/malcolmston/opencv@v0.8.0
```

```go
import "github.com/malcolmston/opencv/surface_matching"
```

## Usage

This is the package's own `Example`, so it compiles and its output is
asserted on every `go test ./surface_matching/`.

```go
model := buildSurface()

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
```

```
candidates: true
rotation recovered: true
translation recovered: true
tight residual: true
```

## Exported surface

### Functions

| Function | What it does |
| --- | --- |
| `func PoseInliers(model, scene *PointCloud, pose Pose3D, inlierDist float64) int` | PoseInliers counts how many model points, after being mapped into the scene by pose, land within inlierDist of some scene point. |
| `func ScorePose(model, scene *PointCloud, pose Pose3D, inlierDist float64) float64` | ScorePose returns the fraction of model points that are inliers under pose (see `PoseInliers`), a value in [0, 1] where 1 means every model point has… |
| `func ScorePoseNormals(model, scene *PointCloud, pose Pose3D, inlierDist, normalThresh float64) float64` | ScorePoseNormals is a stricter verification that counts a model point as an inlier only when its nearest scene point is both within inlierDist and… |
| `func VerifyPose(model, scene *PointCloud, pose Pose3D, inlierDist, minInlierRatio float64) (bool, float64)` | VerifyPose reports whether pose is acceptable, i.e. |
| `func WritePLY(path string, pc *PointCloud) error` | WritePLY writes the cloud to path as an ASCII PLY file with x, y, z vertex coordinates and, when the cloud carries normals, nx, ny, nz. |
| `func WritePLYBinary(path string, pc *PointCloud) error` | WritePLYBinary writes the cloud to path as a little-endian binary PLY file (format binary_little_endian 1.0) with double-precision x, y, z and, when… |

### Types

| Type | What it is |
| --- | --- |
| `ICP` | ICP refines a rigid alignment with point-to-point iterative closest point. |
| `KDTree3D` | KDTree3D is a static, balanced 3-D k-d tree over a set of points. |
| `Mat3` | Mat3 is a row-major 3×3 matrix. |
| `Neighbor` | Neighbor is a query result: the index of a point in the tree's underlying slice and its squared Euclidean distance to the query point. |
| `NormalOrientation` | NormalOrientation selects how estimated normals are oriented. |
| `PPF3DDetector` | PPF3DDetector implements 3-D surface matching with Point-Pair Features (PPF) after Drost et al., "Model Globally, Match Locally" (CVPR 2010). |
| `PPFHashIndex` | PPFHashIndex is a compact, immutable, cache-friendly view of a trained detector's point-pair-feature table. |
| `PointCloud` | PointCloud is an oriented point cloud: a slice of 3-D points and a parallel slice of unit surface normals. |
| `Pose3D` | Pose3D is a rigid transform in 3-D: a rotation matrix R and a translation vector T that together map a model point p to R·p + T. It is the output of… |
| `Vec3` | Vec3 is a point or vector in right-handed 3-D space. |

<details>
<summary><code>ICP</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewICP(maxIterations int, tolerance float64) *ICP` | NewICP returns an ICP configured with the given iteration cap and convergence tolerance and a default outlier-rejection scale of 2.5×. |
| `func (icp *ICP) Register(model, scene *PointCloud, init Pose3D) (Pose3D, float64)` | Register refines the alignment of the model cloud onto the scene cloud starting from init and returns the refined model-to-scene pose together with… |
| `func (icp *ICP) RegisterKD(model, scene *PointCloud, init Pose3D) (Pose3D, float64)` | RegisterKD refines the alignment of model onto scene exactly as `ICP.Register` does, but accelerates the per-iteration nearest-neighbour search with… |
| `func (icp *ICP) RegisterMultiScale(model, scene *PointCloud, init Pose3D, levels int) (Pose3D, float64)` | RegisterMultiScale runs a coarse-to-fine (pyramid) point-to-point ICP. |
| `func (icp *ICP) RegisterPointToPlane(model, scene *PointCloud, init Pose3D) (Pose3D, float64)` | RegisterPointToPlane refines the alignment of model onto scene by minimising the point-to-plane error: the squared distance from each transformed… |

</details>

<details>
<summary><code>KDTree3D</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewKDTree3D(points []Vec3) *KDTree3D` | NewKDTree3D builds a balanced k-d tree over the given points. |
| `func (t *KDTree3D) Len() int` | Len reports how many points the tree indexes. |
| `func (t *KDTree3D) Nearest(q Vec3) (int, float64)` | Nearest returns the index of the point closest to q and the squared distance to it. |
| `func (t *KDTree3D) NearestK(q Vec3, k int) []Neighbor` | NearestK returns up to k neighbours of q ordered by increasing squared distance (with equal distances further ordered by point index). |
| `func (t *KDTree3D) RadiusSearch(q Vec3, radius float64) []int` | RadiusSearch returns the indices of all points within radius of q, ordered by increasing distance. |

</details>

<details>
<summary><code>PPF3DDetector</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewPPF3DDetector(relativeSamplingStep, relativeDistanceStep float64) *PPF3DDetector` | NewPPF3DDetector creates a detector. |
| `func NewPPF3DDetectorAngles(relativeSamplingStep, relativeDistanceStep float64, numAngles int) *PPF3DDetector` | NewPPF3DDetectorAngles is `NewPPF3DDetector` with an explicit number of angular bins. |
| `func (d *PPF3DDetector) BuildHashIndex() *PPFHashIndex` | BuildHashIndex flattens the detector's trained hash table into a `PPFHashIndex`. |
| `func (d *PPF3DDetector) Match(scene *PointCloud, relSampleStep, relDistanceStep float64) []Pose3D` | Match aligns a scene cloud against the trained model and returns candidate poses that map model points into the scene, sorted by descending vote… |
| `func (d *PPF3DDetector) MatchInstances(scene *PointCloud, relSampleStep, relDistanceStep float64, maxInstances int) []Pose3D` | MatchInstances detects up to maxInstances distinct occurrences of the trained model in the scene and returns their poses in descending vote order. |
| `func (d *PPF3DDetector) SampledModel() *PointCloud` | SampledModel returns the down-sampled model cloud built during training, or nil if the detector is untrained. |
| `func (d *PPF3DDetector) TrainModel(model *PointCloud)` | TrainModel builds the point-pair-feature hash table from a model cloud. |
| `func (d *PPF3DDetector) TrainModelSpread(model *PointCloud)` | TrainModelSpread trains the detector like `PPF3DDetector.TrainModel` but additionally inserts every model pair into the immediately neighbouring… |
| `func (d *PPF3DDetector) Trained() bool` | Trained reports whether the detector has a hash table ready for matching. |

</details>

<details>
<summary><code>PPFHashIndex</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func (idx *PPFHashIndex) Len() int` | Len returns the number of distinct feature keys in the index. |
| `func (idx *PPFHashIndex) Lookup(key uint64) []ppfInfo` | Lookup returns the pair records stored under key, or nil if the key is absent. |

</details>

<details>
<summary><code>PointCloud</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func ComputeNormalsPC3d(pc *PointCloud, k int, orient NormalOrientation, viewpoint Vec3) *PointCloud` | ComputeNormalsPC3d estimates a unit surface normal at every point from the local tangent plane of its k nearest neighbours (the smallest-eigenvalue… |
| `func ComputeNormalsRadius(pc *PointCloud, radius float64, orient NormalOrientation, viewpoint Vec3) *PointCloud` | ComputeNormalsRadius estimates normals like `ComputeNormalsPC3d` but from all neighbours within the given radius rather than a fixed count, so the… |
| `func LoadPLY(path string) (*PointCloud, error)` | LoadPLY reads an ASCII PLY file into a point cloud. |
| `func NewPointCloud(points, normals []Vec3) *PointCloud` | NewPointCloud builds a cloud from parallel point and normal slices. |
| `func ReadPLY(path string) (*PointCloud, error)` | ReadPLY reads a PLY file in either ASCII or binary (little- or big-endian) format, auto-detecting from the header. |
| `func SamplePCByQuantization(pc *PointCloud, relStep float64) *PointCloud` | SamplePCByQuantization subdivides the cloud's axis-aligned bounding box into a regular grid of (1/relStep)³ cells — each cell spans relStep of the… |
| `func SamplePCUniform(pc *PointCloud, step int) *PointCloud` | SamplePCUniform keeps every step-th point of the cloud (indices 0, step, 2·step, …), returning a new cloud that carries the matching normals when… |
| `func TransformPCPose(pc *PointCloud, pose Pose3D) *PointCloud` | TransformPCPose returns a new cloud with the pose's rigid transform applied to every point (p ↦ R·p + T) and every normal (n ↦ R·n). |
| `func (pc *PointCloud) Centroid() Vec3` | Centroid returns the arithmetic mean of the points. |
| `func (pc *PointCloud) Clone() *PointCloud` | Clone returns a deep copy of the cloud. |
| `func (pc *PointCloud) ComputeNormals(k int, viewpoint Vec3)` | ComputeNormals estimates a unit surface normal at every point from the local tangent plane of its k nearest neighbours (principal component analysis:… |
| `func (pc *PointCloud) Diameter() float64` | Diameter returns the length of the bounding-box diagonal, the scale used to derive absolute distance and sampling steps from the relative steps… |
| `func (pc *PointCloud) Len() int` | Len returns the number of points in the cloud. |
| `func (pc *PointCloud) Transform(r Mat3, t Vec3) *PointCloud` | Transform returns a new cloud with the rigid transform applied: each point maps to R·p + t and each normal to R·n (translation does not affect… |
| `func (pc *PointCloud) TransformPose(p Pose3D) *PointCloud` | TransformPose returns a new cloud with the pose's rigid transform applied to every point and normal. |
| `func (pc *PointCloud) VoxelDownsample(voxelSize float64) *PointCloud` | VoxelDownsample reduces the cloud by averaging every point (and normal) that falls into the same cubic cell of a regular grid of the given side… |

</details>

<details>
<summary><code>Pose3D</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func AveragePoses(poses []Pose3D) Pose3D` | AveragePoses returns a single pose that averages the given poses: a vote-weighted quaternion mean of the rotations (aligned to a common hemisphere to… |
| `func ClusterPoses(poses []Pose3D, angleThresh, transThresh float64) []Pose3D` | ClusterPoses groups poses whose rotations agree within angleThresh radians and whose translations agree within transThresh, then replaces each group… |
| `func IdentityPose() Pose3D` | IdentityPose returns the identity transform (no rotation, no translation). |
| `func NewPose(r Mat3, t Vec3) Pose3D` | NewPose builds a pose from a rotation matrix and translation vector. |
| `func SortPosesByVotes(poses []Pose3D) []Pose3D` | SortPosesByVotes returns a copy of poses ordered by descending vote support. |
| `func SuppressNonMaximum(poses []Pose3D, angleThresh, transThresh float64) []Pose3D` | SuppressNonMaximum performs greedy non-maximum suppression over a set of pose hypotheses: considering poses in descending vote order, it keeps a pose… |
| `func (p Pose3D) AngleTo(o Pose3D) float64` | AngleTo returns the geodesic rotation angle in radians between two poses' rotations, a scale-free measure of their rotational disagreement. |
| `func (p Pose3D) Apply(pt Vec3) Vec3` | Apply maps a point through the pose, returning R·p + T. |
| `func (p Pose3D) ApplyNormal(n Vec3) Vec3` | ApplyNormal maps a direction (such as a surface normal) through the pose's rotation only; translation does not affect directions. |
| `func (p Pose3D) Compose(q Pose3D) Pose3D` | Compose returns the pose equivalent to applying q first and then p, i.e. |
| `func (p Pose3D) Inverse() Pose3D` | Inverse returns the inverse rigid transform, mapping R·p + T back to p. |
| `func (p Pose3D) Matrix() [4][4]float64` | Matrix returns the pose as a 4×4 homogeneous transform in row-major order: |
| `func (p Pose3D) String() string` | String renders the pose compactly for debugging. |
| `func (p Pose3D) TranslationTo(o Pose3D) float64` | TranslationTo returns the Euclidean distance between two poses' translations. |

</details>

### Constants

`DefaultNumAngles`

Full signatures, doc comments and every runnable example are on
[pkg.go.dev](https://pkg.go.dev/github.com/malcolmston/opencv/surface_matching).

## License

MIT, as part of [`github.com/malcolmston/opencv`](..).
An independent re-implementation, not affiliated with or endorsed by the
original project.
