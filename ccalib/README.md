# ccalib — standard-library-only port of OpenCV's ccalib module

[![Go Reference](https://pkg.go.dev/badge/github.com/malcolmston/opencv/ccalib.svg)](https://pkg.go.dev/github.com/malcolmston/opencv/ccalib)

Package ccalib is a standard-library-only port of OpenCV's ccalib module —
custom calibration patterns and the omnidirectional (fisheye / catadioptric)
camera model — built on top of the root cv package
(github.com/malcolmston/opencv).

It depends only on the Go standard library and the root cv package: no cgo, no
third-party code. The numerical kernels it needs (a symmetric Jacobi
eigensolver, a 3×3 SVD, Gaussian elimination, a normalised-DLT homography
solver and a Levenberg–Marquardt optimiser) are implemented from scratch in
linalg.go.

## Install

```sh
go get github.com/malcolmston/opencv@v0.8.0
```

```go
import "github.com/malcolmston/opencv/ccalib"
```

## Usage

This is the package's own `Example`, so it compiles and its output is
asserted on every `go test ./ccalib/`.

```go
cam := ccalib.OmniModel{
		Fx: 280, Fy: 285, Cx: 320, Cy: 240,
		Xi: 1.05, K1: -0.02, K2: 0.01, P1: 0.001, P2: -0.0008,
	}

	obj := [][3]float64{{0.5, 0.3, 4}}
	pix := ccalib.Omnidir.ProjectPoints(obj, [3]float64{0, 0, 0}, [3]float64{0, 0, 0}, cam.K(), cam.Xi, cam.Dist())

	Knew := [3][3]float64{{250, 0, 320}, {0, 250, 240}, {0, 0, 1}}
	R := [3][3]float64{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}
	rect := ccalib.Omnidir.Undistort(pix, cam.K(), cam.Xi, cam.Dist(), Knew, R)

	fmt.Printf("projected: (%.2f, %.2f)\n", pix[0][0], pix[0][1])
	fmt.Printf("rectified: (%.2f, %.2f)\n", rect[0][0], rect[0][1])
```

```
projected: (336.98, 250.37)
rectified: (351.25, 258.75)
```

## Exported surface

### Types

| Type | What it is |
| --- | --- |
| `CustomPattern` | CustomPattern is a feature-based calibration target built from an arbitrary textured template image, mirroring OpenCV's cv::ccalib::CustomPattern. |
| `MultiCameraObservation` | MultiCameraObservation records one camera's view of one pattern instance: the pattern's model points (ObjectPoints, with Z = 0) and their observed… |
| `MultiCameraResult` | MultiCameraResult holds the recovered geometry. |
| `OmniModel` | OmniModel bundles the intrinsic parameters of an omnidirectional camera: the pinhole intrinsics (focal lengths, principal point and skew), the mirror… |
| `Pose` | Pose is a rigid transform: a rotation R and translation T that map a point p to R·p + T. The zero value is the identity rotation with zero… |
| `RandomPatternCornerFinder` | RandomPatternCornerFinder matches captured views of a random-dot target against a reference image of that target, producing the 3D object points and… |
| `RandomPatternGenerator` | RandomPatternGenerator synthesises a random-dot calibration target: a set of bright filled discs scattered on a dark background. |
| `RectifyFlag` | RectifyFlag selects the target projection used when undistorting an omnidirectional image to a conventional one. |

<details>
<summary><code>CustomPattern</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewCustomPattern() *CustomPattern` | NewCustomPattern returns an empty, uninitialised custom pattern. |
| `func (p *CustomPattern) Create(template *cv.Mat, physWidth, physHeight float64) bool` | Create initialises the pattern from a template image whose printed size is physWidth×physHeight (in any consistent physical unit). |
| `func (p *CustomPattern) FindPattern(scene *cv.Mat) (objPts [][3]float64, imgPts [][2]float64, ok bool)` | FindPattern locates the template inside a scene image via feature matching and returns the matched object points (3D, Z = 0, in the template's… |
| `func (p *CustomPattern) IsInitialized() bool` | IsInitialized reports whether the pattern has been successfully created. |
| `func (p *CustomPattern) KeypointCount() int` | KeypointCount returns the number of template features extracted by `CustomPattern.Create`. |
| `func (p *CustomPattern) SetMinMatches(n int)` | SetMinMatches overrides the minimum number of consistent correspondences required for `CustomPattern.FindPattern` to succeed (default 8). |

</details>

<details>
<summary><code>OmniModel</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewOmniModel(k [3][3]float64, xi float64, dist []float64) OmniModel` | NewOmniModel builds an `OmniModel` from a 3×3 intrinsic matrix, the mirror parameter xi and a distortion slice ordered [K1, K2, P1, P2] (a nil or… |
| `func (m OmniModel) Dist() []float64` | Dist returns the distortion coefficients [K1, K2, P1, P2]. |
| `func (m OmniModel) K() [3][3]float64` | K returns the 3×3 intrinsic matrix of the model in row-major order. |

</details>

<details>
<summary><code>Pose</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func IdentityPose() Pose` | IdentityPose returns the identity rigid transform. |
| `func (p Pose) Apply(v [3]float64) [3]float64` | Apply returns R·p + T. |
| `func (p Pose) Compose(q Pose) Pose` | Compose returns the transform equivalent to applying q first, then p (p ∘ q): (p·q).R = p.R·q.R, (p·q).T = p.R·q.T + p.T. |
| `func (p Pose) Inverse() Pose` | Inverse returns the inverse rigid transform. |
| `func (p Pose) Rvec() [3]float64` | Rvec returns the rotation as an axis-angle vector. |

</details>

<details>
<summary><code>RandomPatternCornerFinder</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewRandomPatternCornerFinder(physWidth, physHeight float64, minMatches int) *RandomPatternCornerFinder` | NewRandomPatternCornerFinder creates a finder for a target whose printed size is physWidth×physHeight (in any consistent unit). |
| `func (f *RandomPatternCornerFinder) ComputeObjectImagePoints(observation *cv.Mat) (objPts [][3]float64, imgPts [][2]float64, ok bool)` | ComputeObjectImagePoints detects the target in the observation image and returns the matched object points (3D, with Z = 0, in the physical units of… |
| `func (f *RandomPatternCornerFinder) LoadPattern(pattern *cv.Mat) bool` | LoadPattern registers the reference image of the target, detecting its dots and building their descriptors. |
| `func (f *RandomPatternCornerFinder) LoadPatternBlobs(centers [][2]float64, refWidth, refHeight int) bool` | LoadPatternBlobs registers the reference dots directly from known centres (for instance those returned by `RandomPatternGenerator.Centers`), skipping… |

</details>

<details>
<summary><code>RandomPatternGenerator</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewRandomPatternGenerator(width, height, numDots int, seed int64) *RandomPatternGenerator` | NewRandomPatternGenerator returns a generator producing a width×height, single-channel pattern of numDots dots, seeded with seed for reproducibility. |
| `func (g *RandomPatternGenerator) Centers() [][2]float64` | Centers returns a copy of the dot centres of the most recently generated pattern, in pixel coordinates. |
| `func (g *RandomPatternGenerator) Generate() *cv.Mat` | Generate renders the pattern and returns it as a single-channel `cv.Mat` with bright dots (value 255) on a black background. |

</details>

### Variables

`Omnidir`

Full signatures, doc comments and every runnable example are on
[pkg.go.dev](https://pkg.go.dev/github.com/malcolmston/opencv/ccalib).

## License

MIT, as part of [`github.com/malcolmston/opencv`](..).
An independent re-implementation, not affiliated with or endorsed by the
original project.
