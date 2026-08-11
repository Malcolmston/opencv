# cudafeatures2d — CPU-backed, API-compatible mirror of OpenCV's cudafeatures2d module

[![Go Reference](https://pkg.go.dev/badge/github.com/malcolmston/opencv/cudafeatures2d.svg)](https://pkg.go.dev/github.com/malcolmston/opencv/cudafeatures2d)

Package cudafeatures2d is a CPU-backed, API-compatible mirror of OpenCV's
cudafeatures2d module for the standard-library-only OpenCV port
github.com/malcolmston/opencv (imported here as cv).

This package contains NO CUDA, NO cgo and NO GPU code. It exists so that code
written against OpenCV's GPU feature-detection API — `GpuMat` device
matrices, `Stream` asynchronous streams, and the cuda ORB / FAST / brute-force
matcher / good-features-to-track classes — can be ported to Go with minimal
surface changes. Every operation runs synchronously on the CPU. `GpuMat` is a
thin wrapper around a host *cv.Mat, `Stream` is a no-op, and the detectors and
matcher delegate to the pure-Go implementations in the root cv package and its
sibling github.com/malcolmston/opencv/features2d.

## Install

```sh
go get github.com/malcolmston/opencv@v0.8.0
```

```go
import "github.com/malcolmston/opencv/cudafeatures2d"
```

## Usage

This is the package's own `Example`, so it compiles and its output is
asserted on every `go test ./cudafeatures2d/`.

```go
img := cudafeatures2d.NewGpuMat(scene(128))
	orb := cudafeatures2d.CreateORB(50)

	kps, desc := orb.DetectAndCompute(img, nil)
	rows, cols := desc.Size()

	fmt.Printf("keypoints>0: %v, one row per keypoint: %v, descriptor bytes: %d\n",
		len(kps) > 0, rows == len(kps), cols)
```

```
keypoints>0: true, one row per keypoint: true, descriptor bytes: 32
```

## Exported surface

### Types

| Type | What it is |
| --- | --- |
| `CornersDetector` | CornersDetector is a CPU-backed mirror of cv::cuda::CornersDetector. |
| `DMatch` | DMatch records a query/train descriptor correspondence, aliased from the features2d package. |
| `DescriptorMatcher` | DescriptorMatcher is a CPU-backed mirror of cv::cuda::DescriptorMatcher. |
| `FastFeatureDetector` | FastFeatureDetector is a CPU-backed mirror of cv::cuda::FastFeatureDetector. |
| `GpuMat` | GpuMat is a CPU-backed stand-in for cv::cuda::GpuMat. |
| `KeyPoint` | KeyPoint is a salient image point, aliased from the features2d package so results interoperate directly with it. |
| `NormType` | NormType selects the distance used to compare descriptors, aliased from the features2d package. |
| `ORB` | ORB is a CPU-backed mirror of cv::cuda::ORB. |
| `Stream` | Stream is a CPU-backed no-op stand-in for cv::cuda::Stream. |

<details>
<summary><code>CornersDetector</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func CreateGoodFeaturesToTrackDetector(maxCorners int, qualityLevel, minDistance float64, blockSize int) *CornersDetector` | CreateGoodFeaturesToTrackDetector returns a Shi–Tomasi corner detector. |
| `func (d *CornersDetector) Detect(image, mask *GpuMat) []cv.Point` | Detect returns the strong Shi–Tomasi corners of the device image as host points, strongest first, optionally filtered by mask (pass a nil or empty… |
| `func (d *CornersDetector) DetectAsync(image, mask *GpuMat, _ *Stream) []cv.Point` | DetectAsync is the streamed form of `CornersDetector.Detect`. |

</details>

<details>
<summary><code>DescriptorMatcher</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func CreateBFMatcher(norm NormType) *DescriptorMatcher` | CreateBFMatcher returns a brute-force DescriptorMatcher using the given norm. |
| `func (m *DescriptorMatcher) KnnMatch(query, train *GpuMat, k int) [][]DMatch` | KnnMatch returns, for each query descriptor, its k nearest train descriptors sorted by ascending distance. |
| `func (m *DescriptorMatcher) Match(query, train *GpuMat) []DMatch` | Match returns the single best train match for each query descriptor, in query order. |
| `func (m *DescriptorMatcher) MatchConvert(raw [][]DMatch) []DMatch` | MatchConvert flattens a raw knn/radius match result (the per-query slices returned by `DescriptorMatcher.KnnMatch` or… |
| `func (m *DescriptorMatcher) RadiusMatch(query, train *GpuMat, maxDistance float64) [][]DMatch` | RadiusMatch returns, for each query descriptor, every train descriptor within maxDistance, sorted by ascending distance. |

</details>

<details>
<summary><code>FastFeatureDetector</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func CreateFastFeatureDetector(threshold int, nonmaxSuppression bool) *FastFeatureDetector` | CreateFastFeatureDetector returns a FAST detector with the given intensity threshold and non-maximum suppression setting. |
| `func (d *FastFeatureDetector) Detect(image, mask *GpuMat) []KeyPoint` | Detect returns the FAST corners of the device image as keypoints, optionally filtered by mask (pass a nil or empty GpuMat for no mask). |
| `func (d *FastFeatureDetector) DetectAsync(image, mask *GpuMat, _ *Stream) []KeyPoint` | DetectAsync is the streamed form of `FastFeatureDetector.Detect`. |

</details>

<details>
<summary><code>GpuMat</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func DescriptorsToGpuMat(rows [][]byte) *GpuMat` | DescriptorsToGpuMat packs host descriptor rows (such as those from `ORB.Convert`) into a device descriptor `GpuMat` suitable for… |
| `func NewGpuMat(m *cv.Mat) *GpuMat` | NewGpuMat returns a GpuMat holding a deep copy of the host matrix m, mirroring a host-to-device upload. |
| `func (g *GpuMat) Channels() int` | Channels returns the number of channels, or 0 when empty. |
| `func (g *GpuMat) Clone() *GpuMat` | Clone returns an independent deep copy of the GpuMat. |
| `func (g *GpuMat) Cols() int` | Cols returns the number of columns, or 0 when empty. |
| `func (g *GpuMat) Download() *cv.Mat` | Download returns a deep copy of the GpuMat's contents as a host *cv.Mat, mirroring cv::cuda::GpuMat::download. |
| `func (g *GpuMat) Empty() bool` | Empty reports whether the GpuMat holds no samples. |
| `func (g *GpuMat) Rows() int` | Rows returns the number of rows, or 0 when empty. |
| `func (g *GpuMat) Size() (rows, cols int)` | Size returns the matrix dimensions as (rows, cols), both 0 when empty. |
| `func (g *GpuMat) Upload(m *cv.Mat)` | Upload copies the host matrix m into the GpuMat, replacing any current contents. |

</details>

<details>
<summary><code>ORB</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func CreateORB(nFeatures int) *ORB` | CreateORB returns an ORB detector retaining up to nFeatures keypoints (nFeatures <= 0 keeps all detected corners). |
| `func (o *ORB) Compute(image *GpuMat, kps []KeyPoint) ([]KeyPoint, *GpuMat)` | Compute computes rotated BRIEF descriptors for the supplied keypoints on the device image, returning the (possibly reduced) keypoints and a device… |
| `func (o *ORB) Convert(descriptors *GpuMat) [][]byte` | Convert unpacks a device descriptor `GpuMat` (as returned by `ORB.DetectAndCompute` or `ORB.Compute`) into host-side bit-packed descriptor rows. |
| `func (o *ORB) DefaultNorm() NormType` | DefaultNorm returns the norm type ORB descriptors should be matched with (`NormHamming`). |
| `func (o *ORB) DescriptorSize() int` | DescriptorSize returns the descriptor length in bytes (32 for the default 256-bit rotated BRIEF descriptor). |
| `func (o *ORB) Detect(image, mask *GpuMat) []KeyPoint` | Detect finds oriented FAST keypoints in the device image, optionally filtered by mask (pass a nil or empty GpuMat for no mask). |
| `func (o *ORB) DetectAndCompute(image, mask *GpuMat) ([]KeyPoint, *GpuMat)` | DetectAndCompute detects oriented FAST keypoints in the device image and computes their rotated BRIEF descriptors in one call, returning the… |
| `func (o *ORB) DetectAndComputeAsync(image, mask *GpuMat, _ *Stream) ([]KeyPoint, *GpuMat)` | DetectAndComputeAsync is the streamed form of `ORB.DetectAndCompute`. |

</details>

<details>
<summary><code>Stream</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewStream() *Stream` | NewStream returns a new (inert) Stream. |
| `func (s *Stream) WaitForCompletion()` | WaitForCompletion returns immediately: all work in this package is already synchronous, so there is nothing to wait for. |

</details>

### Constants

`NormHamming`, `NormL2`

Full signatures, doc comments and every runnable example are on
[pkg.go.dev](https://pkg.go.dev/github.com/malcolmston/opencv/cudafeatures2d).

## License

MIT, as part of [`github.com/malcolmston/opencv`](..).
An independent re-implementation, not affiliated with or endorsed by the
original project.
