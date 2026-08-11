# xfeatures2d — provides additional ("contrib") 2D feature detectors and descriptors on top

[![Go Reference](https://pkg.go.dev/badge/github.com/malcolmston/opencv/xfeatures2d.svg)](https://pkg.go.dev/github.com/malcolmston/opencv/xfeatures2d)

Package xfeatures2d provides additional ("contrib") 2D feature detectors and
descriptors on top of the standard-library-only OpenCV port
github.com/malcolmston/opencv (imported here as cv). It mirrors a useful
subset of OpenCV's opencv_contrib xfeatures2d module and complements the
detectors in the root package (cv.FASTCorners, cv.GoodFeaturesToTrack,
cv.CornerHarris).

The package is self-contained: it depends only on the Go standard library and
the root cv package. It deliberately does not import any sibling cv/*
subpackage, so the keypoint and descriptor types used here are declared
locally (see `KeyPoint`).

## Install

```sh
go get github.com/malcolmston/opencv@v0.8.0
```

```go
import "github.com/malcolmston/opencv/xfeatures2d"
```

## Usage

This is the package's own `ExampleAGAST`, so it compiles and its output is
asserted on every `go test ./xfeatures2d/`.

```go
img := cv.NewMat(60, 60, 1)
	for y := 20; y < 40; y++ {
		for x := 20; x < 40; x++ {
			img.Data[y*60+x] = 255
		}
	}

	kps := xfeatures2d.NewAGAST(30).Detect(img)
	fmt.Println("found corners:", len(kps) > 0)
```

```
found corners: true
```

## Exported surface

### Functions

| Function | What it does |
| --- | --- |
| `func HammingDistance(a, b []byte) int` | HammingDistance returns the number of differing bits between two bit-packed binary descriptors, i.e. |
| `func L2Distance(a, b []float64) float64` | L2Distance returns the Euclidean (L2) distance between two equal-length floating-point descriptors. |
| `func LUCIDDistance(a, b []byte) int` | LUCIDDistance returns the L1 distance between two LUCID rank descriptors, the natural dissimilarity for the rank-order representation. |
| `func SQFD(a, b []SignaturePoint, alpha float64) float64` | SQFD returns the Signature Quadratic Form Distance between two PCT signatures, the distance used with `PCTSignatures` (OpenCV's… |

### Types

| Type | What it is |
| --- | --- |
| `AGAST` | AGAST detects corners with an adaptive-threshold FAST segment test. |
| `BEBLID` | BEBLID computes the Boosted Efficient Binary Local Image Descriptor, a weight-free port of OpenCV's cv::xfeatures2d::BEBLID. |
| `BRIEF` | BRIEF computes the Binary Robust Independent Elementary Features descriptor, a port of OpenCV's cv::xfeatures2d::BriefDescriptorExtractor. |
| `BRISK` | BRISK computes the Binary Robust Invariant Scalable Keypoints descriptor. |
| `BoostDesc` | BoostDesc computes a weight-free boosted gradient binary descriptor in the spirit of OpenCV's cv::xfeatures2d::BoostDesc. |
| `DAISY` | DAISY computes the DAISY dense gradient-histogram descriptor, a port of OpenCV's cv::xfeatures2d::DAISY. |
| `DMatch` | DMatch is a correspondence between a descriptor in a query set and a descriptor in a train set, mirroring OpenCV's cv::DMatch. |
| `FREAK` | FREAK computes the Fast Retina Keypoint binary descriptor, a port of the descriptor stage of OpenCV's cv::xfeatures2d::FREAK. |
| `GFTTDetector` | GFTTDetector wraps cv.GoodFeaturesToTrack (the Shi–Tomasi "good features to track" corner detector) and returns the corners as `KeyPoint`s. |
| `HarrisLaplace` | HarrisLaplace detects scale-adapted corners, a simplified port of OpenCV's cv::xfeatures2d::HarrisLaplaceFeatureDetector. |
| `KeyPoint` | KeyPoint is a salient image point produced by a detector in this package. |
| `LATCH` | LATCH computes the Learned Arrangements of Three patCH codes binary descriptor, a port of OpenCV's cv::xfeatures2d::LATCH. |
| `LOGOSMatcher` | LOGOSMatcher filters putative matches with Local Geometric Support (LOGOS), a port in the spirit of OpenCV's cv::xfeatures2d::matchLOGOS. |
| `LUCID` | LUCID computes the Locally Uniform Comparison Image Descriptor, a port of OpenCV's cv::xfeatures2d::LUCID. |
| `MSDDetector` | MSDDetector detects Maximal Self-Dissimilarity interest points, a port of OpenCV's cv::xfeatures2d::MSDDetector. |
| `PCTSignatures` | PCTSignatures computes Position-Color-Texture signatures, a port of OpenCV's cv::xfeatures2d::PCTSignatures for single-channel (intensity) images. |
| `SURF` | SURF is a lightweight Speeded-Up Robust Features detector and descriptor, following the principles of OpenCV's cv::xfeatures2d::SURF without using… |
| `SignaturePoint` | SignaturePoint is one cluster of a PCT (Position-Color-Texture) signature: a weighted centroid in feature space. |
| `SimpleBlobDetector` | SimpleBlobDetector extracts blobs from an image and describes each as a `KeyPoint`. |
| `StarDetector` | StarDetector detects center–surround extrema, approximating OpenCV's cv::xfeatures2d::StarDetector (the CenSurE detector). |
| `TBMR` | TBMR detects Tree-Based Morse Regions, a port in the spirit of OpenCV's cv::xfeatures2d::TBMR. |
| `TEBLID` | TEBLID computes the Triplet-based Efficient Binary Local Image Descriptor, a weight-free port of OpenCV's cv::xfeatures2d::TEBLID. |
| `VGG` | VGG computes a weight-free gradient-pooling descriptor in the spirit of OpenCV's cv::xfeatures2d::VGG. |

<details>
<summary><code>AGAST</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewAGAST(threshold int) *AGAST` | NewAGAST returns an AGAST detector with the given segment-test threshold and non-maximum suppression enabled. |
| `func (a *AGAST) Detect(img *cv.Mat) []KeyPoint` | Detect finds corners in img and returns them as keypoints. |

</details>

<details>
<summary><code>BEBLID</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewBEBLID(bytes int) *BEBLID` | NewBEBLID returns a BEBLID extractor producing descriptors of the given byte length (typically 32 or 64). |
| `func (b *BEBLID) Compute(img *cv.Mat, keypoints []KeyPoint) ([]KeyPoint, [][]byte)` | Compute describes each keypoint of img and returns the keypoints (with Angle set to the estimated orientation) and their bit-packed descriptors. |
| `func (b *BEBLID) DescriptorSizeBytes() int` | DescriptorSizeBytes returns the number of bytes in each descriptor. |

</details>

<details>
<summary><code>BRIEF</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewBRIEF(bytes int) *BRIEF` | NewBRIEF returns a BRIEF extractor producing descriptors of the given byte length (typically 16, 32 or 64, i.e. |
| `func (b *BRIEF) Compute(img *cv.Mat, keypoints []KeyPoint) ([]KeyPoint, [][]byte)` | Compute describes each keypoint of img and returns the keypoints unchanged together with their bit-packed descriptors (one []byte of length… |
| `func (b *BRIEF) DescriptorSizeBytes() int` | DescriptorSizeBytes returns the number of bytes in each descriptor. |

</details>

<details>
<summary><code>BRISK</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewBRISK(agastThreshold int) *BRISK` | NewBRISK builds a BRISK detector/descriptor with the standard concentric-ring sampling pattern and the given AGAST corner threshold for detection. |
| `func (b *BRISK) Compute(img *cv.Mat, keypoints []KeyPoint) ([]KeyPoint, [][]byte)` | Compute describes each keypoint of img and returns the retained keypoints together with their bit-packed descriptors (one []byte per keypoint, all of… |
| `func (b *BRISK) DescriptorSizeBits() int` | DescriptorSizeBits returns the number of bits in each descriptor (one per short-distance sampling pair). |
| `func (b *BRISK) DescriptorSizeBytes() int` | DescriptorSizeBytes returns the number of bytes in each bit-packed descriptor. |
| `func (b *BRISK) DetectAndCompute(img *cv.Mat) ([]KeyPoint, [][]byte)` | DetectAndCompute detects keypoints with `AGAST` and describes them, returning the retained keypoints and their descriptors. |

</details>

<details>
<summary><code>BoostDesc</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewBoostDesc(bytes int) *BoostDesc` | NewBoostDesc returns a BoostDesc extractor producing descriptors of the given byte length (typically 32 or 64). |
| `func (b *BoostDesc) Compute(img *cv.Mat, keypoints []KeyPoint) ([]KeyPoint, [][]byte)` | Compute describes each keypoint of img and returns the keypoints (with Angle set to the intensity-centroid orientation) and their bit-packed… |
| `func (b *BoostDesc) DescriptorSizeBytes() int` | DescriptorSizeBytes returns the number of bytes in each descriptor. |

</details>

<details>
<summary><code>DAISY</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewDAISY() *DAISY` | NewDAISY returns a DAISY extractor with the common default configuration (radius 15, 3 rings, 8 points per ring, 8 orientations), yielding a… |
| `func (d *DAISY) Compute(img *cv.Mat, keypoints []KeyPoint) ([]KeyPoint, [][]float64)` | Compute describes each keypoint of img and returns the keypoints unchanged together with their float descriptors (one []float64 of length… |
| `func (d *DAISY) DescriptorSize() int` | DescriptorSize returns the number of floats in each descriptor. |

</details>

<details>
<summary><code>DMatch</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func MatchBruteForceHamming(query, train [][]byte) []DMatch` | MatchBruteForceHamming returns, for every query descriptor, its nearest train descriptor by Hamming distance (a brute-force cross-check-free matcher). |
| `func MatchBruteForceL2(query, train [][]float64) []DMatch` | MatchBruteForceL2 returns, for every query descriptor, its nearest train descriptor by L2 distance. |
| `func MatchGMS(rows1, cols1, rows2, cols2 int, kp1, kp2 []KeyPoint, matches []DMatch, withRotation, withScale bool, thresholdFactor float64) []DMatch` | MatchGMS filters putative matches with Grid-based Motion Statistics, a port of OpenCV's cv::xfeatures2d::matchGMS. |

</details>

<details>
<summary><code>FREAK</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewFREAK(agastThreshold int) *FREAK` | NewFREAK returns a FREAK descriptor with the standard retinal pattern and the given AGAST corner threshold for detection. |
| `func (f *FREAK) Compute(img *cv.Mat, keypoints []KeyPoint) ([]KeyPoint, [][]byte)` | Compute describes each keypoint of img and returns the keypoints (with Angle set to the estimated orientation) and their bit-packed descriptors. |
| `func (f *FREAK) DescriptorSizeBits() int` | DescriptorSizeBits returns the number of bits in each descriptor. |
| `func (f *FREAK) DescriptorSizeBytes() int` | DescriptorSizeBytes returns the number of bytes in each bit-packed descriptor. |
| `func (f *FREAK) DetectAndCompute(img *cv.Mat) ([]KeyPoint, [][]byte)` | DetectAndCompute detects keypoints with `AGAST` (sized to the FREAK base diameter) and describes them. |

</details>

<details>
<summary><code>GFTTDetector</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewGFTTDetector(maxCorners int) *GFTTDetector` | NewGFTTDetector returns a GFTTDetector with defaults matching OpenCV's (up to maxCorners corners, quality level 0.01, minimum distance 1, block size… |
| `func (g *GFTTDetector) Detect(img *cv.Mat) []KeyPoint` | Detect finds Shi–Tomasi corners in img and returns them as keypoints. |

</details>

<details>
<summary><code>HarrisLaplace</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewHarrisLaplace() *HarrisLaplace` | NewHarrisLaplace returns a HarrisLaplace detector with sensible defaults. |
| `func (h *HarrisLaplace) Detect(img *cv.Mat) []KeyPoint` | Detect finds scale-adapted Harris corners in img and returns them as keypoints. |

</details>

<details>
<summary><code>LATCH</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewLATCH(bytes int) *LATCH` | NewLATCH returns a LATCH extractor producing descriptors of the given byte length (typically 1, 2, 4, 8, 16 or 32 bytes). |
| `func (l *LATCH) Compute(img *cv.Mat, keypoints []KeyPoint) ([]KeyPoint, [][]byte)` | Compute describes each keypoint of img and returns the keypoints (with Angle set when RotationInvariant) and their bit-packed descriptors. |
| `func (l *LATCH) DescriptorSizeBytes() int` | DescriptorSizeBytes returns the number of bytes in each descriptor. |

</details>

<details>
<summary><code>LOGOSMatcher</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewLOGOSMatcher() *LOGOSMatcher` | NewLOGOSMatcher returns a LOGOSMatcher with sensible defaults. |
| `func (l *LOGOSMatcher) Filter(kp1, kp2 []KeyPoint, matches []DMatch) []DMatch` | Filter returns the subset of matches with enough local geometric support. |

</details>

<details>
<summary><code>LUCID</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewLUCID(kernelSize, blurSize int) *LUCID` | NewLUCID returns a LUCID extractor with the given neighbourhood and blur half sizes. |
| `func (l *LUCID) Compute(img *cv.Mat, keypoints []KeyPoint) ([]KeyPoint, [][]byte)` | Compute describes each keypoint of img and returns the keypoints unchanged together with their rank descriptors (one []byte of length DescriptorSize… |
| `func (l *LUCID) DescriptorSize() int` | DescriptorSize returns the number of rank entries in each descriptor, one per pixel of the sampled neighbourhood. |

</details>

<details>
<summary><code>MSDDetector</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewMSDDetector() *MSDDetector` | NewMSDDetector returns an MSDDetector with sensible defaults. |
| `func (m *MSDDetector) Detect(img *cv.Mat) []KeyPoint` | Detect finds MSD keypoints in img. |

</details>

<details>
<summary><code>PCTSignatures</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewPCTSignatures() *PCTSignatures` | NewPCTSignatures returns a PCTSignatures extractor with sensible defaults (400 samples, up to 8 clusters, 8 iterations). |
| `func (p *PCTSignatures) ComputeSignature(img *cv.Mat) []SignaturePoint` | ComputeSignature samples and clusters img and returns its PCT signature as a slice of weighted `SignaturePoint`s (weights sum to 1). |

</details>

<details>
<summary><code>SURF</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewSURF(hessianThreshold float64) *SURF` | NewSURF returns a SURF detector/descriptor with the given Hessian threshold and the standard first-octave filter sizes. |
| `func (s *SURF) Compute(img *cv.Mat, keypoints []KeyPoint) ([]KeyPoint, [][]float64)` | Compute describes each keypoint of img and returns the keypoints (with Angle set unless Upright) together with their 64-element float descriptors, L2… |
| `func (s *SURF) DescriptorSize() int` | DescriptorSize returns the number of floats in each descriptor (64). |
| `func (s *SURF) Detect(img *cv.Mat) []KeyPoint` | Detect finds Fast-Hessian keypoints in img. |
| `func (s *SURF) DetectAndCompute(img *cv.Mat) ([]KeyPoint, [][]float64)` | DetectAndCompute detects Fast-Hessian keypoints and describes them, returning the keypoints and their descriptors. |

</details>

<details>
<summary><code>SimpleBlobDetector</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewSimpleBlobDetector() *SimpleBlobDetector` | NewSimpleBlobDetector returns a SimpleBlobDetector configured with defaults close to OpenCV's: a threshold sweep of 50–220 in steps of 10, a… |
| `func (d *SimpleBlobDetector) Detect(img *cv.Mat) []KeyPoint` | Detect finds blobs in img and returns them as keypoints, sorted by descending response (repeatability). |

</details>

<details>
<summary><code>StarDetector</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewStarDetector() *StarDetector` | NewStarDetector returns a StarDetector with a default maximum scale and response threshold. |
| `func (s *StarDetector) Detect(img *cv.Mat) []KeyPoint` | Detect finds center–surround extrema in img and returns them as keypoints. |

</details>

<details>
<summary><code>TBMR</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewTBMR() *TBMR` | NewTBMR returns a TBMR detector with sensible defaults. |
| `func (t *TBMR) Detect(img *cv.Mat) []KeyPoint` | Detect finds stable extremal regions in img and returns them as keypoints. |

</details>

<details>
<summary><code>TEBLID</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewTEBLID(bytes int) *TEBLID` | NewTEBLID returns a TEBLID extractor producing descriptors of the given byte length (typically 32 or 64). |
| `func (t *TEBLID) Compute(img *cv.Mat, keypoints []KeyPoint) ([]KeyPoint, [][]byte)` | Compute describes each keypoint of img and returns the keypoints (with Angle set to the estimated orientation) and their bit-packed descriptors. |
| `func (t *TEBLID) DescriptorSizeBytes() int` | DescriptorSizeBytes returns the number of bytes in each descriptor. |

</details>

<details>
<summary><code>VGG</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewVGG() *VGG` | NewVGG returns a VGG descriptor with a GLOH-like default configuration (radius 20, 2 rings, 8 sectors, 8 orientation bins), a (1+2*8)*8 =… |
| `func (v *VGG) Compute(img *cv.Mat, keypoints []KeyPoint) ([]KeyPoint, [][]float64)` | Compute describes each keypoint of img and returns the keypoints (with Angle set to the intensity-centroid orientation) together with their float… |
| `func (v *VGG) DescriptorSize() int` | DescriptorSize returns the number of floats in each descriptor. |

</details>

Full signatures, doc comments and every runnable example are on
[pkg.go.dev](https://pkg.go.dev/github.com/malcolmston/opencv/xfeatures2d).

## License

MIT, as part of [`github.com/malcolmston/opencv`](..).
An independent re-implementation, not affiliated with or endorsed by the
original project.
