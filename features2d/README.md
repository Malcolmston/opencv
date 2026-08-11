# features2d — provides 2D feature detection, description and matching on top

[![Go Reference](https://pkg.go.dev/badge/github.com/malcolmston/opencv/features2d.svg)](https://pkg.go.dev/github.com/malcolmston/opencv/features2d)

Package features2d provides 2D feature detection, description and matching on
top of the standard-library-only OpenCV port github.com/malcolmston/opencv
(imported here as cv). It mirrors a useful subset of OpenCV's features2d
module: keypoints, binary descriptors, a detector/descriptor and a brute force
matcher.

A feature pipeline has three stages. Detection finds repeatable interest
points (`KeyPoint`s) such as corners. Description summarises the image
neighbourhood around each keypoint as a compact `Descriptors` vector — here
a binary string of bits packed into bytes. Matching compares two sets of
descriptors and reports the corresponding pairs as `DMatch`es.

## Install

```sh
go get github.com/malcolmston/opencv@v0.8.0
```

```go
import "github.com/malcolmston/opencv/features2d"
```

## Exported surface

### Functions

| Function | What it does |
| --- | --- |
| `func DrawKeypoints(img *cv.Mat, kps []KeyPoint) *cv.Mat` | DrawKeypoints returns a 3-channel copy of img with each keypoint drawn as a circle whose radius reflects its Size and, when an orientation is set… |
| `func DrawKeypointsFlags(img *cv.Mat, kps []KeyPoint, color cv.Scalar, flags DrawFlags) *cv.Mat` | DrawKeypointsFlags returns a 3-channel copy of img with the keypoints drawn in the given colour (RGB). |
| `func DrawMatches(img1 *cv.Mat, kp1 []KeyPoint, img2 *cv.Mat, kp2 []KeyPoint, matches []DMatch) *cv.Mat` | DrawMatches renders img1 and img2 side by side (img1 on the left) in a new 3-channel Mat, draws every keypoint, and connects each matched pair with a… |
| `func EvaluateFeatureDetector(img1, img2 *cv.Mat, h1to2 cv.PerspectiveMatrix, det Detector, tolerance float64) (repeatability float64, correspondences int)` | EvaluateFeatureDetector measures the repeatability of a `Detector` across an image pair related by the homography h1to2 (mapping img1 coordinates… |
| `func HammingDistance(a, b []byte) int` | HammingDistance returns the number of differing bits between two equal-length bit-packed descriptors. |

### Types

| Type | What it is |
| --- | --- |
| `AKAZE` | AKAZE is the Accelerated-KAZE detector and descriptor. |
| `AgastFeatureDetector` | AgastFeatureDetector implements the AGAST corner detector using the OAST 9_16 mask (an arc of at least 9 contiguous circle pixels all brighter than… |
| `BFMatcher` | BFMatcher is a brute-force descriptor matcher: it compares every query descriptor against every train descriptor. |
| `BOWImgDescriptorExtractor` | BOWImgDescriptorExtractor computes the bag-of-visual-words descriptor of an image from its feature descriptors and a vocabulary, mirroring OpenCV's… |
| `BOWKMeansTrainer` | BOWKMeansTrainer clusters float feature descriptors into a visual vocabulary with k-means, mirroring OpenCV's cv::BOWKMeansTrainer. |
| `BRIEF` | BRIEF computes a Binary Robust Independent Elementary Features descriptor. |
| `DMatch` | DMatch records a correspondence between a query descriptor and a train descriptor, as produced by a `BFMatcher`. |
| `DescriptorMatcher` | DescriptorMatcher is the subset of matcher behaviour used by `BOWImgDescriptorExtractor`: mapping each query descriptor to its single best train… |
| `Descriptors` | Descriptors holds a set of feature descriptors, one per keypoint. |
| `Detector` | Detector is implemented by any type that can find keypoints in an image, such as `ORB`, `SIFT`, `KAZE`, `AKAZE`, `FastFeatureDetector`,… |
| `DrawFlags` | DrawFlags controls how `DrawKeypointsFlags` renders keypoints, mirroring OpenCV's cv::DrawMatchesFlags. |
| `FastFeatureDetector` | FastFeatureDetector wraps the FAST-9 corner detector (cv.FASTCorners) in the OpenCV cv::FastFeatureDetector class form, producing `KeyPoint`s whose… |
| `FlannBasedMatcher` | FlannBasedMatcher is an approximate nearest-neighbour descriptor matcher built on a forest of randomised k-d trees, mirroring OpenCV's… |
| `GFTTDetector` | GFTTDetector wraps the Shi–Tomasi "good features to track" corner detector (cv.GoodFeaturesToTrack) in OpenCV's cv::GFTTDetector class form. |
| `KAZE` | KAZE is a multiscale 2D feature detector and descriptor operating in a nonlinear diffusion scale space. |
| `KeyPoint` | KeyPoint is a salient image point produced by a detector. |
| `KeyPointsFilter` | KeyPointsFilter provides the keypoint-list filtering utilities from OpenCV's cv::KeyPointsFilter. |
| `NormType` | NormType selects the distance used to compare descriptors. |
| `ORB` | ORB is an Oriented FAST and Rotated BRIEF detector and descriptor. |
| `RecallPrecisionPoint` | RecallPrecisionPoint is one sample of a recall/precision curve produced by `ComputeRecallPrecisionCurve`. |
| `SIFT` | SIFT is the Scale-Invariant Feature Transform detector and descriptor. |
| `SimpleBlobDetector` | SimpleBlobDetector detects blobs — connected regions that are darker or brighter than their surroundings — following the algorithm of OpenCV's… |

<details>
<summary><code>AKAZE</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewAKAZE(nFeatures int) *AKAZE` | NewAKAZE returns an AKAZE detector retaining up to nFeatures keypoints (pass <= 0 for all) with default scale-space parameters. |
| `func (a *AKAZE) Detect(img *cv.Mat) []KeyPoint` | Detect returns the AKAZE keypoints of img without descriptors. |
| `func (a *AKAZE) DetectAndCompute(img *cv.Mat) ([]KeyPoint, [][]byte)` | DetectAndCompute detects AKAZE keypoints and computes their binary M-LDB descriptors, returning the keypoints and one bit-packed descriptor row each. |

</details>

<details>
<summary><code>AgastFeatureDetector</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewAgastFeatureDetector(threshold int) *AgastFeatureDetector` | NewAgastFeatureDetector returns an AGAST detector with the given threshold and non-maximum suppression enabled. |
| `func (d *AgastFeatureDetector) Detect(img *cv.Mat) []KeyPoint` | Detect returns the AGAST corners of img. |

</details>

<details>
<summary><code>BFMatcher</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewBFMatcher(norm NormType) *BFMatcher` | NewBFMatcher returns a brute-force matcher using the given norm. |
| `func (m *BFMatcher) KnnMatch(query, train Descriptors, k int) [][]DMatch` | KnnMatch returns, for each query descriptor, its k nearest train descriptors as a slice of DMatch sorted by ascending distance (ties broken by lower… |
| `func (m *BFMatcher) Match(query, train Descriptors) []DMatch` | Match returns the best train match for each query descriptor. |

</details>

<details>
<summary><code>BOWImgDescriptorExtractor</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewBOWImgDescriptorExtractor(matcher DescriptorMatcher) *BOWImgDescriptorExtractor` | NewBOWImgDescriptorExtractor returns an extractor that assigns descriptors to words using the given matcher (typically a float `BFMatcher` or a… |
| `func (e *BOWImgDescriptorExtractor) Compute(desc Descriptors) []float64` | Compute returns the bag-of-words histogram for the given image descriptors: an L1-normalised vector of length VocabularySize. |
| `func (e *BOWImgDescriptorExtractor) SetVocabulary(vocabulary Descriptors)` | SetVocabulary stores the visual vocabulary (float cluster centres, as returned by `BOWKMeansTrainer.Cluster`). |
| `func (e *BOWImgDescriptorExtractor) VocabularySize() int` | VocabularySize returns the number of visual words (histogram bins). |

</details>

<details>
<summary><code>BOWKMeansTrainer</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewBOWKMeansTrainer(clusterCount, maxIter int, epsilon float64) *BOWKMeansTrainer` | NewBOWKMeansTrainer returns a trainer that will produce a vocabulary of clusterCount words. |
| `func (t *BOWKMeansTrainer) Add(desc Descriptors)` | Add appends the float descriptor rows of one image (or any batch) to the training set. |
| `func (t *BOWKMeansTrainer) Cluster() Descriptors` | Cluster runs k-means over the accumulated descriptors and returns the cluster centres as float `Descriptors`. |
| `func (t *BOWKMeansTrainer) DescriptorsCount() int` | DescriptorsCount returns the number of descriptors added so far. |

</details>

<details>
<summary><code>BRIEF</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewBRIEF(patchSize, numBits int) *BRIEF` | NewBRIEF returns a BRIEF descriptor extractor with the given patch size and a freshly generated deterministic pattern of numBits tests. |
| `func (b *BRIEF) Compute(img *cv.Mat, kps []KeyPoint) ([]KeyPoint, [][]byte)` | Compute describes the given keypoints on img and returns the surviving keypoints together with their bit-packed descriptors, one row per keypoint. |

</details>

<details>
<summary><code>DMatch</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func RatioTest(knn [][]DMatch, ratio float64) []DMatch` | RatioTest applies Lowe's ratio test to a k>=2 KnnMatch result: a query's best match is kept only when its distance is strictly less than ratio times… |

</details>

<details>
<summary><code>Descriptors</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewBinaryDescriptors(rows [][]byte) Descriptors` | NewBinaryDescriptors wraps bit-packed descriptor rows (such as the [][]byte returned by `ORB.DetectAndCompute`) in a `Descriptors`. |
| `func NewFloatDescriptors(rows [][]float64) Descriptors` | NewFloatDescriptors wraps float descriptor rows in a `Descriptors`. |
| `func (d Descriptors) IsBinary() bool` | IsBinary reports whether the descriptors are binary (Hamming) rather than float (Euclidean). |
| `func (d Descriptors) Len() int` | Len returns the number of descriptor rows. |

</details>

<details>
<summary><code>FastFeatureDetector</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewFastFeatureDetector(threshold int) *FastFeatureDetector` | NewFastFeatureDetector returns a FAST detector with the given threshold and non-maximum suppression enabled. |
| `func (d *FastFeatureDetector) Detect(img *cv.Mat) []KeyPoint` | Detect returns the FAST corners of img as keypoints of Size 7 (the diameter of the FAST detection circle) and Angle -1 (FAST assigns no orientation). |

</details>

<details>
<summary><code>FlannBasedMatcher</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewFlannBasedMatcher(trees, checks int) *FlannBasedMatcher` | NewFlannBasedMatcher returns a FLANN matcher with the given number of trees and search-check budget. |
| `func (m *FlannBasedMatcher) KnnMatch(query, train Descriptors, k int) [][]DMatch` | KnnMatch returns, for each query descriptor, up to its k approximate nearest train descriptors sorted by ascending distance. |
| `func (m *FlannBasedMatcher) Match(query, train Descriptors) []DMatch` | Match returns the single best train match for each query descriptor, in query order. |

</details>

<details>
<summary><code>GFTTDetector</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewGFTTDetector(maxCorners int) *GFTTDetector` | NewGFTTDetector returns a GFTT detector retaining up to maxCorners corners with the default quality level, spacing and block size. |
| `func (d *GFTTDetector) Detect(img *cv.Mat) []KeyPoint` | Detect returns the strong Shi–Tomasi corners of img as keypoints of Size equal to the block size and Angle -1. |

</details>

<details>
<summary><code>KAZE</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewKAZE(nFeatures int) *KAZE` | NewKAZE returns a KAZE detector retaining up to nFeatures keypoints (pass <= 0 for all) with default scale-space parameters. |
| `func (a *KAZE) Detect(img *cv.Mat) []KeyPoint` | Detect returns the KAZE keypoints of img without descriptors. |
| `func (a *KAZE) DetectAndCompute(img *cv.Mat) ([]KeyPoint, [][]float64)` | DetectAndCompute detects KAZE keypoints and computes their 64-dimensional float M-SURF descriptors, returning parallel slices. |

</details>

<details>
<summary><code>KeyPointsFilter</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func (KeyPointsFilter) RemoveDuplicated(kps []KeyPoint) []KeyPoint` | RemoveDuplicated removes keypoints that share the same integer position, Size and Angle, keeping the first occurrence of each. |
| `func (KeyPointsFilter) RetainBest(kps []KeyPoint, n int) []KeyPoint` | RetainBest returns the n keypoints with the strongest Response, keeping every keypoint whose response ties the n-th strongest (so the result may… |
| `func (KeyPointsFilter) RunByImageBorder(kps []KeyPoint, rows, cols, borderSize int) []KeyPoint` | RunByImageBorder removes keypoints whose neighbourhood (a disc of radius borderSize, or the keypoint centre when borderSize <= 0) falls outside the… |
| `func (KeyPointsFilter) RunByKeypointSize(kps []KeyPoint, minSize, maxSize float64) []KeyPoint` | RunByKeypointSize keeps only keypoints whose Size lies in the half-open range [minSize, maxSize). |

</details>

<details>
<summary><code>ORB</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewORB(nFeatures int) *ORB` | NewORB returns an ORB detector retaining up to nFeatures keypoints with the default FAST threshold and patch size. |
| `func (o *ORB) Detect(img *cv.Mat) []KeyPoint` | Detect finds oriented FAST keypoints in img without computing descriptors. |
| `func (o *ORB) DetectAndCompute(img *cv.Mat) ([]KeyPoint, [][]byte)` | DetectAndCompute detects oriented FAST keypoints in img and computes their rotated BRIEF descriptors, returning parallel slices: the i-th descriptor… |

</details>

<details>
<summary><code>RecallPrecisionPoint</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func ComputeRecallPrecisionCurve(matches [][]DMatch, correct [][]bool) []RecallPrecisionPoint` | ComputeRecallPrecisionCurve builds a recall/precision curve from k-nearest matches and a mask marking which of those matches are geometrically… |

</details>

<details>
<summary><code>SIFT</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewSIFT(nFeatures int) *SIFT` | NewSIFT returns a SIFT detector retaining up to nFeatures keypoints (pass nFeatures <= 0 to keep all) with all other parameters at their defaults. |
| `func (s *SIFT) Detect(img *cv.Mat) []KeyPoint` | Detect returns the SIFT keypoints of img without descriptors. |
| `func (s *SIFT) DetectAndCompute(img *cv.Mat) ([]KeyPoint, [][]float64)` | DetectAndCompute detects SIFT keypoints and computes their 128-dimensional float descriptors, returning parallel slices. |

</details>

<details>
<summary><code>SimpleBlobDetector</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewSimpleBlobDetector() *SimpleBlobDetector` | NewSimpleBlobDetector returns a blob detector configured for dark blobs on a light background with default thresholds and area bounds. |
| `func (d *SimpleBlobDetector) Detect(img *cv.Mat) []KeyPoint` | Detect returns the detected blobs as keypoints, with Pt at the blob centroid, Size the mean blob diameter and Response the number of thresholds that… |

</details>

Full signatures, doc comments and every runnable example are on
[pkg.go.dev](https://pkg.go.dev/github.com/malcolmston/opencv/features2d).

## License

MIT, as part of [`github.com/malcolmston/opencv`](..).
An independent re-implementation, not affiliated with or endorsed by the
original project.
