# linedescriptor — from-scratch, standard-library-only port of a useful subset of OpenCV's contrib module

[![Go Reference](https://pkg.go.dev/badge/github.com/malcolmston/opencv/linedescriptor.svg)](https://pkg.go.dev/github.com/malcolmston/opencv/linedescriptor)

Package linedescriptor is a from-scratch, standard-library-only port of a
useful subset of OpenCV's contrib module line_descriptor. It is built entirely
on top of the root package github.com/malcolmston/opencv (imported here as cv)
and the Go standard library: it uses no cgo, no third-party code and it does
not import any of the sibling cv/* subpackages.

The module detects straight line segments in an image, describes each segment
with a compact binary code so that segments can be recognised across images,
and matches those codes. Every routine consumes the central `cv.Mat` type
(row-major, channel-interleaved, 8-bit samples), so results compose directly
with the filters, gradients and colour conversions in the root package.

## Install

```sh
go get github.com/malcolmston/opencv@v0.8.0
```

```go
import "github.com/malcolmston/opencv/linedescriptor"
```

## Usage

This is the package's own `ExampleBinaryDescriptorMatcher_Match`, so it compiles and its output is
asserted on every `go test ./linedescriptor/`.

```go
query := [][]byte{
		{0b0000_0000},
		{0b1111_1111},
	}
	train := [][]byte{
		{0b1111_1110},
		{0b0000_0001},
	}
	m := linedescriptor.NewBinaryDescriptorMatcher()
	for _, mt := range m.Match(query, train) {
		fmt.Printf("query %d -> train %d (dist %d)\n", mt.QueryIdx, mt.TrainIdx, mt.Distance)
	}
```

```
query 0 -> train 1 (dist 1)
query 1 -> train 0 (dist 1)
```

## Exported surface

### Functions

| Function | What it does |
| --- | --- |
| `func DrawKeylines(img *cv.Mat, lines []KeyLine, color cv.Scalar, thickness int) *cv.Mat` | DrawKeylines renders lines onto a copy of img and returns the result, echoing cv::line_descriptor::drawKeylines. |
| `func DrawKeylinesEx(img *cv.Mat, lines []KeyLineEx, color cv.Scalar, baseThickness int) *cv.Mat` | DrawKeylinesEx renders multi-octave segments onto a colour copy of img, scaling the drawn thickness by each segment's octave so that segments from… |
| `func DrawLineMatches(img1 *cv.Mat, lines1 []KeyLine, img2 *cv.Mat, lines2 []KeyLine, matches []DMatch, matchColor, singleColor cv.Scalar, thickness int) *cv.Mat` | DrawLineMatches renders two images side by side, draws the detected segments of each and connects matched pairs with a coloured line, mirroring… |
| `func HammingDistance(a, b []byte) int` | HammingDistance returns the number of differing bits between two equal-length bit-packed line descriptors. |
| `func RectMask(rows, cols, y, x, height, width int) *cv.Mat` | RectMask builds a single-channel mask of the given size in which the axis-aligned rectangle [x, x+width) × [y, y+height) is set to 255 and every… |
| `func SegmentOverlap(a, b KeyLine) float64` | SegmentOverlap returns the fraction, in [0, 1], of the shorter segment's projected extent that overlaps the longer segment when both are projected… |

### Types

| Type | What it is |
| --- | --- |
| `BinaryDescriptor` | BinaryDescriptor computes an LBD-style ("Line Band Descriptor") binary code for each line segment. |
| `BinaryDescriptorMatcher` | BinaryDescriptorMatcher is a brute-force matcher over the binary line codes produced by `BinaryDescriptor`. |
| `BinaryDescriptorParams` | BinaryDescriptorParams collects the tunables of the LBD descriptor, mirroring cv::line_descriptor::BinaryDescriptor::Params. |
| `DMatch` | DMatch records a correspondence between a query descriptor and a train descriptor, as produced by a `BinaryDescriptorMatcher`. |
| `EDLinesDetector` | EDLinesDetector detects straight line segments with the Edge Drawing Lines (EDLines) algorithm of Akinlar and Topal (2011), an alternative front end… |
| `GeometricMatchParams` | GeometricMatchParams controls the geometric consistency test applied by `MatchLineSegments` on top of the descriptor (appearance) distance. |
| `KeyLine` | KeyLine is a detected straight line segment, mirroring OpenCV's cv::line_descriptor::KeyLine. |
| `KeyLineEx` | KeyLineEx is a fully-populated line segment mirroring every field of OpenCV's cv::line_descriptor::KeyLine. |
| `LSDDetector` | LSDDetector detects straight line segments with a Line Segment Detector in the style of Grompone von Gioi, Jakubowicz, Morel and Randall (2010). |
| `LSHIndex` | LSHIndex is a multi-index locality-sensitive-hashing structure over binary line descriptors, modelled on the FLANN LSH matcher that the upstream… |
| `PointF` | PointF is a sub-pixel image coordinate (X is the column, Y the row). |
| `ScalePyramid` | ScalePyramid is a Gaussian-style scale pyramid of grayscale images used for multi-octave line detection and description. |

<details>
<summary><code>BinaryDescriptor</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewBinaryDescriptor() *BinaryDescriptor` | NewBinaryDescriptor returns a descriptor with the default geometry of 8 bands of 7 pixels each, yielding a 32-bit (4-byte) code per line. |
| `func NewBinaryDescriptorWithParams(p BinaryDescriptorParams) *BinaryDescriptor` | NewBinaryDescriptorWithParams builds a `BinaryDescriptor` whose band geometry is taken from p. |
| `func (bd *BinaryDescriptor) Compute(img *cv.Mat, lines []KeyLine) ([]KeyLine, [][]byte)` | Compute describes each line in lines using the gradients of img and returns the (unchanged) lines alongside one binary code per line, aligned by… |
| `func (bd *BinaryDescriptor) ComputeMultiOctave(pyr *ScalePyramid, lines []KeyLineEx) ([]KeyLineEx, [][]byte)` | ComputeMultiOctave describes each multi-octave segment using the image data of the octave it was actually detected in, so that a segment found in a… |
| `func (bd *BinaryDescriptor) DescriptorSize() int` | DescriptorSize returns the length in bytes of every code produced by this descriptor. |

</details>

<details>
<summary><code>BinaryDescriptorMatcher</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewBinaryDescriptorMatcher() *BinaryDescriptorMatcher` | NewBinaryDescriptorMatcher returns a ready-to-use matcher. |
| `func (m *BinaryDescriptorMatcher) KnnMatch(query, train [][]byte, k int) [][]DMatch` | KnnMatch returns, for each query descriptor, its k nearest train descriptors as a slice of `DMatch` sorted by ascending distance (ties broken by the… |
| `func (m *BinaryDescriptorMatcher) Match(query, train [][]byte) []DMatch` | Match returns, for each query descriptor, its single closest train descriptor. |
| `func (m *BinaryDescriptorMatcher) RadiusMatch(query, train [][]byte, maxDistance int) [][]DMatch` | RadiusMatch returns, for each query descriptor, every train descriptor within maxDistance Hamming bits of it, as a slice of `DMatch` sorted by… |

</details>

<details>
<summary><code>BinaryDescriptorParams</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func DefaultBinaryDescriptorParams() BinaryDescriptorParams` | DefaultBinaryDescriptorParams returns the upstream default configuration: a single octave, 8 bands of 7 pixels each and a pyramid reduction ratio of… |
| `func (p BinaryDescriptorParams) Validate() error` | Validate reports the first problem with the parameters, or nil if they are usable. |

</details>

<details>
<summary><code>DMatch</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func MatchLineSegments(lines1 []KeyLine, codes1 [][]byte, lines2 []KeyLine, codes2 [][]byte, params GeometricMatchParams) []DMatch` | MatchLineSegments matches two sets of line segments using both appearance and geometry, mirroring the geometric verification stage that upstream line… |

</details>

<details>
<summary><code>EDLinesDetector</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewEDLinesDetector() *EDLinesDetector` | NewEDLinesDetector returns an EDLines detector with defaults matching the common EDLines configuration: a gradient floor of 12, an anchor margin of… |
| `func (e *EDLinesDetector) Detect(img *cv.Mat) []KeyLine` | Detect finds straight line segments in img and returns them as a slice of `KeyLine` sorted by descending `KeyLine.Response` (segment length). |

</details>

<details>
<summary><code>GeometricMatchParams</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func DefaultGeometricMatchParams() GeometricMatchParams` | DefaultGeometricMatchParams returns a lenient-but-useful configuration: a 12-bit Hamming ceiling, a 15° orientation tolerance and a 2× length-ratio… |

</details>

<details>
<summary><code>KeyLine</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func FilterByLength(lines []KeyLine, minLength float64) []KeyLine` | FilterByLength returns the subset of lines whose `KeyLine.Length` is at least minLength, preserving order. |
| `func FilterByResponse(lines []KeyLine, minResponse float64) []KeyLine` | FilterByResponse returns the subset of lines whose `KeyLine.Response` is at least minResponse, preserving order. |
| `func FilterCodesByLength(lines []KeyLine, codes [][]byte, minLength float64) ([]KeyLine, [][]byte)` | FilterCodesByLength keeps the (line, code) pairs whose line is at least minLength long, returning the filtered lines and their aligned codes. |
| `func ToKeyLines(lines []KeyLineEx) []KeyLine` | ToKeyLines extracts the embedded original-image `KeyLine` of every segment, discarding the octave-space extras. |
| `func TopN(lines []KeyLine, n int) []KeyLine` | TopN returns at most the n highest-`KeyLine.Response` segments from lines, sorted by descending response (ties broken by longer length, then by… |

</details>

<details>
<summary><code>KeyLineEx</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func FilterByOctave(lines []KeyLineEx, octave int) []KeyLineEx` | FilterByOctave returns the subset of multi-octave segments detected in the given octave, preserving order. |

</details>

<details>
<summary><code>LSDDetector</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewLSDDetector() *LSDDetector` | NewLSDDetector returns a detector configured with the defaults used by the upstream LSD algorithm: a 22.5° orientation tolerance, the derived… |
| `func (d *LSDDetector) Detect(img *cv.Mat) []KeyLine` | Detect finds straight line segments in img and returns them as a slice of `KeyLine`, sorted by descending `KeyLine.Response` (i.e. |
| `func (d *LSDDetector) DetectPyramid(img *cv.Mat, numOctaves int, scaleFactor float64) []KeyLineEx` | DetectPyramid performs true multi-octave line detection. |
| `func (d *LSDDetector) DetectWithMask(img *cv.Mat, mask *cv.Mat) []KeyLine` | DetectWithMask detects line segments in img but keeps only those whose midpoint falls on a non-zero pixel of mask, mirroring the mask parameter of… |
| `func (d *LSDDetector) DetectWithScale(img *cv.Mat, octave int) []KeyLine` | DetectWithScale runs the detector on a single downscaled octave of img and returns the segments back-projected into original-image coordinates. |

</details>

<details>
<summary><code>LSHIndex</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewLSHIndex(numTables, keyBits int) *LSHIndex` | NewLSHIndex creates an empty multi-index with numTables hash tables, each keyed on keyBits descriptor bits. |
| `func (x *LSHIndex) Add(train [][]byte)` | Add stores train descriptors in the index, assigning them indices starting at the current descriptor count so repeated calls accumulate. |
| `func (x *LSHIndex) KnnMatch(query [][]byte, k int) [][]DMatch` | KnnMatch returns, for each query descriptor, its k nearest indexed descriptors ranked by exact Hamming distance, restricted to the candidates that… |
| `func (x *LSHIndex) RadiusMatch(query [][]byte, maxDistance int) [][]DMatch` | RadiusMatch returns, for each query descriptor, every candidate indexed descriptor within maxDistance Hamming bits, ranked by ascending distance. |
| `func (x *LSHIndex) Size() int` | Size reports how many descriptors are currently stored in the index. |

</details>

<details>
<summary><code>ScalePyramid</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func BuildScalePyramid(img *cv.Mat, numOctaves int, scaleFactor float64) *ScalePyramid` | BuildScalePyramid constructs a numOctaves-level `ScalePyramid` from img by repeatedly shrinking it by scaleFactor. |

</details>

Full signatures, doc comments and every runnable example are on
[pkg.go.dev](https://pkg.go.dev/github.com/malcolmston/opencv/linedescriptor).

## License

MIT, as part of [`github.com/malcolmston/opencv`](..).
An independent re-implementation, not affiliated with or endorsed by the
original project.
