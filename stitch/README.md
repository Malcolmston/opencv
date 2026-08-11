# stitch — implements image stitching and panorama construction on top of the core

[![Go Reference](https://pkg.go.dev/badge/github.com/malcolmston/opencv/stitch.svg)](https://pkg.go.dev/github.com/malcolmston/opencv/stitch)

Package stitch implements image stitching and panorama construction on top of
the core `github.com/malcolmston/opencv` image types, using only the Go
standard library.

- homography.go — floating-point correspondences (`PointF`, `Match`), the
projective `Homography` type, a normalized-DLT estimator and a RANSAC-robust
estimator for recovering the transform between two images from matched feature
points. - features.go — a deterministic Harris corner detector and a
normalized-cross-correlation matcher, so a pair of overlapping images can be
aligned end to end without any external feature library. - warp.go —
cylindrical and spherical projection (`WarpCylindrical`, `WarpSpherical`) and
projective warping into a shared canvas (`WarpPerspectiveToCanvas`). -
canvas.go — integer `Bounds` arithmetic and the geometry needed to place
warped images onto a common mosaic canvas. - align.go — bundle-free
incremental alignment: composing pairwise homographies into a
globally-consistent set of transforms (`GlobalTransforms`,
`IncrementalAligner`). - seam.go — dynamic-programming seam finding
(`FindVerticalSeamDP`, `DPSeamFinder`) that routes the cut between two
overlapping images through the lowest-difference pixels. - exposure.go —
gain-based exposure compensation (`GainCompensator`, `EstimateGains`) that
equalises brightness across the mosaic. - blend.go — feather and multi-band
(Burt–Adelson) blending (`FeatherBlender`, `MultiBandBlender`) plus the
Gaussian/Laplacian pyramid primitives they rely on.

## Install

```sh
go get github.com/malcolmston/opencv@v0.8.0
```

```go
import "github.com/malcolmston/opencv/stitch"
```

## Usage

This is the package's own `ExampleEstimateHomographyDLT`, so it compiles and its output is
asserted on every `go test ./stitch/`.

```go
want := stitch.TranslationHomography(5, 3)
	var matches []stitch.Match
	for _, p := range []stitch.PointF{{X: 0, Y: 0}, {X: 10, Y: 0}, {X: 0, Y: 10}, {X: 10, Y: 10}} {
		matches = append(matches, stitch.Match{Src: p, Dst: want.Apply(p)})
	}

	h, ok := stitch.EstimateHomographyDLT(matches)
	if !ok {
		fmt.Println("estimation failed")
		return
	}
	x, y := h.ApplyXY(4, 4)
	fmt.Printf("ok=%v (4,4) -> (%.2f,%.2f)\n", ok, x, y)
	fmt.Printf("mean reprojection error: %.6f\n", stitch.MeanReprojectionError(h, matches))
```

```
ok=true (4,4) -> (9.00,7.00)
mean reprojection error: 0.000000
```

## Exported surface

### Functions

| Function | What it does |
| --- | --- |
| `func ApplyGain(img *cv.Mat, gain float64) *cv.Mat` | ApplyGain returns a copy of img with every sample multiplied by gain and clamped to the valid 8-bit range. |
| `func BuildGaussianPyramid(img *cv.FloatMat, levels int) []*cv.FloatMat` | BuildGaussianPyramid returns the Gaussian pyramid of img with levels reduce steps: the result has length levels+1, element 0 being img itself and… |
| `func BuildLaplacianPyramid(img *cv.FloatMat, levels int) []*cv.FloatMat` | BuildLaplacianPyramid returns the Laplacian pyramid of img with levels bands. |
| `func CollapseLaplacianPyramid(pyr []*cv.FloatMat) *cv.FloatMat` | CollapseLaplacianPyramid reconstructs the image from its Laplacian pyramid, the inverse of `BuildLaplacianPyramid`. |
| `func EstimateGains(means, counts [][]float64, sigmaN, sigmaG float64) []float64` | EstimateGains solves for the per-image exposure gains that best equalise brightness across a mosaic, following Brown and Lowe's gain-compensation… |
| `func FeatherWeightMap(width, height int, sharpness float64) *cv.FloatMat` | FeatherWeightMap returns a width×height feather weight map that is largest at the image centre and tapers toward the border, so overlapping images… |
| `func FindHorizontalSeamDP(cost *cv.FloatMat) []int` | FindHorizontalSeamDP finds the minimum-cost left-to-right seam through the cost map. |
| `func FindVerticalSeamDP(cost *cv.FloatMat) []int` | FindVerticalSeamDP finds the minimum-cost top-to-bottom seam through the cost map by dynamic programming. |
| `func HarrisResponse(img *cv.Mat, window int, k float64) *cv.FloatMat` | HarrisResponse computes the Harris corner response map of img over a (2*window+1)² aggregation neighbourhood, using the sensitivity parameter k… |
| `func MeanIntensityMasked(img *cv.Mat, weight *cv.FloatMat) (mean float64, count int)` | MeanIntensityMasked returns the mean grayscale intensity of img over the pixels whose weight is strictly positive, together with the number of such… |
| `func MeanReprojectionError(h Homography, matches []Match) float64` | MeanReprojectionError returns the average `ReprojectionError` over matches. |
| `func NormalizedCrossCorrelation(a, b *cv.Mat, ax, ay, bx, by, half int) (float64, bool)` | NormalizedCrossCorrelation returns the zero-mean normalized cross-correlation of the grayscale patches of half-size half centred at (ax, ay) in a and… |
| `func PlaceImage(dst, src *cv.Mat, offsetX, offsetY int)` | PlaceImage copies src into dst so that the top-left corner of src lands at (offsetX, offsetY) in dst. |
| `func ProjectCylindrical(x, y, focal, cx, cy float64) (u, v float64)` | ProjectCylindrical maps an image point (x, y) onto a cylinder of radius focal whose axis passes through the projection centre (cx, cy). |
| `func ProjectSpherical(x, y, focal, cx, cy float64) (u, v float64)` | ProjectSpherical maps an image point (x, y) onto a unit sphere of radius focal centred on (cx, cy). |
| `func PyrDownFloat(src *cv.FloatMat) *cv.FloatMat` | PyrDownFloat blurs src with a 5-tap Gaussian and subsamples it by two, halving each dimension (rounding up). |
| `func PyrUpFloat(src *cv.FloatMat, dstRows, dstCols int) *cv.FloatMat` | PyrUpFloat upsamples src to the given target dimensions by zero-insertion followed by a 5-tap Gaussian smoothing (scaled to preserve brightness). |
| `func ReprojectionError(h Homography, m Match) float64` | ReprojectionError returns the Euclidean distance in the destination image between the mapped source point of m and its observed destination point. |
| `func SeamCostMap(a, b *cv.Mat) *cv.FloatMat` | SeamCostMap returns the per-pixel disagreement cost between two same-sized images, the energy a seam finder routes around. |
| `func SeamMaskFromColumns(seam []int, rows, cols int) *cv.Mat` | SeamMaskFromColumns builds a single-channel mask of size rows×cols from a vertical seam (one column per row, as returned by `FindVerticalSeamDP`). |
| `func WarpCylindrical(img *cv.Mat, focal float64) (out *cv.Mat, offsetX, offsetY int)` | WarpCylindrical projects img onto a cylinder of radius focal (the camera focal length in pixels) and returns the rectified image together with the… |
| `func WarpSpherical(img *cv.Mat, focal float64) (out *cv.Mat, offsetX, offsetY int)` | WarpSpherical projects img onto a sphere of radius focal (the camera focal length in pixels) and returns the rectified image together with the… |

### Types

| Type | What it is |
| --- | --- |
| `Blender` | Blender combines the overlapping layers of a mosaic into one image. |
| `Bounds` | Bounds is an axis-aligned integer rectangle on the mosaic canvas. |
| `DPSeamFinder` | DPSeamFinder routes the boundary between two overlapping images through the pixels where they agree most, using dynamic-programming seam finding on… |
| `FeatherBlender` | FeatherBlender averages the layers pixel by pixel, weighting each contribution by its `Layer`.Weight. |
| `GainCompensator` | GainCompensator estimates and applies per-image exposure gains so that overlapping images share a consistent brightness. |
| `Homography` | Homography is a 3×3 projective transform in row-major order. |
| `IncrementalAligner` | IncrementalAligner builds a globally-consistent chain of transforms one image at a time, without a global bundle adjustment. |
| `Layer` | Layer is one source image warped onto the mosaic canvas together with a per-pixel blending weight over the same canvas. |
| `Match` | Match is a single point correspondence between two images: Src is the point in the source image and Dst the matching point in the destination image. |
| `MultiBandBlender` | MultiBandBlender implements Burt–Adelson multi-band blending: each layer is decomposed into a Laplacian pyramid, the bands are merged using… |
| `PointF` | PointF is a floating-point image coordinate (x is the column, y is the row). |

<details>
<summary><code>Bounds</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func UnionBounds(bs []Bounds) Bounds` | UnionBounds returns the smallest rectangle containing every rectangle in bs. |
| `func WarpedBounds(width, height int, h Homography) Bounds` | WarpedBounds returns the integer bounding box that a width×height image occupies after being transformed by h. |
| `func (b Bounds) Contains(x, y int) bool` | Contains reports whether the pixel (x, y) lies inside the rectangle. |
| `func (b Bounds) Empty() bool` | Empty reports whether the rectangle encloses no pixels. |
| `func (b Bounds) Height() int` | Height returns the height of the rectangle in pixels, or 0 if empty. |
| `func (b Bounds) Intersect(other Bounds) Bounds` | Intersect returns the overlap of b and other, which is empty if they do not overlap. |
| `func (b Bounds) Union(other Bounds) Bounds` | Union returns the smallest rectangle that contains both b and other. |
| `func (b Bounds) Width() int` | Width returns the width of the rectangle in pixels, or 0 if empty. |

</details>

<details>
<summary><code>DPSeamFinder</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func (f DPSeamFinder) Find(a, b *cv.Mat) *cv.Mat` | Find computes the optimal seam between two same-sized, same-channel images a and b and returns a single-channel mask: 255 where image a should be… |

</details>

<details>
<summary><code>FeatherBlender</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewFeatherBlender() *FeatherBlender` | NewFeatherBlender returns a ready-to-use feather blender. |
| `func (FeatherBlender) Blend(layers []Layer) (*cv.Mat, error)` | Blend implements `Blender` with per-pixel weighted averaging. |

</details>

<details>
<summary><code>GainCompensator</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewGainCompensator() *GainCompensator` | NewGainCompensator returns a compensator with default noise and gain-prior parameters. |
| `func (g *GainCompensator) Apply(index int, layer Layer) Layer` | Apply multiplies the colour of layer index by its estimated gain and returns the corrected layer (sharing the original weight map). |
| `func (g *GainCompensator) Feed(layers []Layer) []float64` | Feed measures the pairwise overlap statistics of the layers and solves for the exposure gains, storing them for `GainCompensator.Gains` and… |
| `func (g *GainCompensator) Gains() []float64` | Gains returns the exposure gains computed by the most recent `GainCompensator.Feed`, one per layer, or nil if Feed has not been called. |

</details>

<details>
<summary><code>Homography</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func ChainHomographies(hs []Homography) Homography` | ChainHomographies composes a sequence of pairwise homographies into a single transform. |
| `func EstimateHomographyDLT(matches []Match) (Homography, bool)` | EstimateHomographyDLT estimates the homography that best maps the source points to the destination points of matches, in the total least-squares… |
| `func EstimateHomographyRANSAC(matches []Match, threshold float64, iterations int, seed int64) (Homography, []int, bool)` | EstimateHomographyRANSAC robustly estimates the homography between the source and destination points of matches in the presence of outliers. |
| `func GlobalTransforms(pairwise []Homography, reference int) []Homography` | GlobalTransforms turns a chain of pairwise homographies into a globally-consistent set of transforms into a common reference frame, the bundle-free… |
| `func IdentityHomography() Homography` | IdentityHomography returns the identity transform, which maps every point to itself. |
| `func ScaleHomography(sx, sy float64) Homography` | ScaleHomography returns a homography that scales the x axis by sx and the y axis by sy about the origin. |
| `func TranslationHomography(tx, ty float64) Homography` | TranslationHomography returns a homography that shifts points by (tx, ty). |
| `func (h Homography) Apply(p PointF) PointF` | Apply transforms the point p and returns the mapped point. |
| `func (h Homography) ApplyXY(x, y float64) (float64, float64)` | ApplyXY transforms the point (x, y) and returns the mapped coordinates. |
| `func (h Homography) Det() float64` | Det returns the determinant of the 3×3 matrix. |
| `func (h Homography) Inverse() (Homography, bool)` | Inverse returns the inverse transform and true, or the zero homography and false if the matrix is singular. |
| `func (h Homography) Mul(other Homography) Homography` | Mul returns the composition h·other, whose action first applies other and then h. |
| `func (h Homography) Normalize() Homography` | Normalize scales the matrix so its bottom-right entry equals 1, giving a canonical representation of the same transform. |
| `func (h Homography) TransferPoints(pts []PointF) []PointF` | TransferPoints applies the homography to every point in pts and returns the mapped points in a new slice. |

</details>

<details>
<summary><code>IncrementalAligner</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewIncrementalAligner() *IncrementalAligner` | NewIncrementalAligner returns an empty aligner. |
| `func (a *IncrementalAligner) Add(relToPrev Homography) Homography` | Add registers the next image. |
| `func (a *IncrementalAligner) Count() int` | Count returns the number of images registered so far. |
| `func (a *IncrementalAligner) Transform(i int) (Homography, bool)` | Transform returns the global transform of image i (into the reference frame) and true, or the identity and false if i is out of range. |
| `func (a *IncrementalAligner) Transforms() []Homography` | Transforms returns a copy of every global transform accumulated so far, indexed by the order images were added. |

</details>

<details>
<summary><code>Layer</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func WarpPerspectiveToCanvas(img *cv.Mat, h Homography, canvas Bounds) Layer` | WarpPerspectiveToCanvas warps img by the homography h onto the mosaic region described by canvas, producing a `Layer` sized to that region. |

</details>

<details>
<summary><code>Match</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func MatchCornersNCC(a, b *cv.Mat, cornersA, cornersB []PointF, half, searchRadius int, minCorrelation float64) []Match` | MatchCornersNCC matches corners between images a and b by normalized cross-correlation of local patches. |

</details>

<details>
<summary><code>MultiBandBlender</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewMultiBandBlender(numBands int) *MultiBandBlender` | NewMultiBandBlender returns a multi-band blender using numBands pyramid levels. |
| `func (b MultiBandBlender) Blend(layers []Layer) (*cv.Mat, error)` | Blend implements `Blender` with multi-band (pyramid) blending. |

</details>

<details>
<summary><code>PointF</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func HarrisCorners(img *cv.Mat, maxCorners int, quality float64, minDistance int) []PointF` | HarrisCorners detects up to maxCorners strong Harris corners in img. |
| `func (p PointF) Distance(q PointF) float64` | Distance returns the Euclidean distance between p and q. |
| `func (p PointF) Sub(q PointF) PointF` | Sub returns the vector from q to p (p minus q). |

</details>

Full signatures, doc comments and every runnable example are on
[pkg.go.dev](https://pkg.go.dev/github.com/malcolmston/opencv/stitch).

## License

MIT, as part of [`github.com/malcolmston/opencv`](..).
An independent re-implementation, not affiliated with or endorsed by the
original project.
