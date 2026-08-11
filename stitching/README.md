# stitching — assembles overlapping images into a single panorama on top of the standard-library-only

[![Go Reference](https://pkg.go.dev/badge/github.com/malcolmston/opencv/stitching.svg)](https://pkg.go.dev/github.com/malcolmston/opencv/stitching)

Package stitching assembles overlapping images into a single panorama on top
of the standard-library-only OpenCV port github.com/malcolmston/opencv
(imported here as cv). It mirrors a useful subset of OpenCV's stitching module
with a self-contained, deterministic implementation that depends only on the
root cv package and the Go standard library.

1. Detection. Shi–Tomasi corners are found in each image with
cv.GoodFeaturesToTrack. 2. Description. Every corner is summarised by a
normalised intensity-patch descriptor — a mean-subtracted, L2-normalised
window of pixels — so that descriptor distance is invariant to local
brightness and contrast. 3. Matching. Descriptors of adjacent images are
matched by nearest neighbour and filtered with Lowe's ratio test, which
rejects ambiguous matches in repetitive texture. 4. Estimation. The pairwise
homography is fitted with a normalised Direct Linear Transform inside a RANSAC
loop, giving a model robust to the outliers that survive matching. See
`Stitcher.EstimateTransform`. 5. Composition. Neighbour homographies are
chained into a common reference frame, every image is warped into a shared
canvas with cv.WarpPerspective, and the overlaps are blended. See
`Stitcher.ComposePanorama`.

## Install

```sh
go get github.com/malcolmston/opencv@v0.8.0
```

```go
import "github.com/malcolmston/opencv/stitching"
```

## Usage

This is the package's own `ExampleCylindricalWarper`, so it compiles and its output is
asserted on every `go test ./stitching/`.

```go
src := cv.NewMat(60, 80, 1)
	for i := range src.Data {
		src.Data[i] = uint8(i % 200)
	}
	w := stitching.CylindricalWarper{}
	warped, corner := w.Warp(src, 100)
	back := w.WarpBackward(warped, 100, corner, src.Cols, src.Rows)
	fmt.Printf("warped %dx%d, restored %dx%d\n", warped.Cols, warped.Rows, back.Cols, back.Rows)
```

```
warped 77x61, restored 80x60
```

## Exported surface

### Functions

| Function | What it does |
| --- | --- |
| `func EstimateFocalsFromHomography(h cv.PerspectiveMatrix) (f0, f1 float64, ok0, ok1 bool)` | EstimateFocalsFromHomography recovers the focal lengths of the two cameras related by the homography h (which must be normalised so h8 = 1),… |
| `func WaveCorrect(cams []CameraParams, kind WaveCorrectKind)` | WaveCorrect straightens the global "wave" — the slow up-and-down (or side-to-side) drift of the horizon — that remains in a set of camera… |

### Types

| Type | What it is |
| --- | --- |
| `Blender` | Blender combines the overlapping `Layer`s of a panorama into a single image of the given size. |
| `BlocksGainCompensator` | BlocksGainCompensator estimates a spatially-varying gain field per image by dividing every image into a grid of blocks and solving for one gain per… |
| `BundleAdjuster` | BundleAdjuster globally refines the `CameraParams` produced by an `Estimator`. |
| `BundleAdjusterRay` | BundleAdjusterRay refines the cameras by minimising the discrepancy between the back-projected viewing rays of matched points. |
| `BundleAdjusterReproj` | BundleAdjusterReproj refines the cameras by minimising the reprojection error, in pixels, of matched points projected from each image into the other. |
| `CameraParams` | CameraParams holds the intrinsic and extrinsic parameters recovered for one image of a rotational panorama: the focal length and principal point (the… |
| `CylindricalWarper` | CylindricalWarper projects onto a vertical cylinder. |
| `DpSeamFinder` | DpSeamFinder places a minimum-cost seam through each overlap with dynamic programming: the seam follows the path of least accumulated colour… |
| `Estimator` | Estimator recovers per-image `CameraParams` (focal lengths and rotations) from the pairwise correspondences of a panorama. |
| `ExposureCompensator` | ExposureCompensator removes the brightness and colour differences between the overlapping images of a panorama that arise from auto-exposure and… |
| `Feather` | Feather is a distance-weighted (alpha) blender. |
| `GainCompensator` | GainCompensator estimates one multiplicative gain per image by minimising the intensity mismatch between overlapping images, following Brown and… |
| `GraphCutSeamFinder` | GraphCutSeamFinder finds the globally optimal seam through each overlap as the minimum cut of a graph whose edge weights are the colour difference… |
| `HomographyBasedEstimator` | HomographyBasedEstimator recovers focal lengths from the inter-image homographies (via `EstimateFocalsFromHomography`) and chains the pairwise… |
| `Layer` | Layer is one image warped into the panorama canvas, ready for blending. |
| `MatchesInfo` | MatchesInfo carries the inlier point correspondences between two images, used by the `Estimator` and `BundleAdjuster` to relate the views. |
| `MultiBandBlend` | MultiBandBlend is a Laplacian-pyramid (multi-band) blender. |
| `NoExposureCompensator` | NoExposureCompensator is the identity `ExposureCompensator`; it learns nothing and applies no correction. |
| `NoSeamFinder` | NoSeamFinder leaves the masks untouched, so overlaps are resolved entirely by the blender. |
| `Pipeline` | Pipeline is a configurable, higher-level panorama builder that wires together the surface `Warper`, `ExposureCompensator`, `SeamFinder` and `Blender`… |
| `PlaneWarper` | PlaneWarper is the trivial `Warper` whose surface is the image plane; Warp is an identity resample. |
| `RANSACParams` | RANSACParams configures the random-sample-consensus homography estimator used by `Stitcher.EstimateTransform`. |
| `SeamFinder` | SeamFinder decides, in the regions where two panorama images overlap, which image should supply each pixel. |
| `SphericalWarper` | SphericalWarper projects onto a sphere. |
| `StitchMode` | StitchMode selects the geometric model a `Pipeline` assumes for the scene. |
| `Stitcher` | Stitcher builds a panorama from overlapping images. |
| `Timelapser` | Timelapser composites the warped panorama images onto a single fixed-size canvas, one at a time, so that a sequence of frames can be saved as a… |
| `VoronoiSeamFinder` | VoronoiSeamFinder assigns each overlap pixel to the image whose centre is nearest, producing a straight Voronoi boundary. |
| `Warper` | Warper projects a source image onto a curved surface (a cylinder or sphere) or leaves it planar, parameterised by a focal length in pixels. |
| `WaveCorrectKind` | WaveCorrectKind selects the axis a `WaveCorrect` call straightens. |

<details>
<summary><code>BlocksGainCompensator</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func (bc *BlocksGainCompensator) Apply(index int, _ image.Point, img *cv.Mat)` | Apply multiplies each pixel of the image by its gain, bilinearly interpolated from the surrounding block-centre gains. |
| `func (bc *BlocksGainCompensator) Feed(corners []image.Point, images []*cv.Mat, masks []*cv.FloatMat)` | Feed solves the block-level system and stores a gain grid per image. |

</details>

<details>
<summary><code>BundleAdjusterRay</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func (ba BundleAdjusterRay) Refine(cameras []CameraParams, matches []MatchesInfo) ([]CameraParams, bool)` | Refine implements `BundleAdjuster` using the ray-divergence cost. |

</details>

<details>
<summary><code>BundleAdjusterReproj</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func (ba BundleAdjusterReproj) Refine(cameras []CameraParams, matches []MatchesInfo) ([]CameraParams, bool)` | Refine implements `BundleAdjuster` using the reprojection-error cost. |

</details>

<details>
<summary><code>CameraParams</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func (c CameraParams) K() cv.PerspectiveMatrix` | K returns the 3×3 intrinsic matrix K = [f 0 ppx; 0 f·aspect ppy; 0 0 1]. |

</details>

<details>
<summary><code>CylindricalWarper</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func (CylindricalWarper) Name() string` | Name returns "cylindrical". |
| `func (CylindricalWarper) Warp(src *cv.Mat, focal float64) (*cv.Mat, image.Point)` | Warp projects src onto the cylinder. |
| `func (CylindricalWarper) WarpBackward(warped *cv.Mat, focal float64, corner image.Point, width, height int) *cv.Mat` | WarpBackward reconstructs the planar image from a cylindrical projection. |
| `func (CylindricalWarper) WarpPoint(x, y, focal, cx, cy float64) (float64, float64)` | WarpPoint maps a source point onto the cylinder. |

</details>

<details>
<summary><code>DpSeamFinder</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func (dp *DpSeamFinder) Find(images []*cv.Mat, corners []image.Point, masks []*cv.FloatMat)` | Find routes a minimum-cost seam through every overlap. |
| `func (dp *DpSeamFinder) SeamCost() float64` | SeamCost returns the accumulated colour-difference cost of the seam found by the most recent Find call; a low value means the images agree along the… |

</details>

<details>
<summary><code>Feather</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func (Feather) Blend(layers []Layer, rows, cols, channels int) (*cv.Mat, error)` | Blend implements `Blender` with per-pixel weighted averaging. |

</details>

<details>
<summary><code>GainCompensator</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func (gc *GainCompensator) Apply(index int, _ image.Point, img *cv.Mat)` | Apply scales the image by its learned gain. |
| `func (gc *GainCompensator) Feed(corners []image.Point, images []*cv.Mat, masks []*cv.FloatMat)` | Feed solves the linear system that balances the overlap intensities and stores one gain per image. |
| `func (gc *GainCompensator) Gains() []float64` | Gains returns the per-image gains learned by the most recent Feed. |

</details>

<details>
<summary><code>GraphCutSeamFinder</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func (gc *GraphCutSeamFinder) CutCost() float64` | CutCost returns the total edge weight severed by the most recent Find; a low value means the seam runs where the images agree. |
| `func (gc *GraphCutSeamFinder) Find(images []*cv.Mat, corners []image.Point, masks []*cv.FloatMat)` | Find computes a minimum-cut seam through every overlap. |

</details>

<details>
<summary><code>HomographyBasedEstimator</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func (e HomographyBasedEstimator) Estimate(sizes []image.Point, matches []MatchesInfo) ([]CameraParams, bool)` | Estimate implements `Estimator`. |

</details>

<details>
<summary><code>MultiBandBlend</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func (mb MultiBandBlend) Blend(layers []Layer, rows, cols, channels int) (*cv.Mat, error)` | Blend implements `Blender` with multi-band pyramid blending. |

</details>

<details>
<summary><code>NoExposureCompensator</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func (NoExposureCompensator) Apply(int, image.Point, *cv.Mat)` | Apply does nothing. |
| `func (NoExposureCompensator) Feed([]image.Point, []*cv.Mat, []*cv.FloatMat)` | Feed does nothing. |

</details>

<details>
<summary><code>NoSeamFinder</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func (NoSeamFinder) Find([]*cv.Mat, []image.Point, []*cv.FloatMat)` | Find does nothing. |

</details>

<details>
<summary><code>Pipeline</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewPipeline(mode StitchMode) *Pipeline` | NewPipeline returns a Pipeline for the given mode with default stages: no exposure compensation, no seam finding, feather blending, and (for… |
| `func (p *Pipeline) Mode() StitchMode` | Mode returns the pipeline's geometric mode. |
| `func (p *Pipeline) SetBlender(b Blender)` | SetBlender selects the blender that merges the warped, compensated, seam-cut images into the final panorama. |
| `func (p *Pipeline) SetExposureCompensator(e ExposureCompensator)` | SetExposureCompensator selects the exposure compensator applied to the warped images before blending. |
| `func (p *Pipeline) SetSeamFinder(s SeamFinder)` | SetSeamFinder selects the seam finder used to partition overlaps before blending. |
| `func (p *Pipeline) SetWarper(w Warper)` | SetWarper selects the surface warper. |
| `func (p *Pipeline) Stitch(images []*cv.Mat) (*cv.Mat, error)` | Stitch assembles images into a panorama, running the full configured pipeline: optional surface warping, pairwise transform estimation and chaining,… |
| `func (p *Pipeline) Stitcher() *Stitcher` | Stitcher returns the underlying feature-based `Stitcher`, whose detection, matching and RANSAC parameters may be tuned directly. |

</details>

<details>
<summary><code>PlaneWarper</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func (PlaneWarper) Name() string` | Name returns "plane". |
| `func (PlaneWarper) Warp(src *cv.Mat, _ float64) (*cv.Mat, image.Point)` | Warp returns a clone of src with a zero corner; the plane projection is the identity. |
| `func (PlaneWarper) WarpBackward(warped *cv.Mat, focal float64, corner image.Point, width, height int) *cv.Mat` | WarpBackward reconstructs a planar image from a plane-warped one. |
| `func (PlaneWarper) WarpPoint(x, y, focal, cx, cy float64) (float64, float64)` | WarpPoint maps a source point through the identity projection. |

</details>

<details>
<summary><code>SphericalWarper</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func (SphericalWarper) Name() string` | Name returns "spherical". |
| `func (SphericalWarper) Warp(src *cv.Mat, focal float64) (*cv.Mat, image.Point)` | Warp projects src onto the sphere. |
| `func (SphericalWarper) WarpBackward(warped *cv.Mat, focal float64, corner image.Point, width, height int) *cv.Mat` | WarpBackward reconstructs the planar image from a spherical projection. |
| `func (SphericalWarper) WarpPoint(x, y, focal, cx, cy float64) (float64, float64)` | WarpPoint maps a source point onto the sphere. |

</details>

<details>
<summary><code>Stitcher</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewStitcher() *Stitcher` | NewStitcher returns a Stitcher preconfigured with robust defaults for small to medium overlapping images and a feather blender. |
| `func (s *Stitcher) ComposePanorama(images []*cv.Mat, transforms []cv.PerspectiveMatrix) (*cv.Mat, error)` | ComposePanorama warps each image by its transform into a common canvas and blends the results. |
| `func (s *Stitcher) EstimateTransform(imgA, imgB *cv.Mat) (cv.PerspectiveMatrix, error)` | EstimateTransform estimates the homography that maps the coordinate system of imgB into that of imgA, so that warping imgB by the result overlays it… |
| `func (s *Stitcher) Stitch(images []*cv.Mat) (*cv.Mat, error)` | Stitch assembles the images into a single panorama. |

</details>

<details>
<summary><code>Timelapser</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewTimelapser(rows, cols, channels int) *Timelapser` | NewTimelapser allocates a Timelapser with a blank rows×cols canvas of the given channel count, whose top-left corner is the coordinate origin. |
| `func NewTimelapserForCorners(corners, sizes []image.Point, channels int) *Timelapser` | NewTimelapserForCorners allocates a Timelapser whose canvas is exactly the bounding box of the given image corners and sizes (sizes`i` is (X:width,… |
| `func (t *Timelapser) Canvas() *cv.Mat` | Canvas returns the shared accumulation canvas. |
| `func (t *Timelapser) Frame(img *cv.Mat, mask *cv.FloatMat, corner image.Point) *cv.Mat` | Frame returns a fresh canvas-sized image containing only img, drawn at the given global corner, with every other pixel left blank. |
| `func (t *Timelapser) Process(img *cv.Mat, mask *cv.FloatMat, corner image.Point)` | Process draws img onto the shared canvas at the given global corner. |
| `func (t *Timelapser) Size() (rows, cols int)` | Size returns the canvas dimensions as (rows, cols). |

</details>

<details>
<summary><code>VoronoiSeamFinder</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func (VoronoiSeamFinder) Find(images []*cv.Mat, corners []image.Point, masks []*cv.FloatMat)` | Find partitions overlaps by nearest image centre. |

</details>

### Variables

`ErrNoImages`, `ErrChannelMismatch`, `ErrNotEnoughMatches`, `ErrHomography`, `ErrCanvasTooLarge`

Full signatures, doc comments and every runnable example are on
[pkg.go.dev](https://pkg.go.dev/github.com/malcolmston/opencv/stitching).

## License

MIT, as part of [`github.com/malcolmston/opencv`](..).
An independent re-implementation, not affiliated with or endorsed by the
original project.
