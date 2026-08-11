# textdet — implements classical (non-neural) text-detection and document-analysis primitives

[![Go Reference](https://pkg.go.dev/badge/github.com/malcolmston/opencv/textdet.svg)](https://pkg.go.dev/github.com/malcolmston/opencv/textdet)

Package textdet implements classical (non-neural) text-detection and
document-analysis primitives on top of the parent module's `cv.Mat` image
type.

The package is a pure-Go, standard-library-only companion to the core opencv
module. It never defines its own image type: every routine that produces an
image returns a `cv.Mat`, every routine that consumes one accepts a `cv.Mat`,
and geometry is expressed with the parent package's `cv.Rect` and `cv.Point`,
so results interoperate directly with the rest of the library (drawing,
morphology, thresholding, I/O and so on).

## Install

```sh
go get github.com/malcolmston/opencv@v0.8.0
```

```go
import "github.com/malcolmston/opencv/textdet"
```

## Usage

This is the package's own `ExampleEstimateSkew`, so it compiles and its output is
asserted on every `go test ./textdet/`.

```go
upright := page()

	angle, err := textdet.EstimateSkew(upright, 10, 1)
	if err != nil {
		panic(err)
	}
	fmt.Printf("upright page skew: %.0f degrees\n", angle)

	tilted, err := textdet.RotateImage(upright, 5)
	if err != nil {
		panic(err)
	}
	got, err := textdet.EstimateSkew(tilted, 10, 0.5)
	if err != nil {
		panic(err)
	}
	fmt.Printf("recovered a non-zero skew: %v\n", got != 0)
```

```
upright page skew: 0 degrees
recovered a non-zero skew: true
```

## Exported surface

### Functions

| Function | What it does |
| --- | --- |
| `func AdaptiveMean(src *cv.Mat, radius int, c float64, p Polarity) (*cv.Mat, error)` | AdaptiveMean binarizes src with a local mean threshold: a pixel is ink when its value differs from the mean of its (2*radius+1)-square neighbourhood… |
| `func Bernsen(src *cv.Mat, radius int, contrastThresh float64, p Polarity) (*cv.Mat, error)` | Bernsen binarizes src with Bernsen's local method. |
| `func Binarize(src *cv.Mat, thresh int, p Polarity) (*cv.Mat, error)` | Binarize thresholds src at a fixed grey level and returns a single-channel 0/255 mask in which ink pixels (selected by p) are 255. |
| `func CorrectSkew(binary *cv.Mat, maxAngle, step float64) (*cv.Mat, float64, error)` | CorrectSkew estimates the skew of a binary text image with `EstimateSkew` and returns a deskewed copy rotated to bring the text back to horizontal,… |
| `func EdgeDensityMap(src *cv.Mat, radius int, magThresh float64) (*cv.FloatMat, error)` | EdgeDensityMap measures local edge density: it computes the gradient magnitude of src, marks strong edges (magnitude above magThresh), and for every… |
| `func EstimateSkew(binary *cv.Mat, maxAngle, step float64) (float64, error)` | EstimateSkew estimates the skew angle of a binary text image in degrees by maximising the variance of its horizontal projection over candidate angles. |
| `func ForegroundRatio(src *cv.Mat, thresh int, p Polarity) (float64, error)` | ForegroundRatio reports the fraction of ink pixels (selected by p) that a fixed-level threshold at thresh would produce, a value in [0,1]. |
| `func Gray(src *cv.Mat) (*cv.Mat, error)` | Gray returns a fresh single-channel `cv.Mat` holding the luma of src. |
| `func HorizontalProjection(binary *cv.Mat) ([]int, error)` | HorizontalProjection returns the per-row foreground counts of a binary single-channel image (any non-zero sample counts as foreground). |
| `func LocalizeByEdgeDensity(src *cv.Mat, radius int, magThresh, densityThresh float64, minArea int) ([]cv.Rect, error)` | LocalizeByEdgeDensity localizes candidate text regions by edge density. |
| `func Niblack(src *cv.Mat, radius int, k float64, p Polarity) (*cv.Mat, error)` | Niblack binarizes src with Niblack's local method. |
| `func Otsu(src *cv.Mat, p Polarity) (*cv.Mat, int, error)` | Otsu binarizes src with a global Otsu threshold and returns the 0/255 ink mask together with the grey level that was used. |
| `func OtsuThreshold(src *cv.Mat) (int, error)` | OtsuThreshold computes the global Otsu threshold of src and returns the grey level in [0,255]. |
| `func ProjectionVariance(binary *cv.Mat, angle float64) (float64, error)` | ProjectionVariance returns the variance of the horizontal projection of a binary image rotated by angle degrees about its centre. |
| `func RegionsToMask(regions []MSERRegion, rows, cols int) (*cv.Mat, error)` | RegionsToMask renders a set of MSER regions into a single fresh single-channel 0/255 `cv.Mat` of the given size, painting every region pixel 255. |
| `func RotateImage(src *cv.Mat, angle float64) (*cv.Mat, error)` | RotateImage rotates src about its centre by angle degrees (counter-clockwise) using bilinear interpolation, keeping the original dimensions. |
| `func Sauvola(src *cv.Mat, radius int, k, r float64, p Polarity) (*cv.Mat, error)` | Sauvola binarizes src with Sauvola's local method, an improvement on Niblack for document images. |
| `func SegmentCharacters(lineImage *cv.Mat, colThreshold, minWidth int) ([]cv.Rect, error)` | SegmentCharacters locates individual characters within a single text-line image by thresholding its vertical projection with no gap merging, so every… |
| `func SegmentTextLines(binary *cv.Mat, rowThreshold, minGap, minHeight int) ([]cv.Rect, error)` | SegmentTextLines locates text lines in a binary page image by thresholding its horizontal projection. |
| `func SegmentWords(lineImage *cv.Mat, colThreshold, minGap, minWidth int) ([]cv.Rect, error)` | SegmentWords locates words within a single text-line image by thresholding its vertical projection. |
| `func SmoothProfile(prof []int, radius int) ([]float64, error)` | SmoothProfile returns a moving-average smoothing of prof with a window of (2*radius+1) samples and replicated borders. |
| `func VerticalProjection(binary *cv.Mat) ([]int, error)` | VerticalProjection returns the per-column foreground counts of a binary single-channel image. |
| `func Wolf(src *cv.Mat, radius int, k float64, p Polarity) (*cv.Mat, error)` | Wolf binarizes src with the Wolf-Jolion local method, which normalises Niblack by the global minimum grey level and the maximum local standard… |

### Types

| Type | What it is |
| --- | --- |
| `Band` | Band is a contiguous run of indices in a projection profile whose values exceed the segmentation threshold, such as a text line (horizontal profile)… |
| `Component` | Component describes one connected foreground region: its label, pixel area, axis-aligned bounding box and centroid. |
| `ComponentSet` | ComponentSet is the result of connected-component labelling: the dense label image together with per-component statistics. |
| `Connectivity` | Connectivity selects the pixel neighbourhood used when grouping foreground pixels into connected components. |
| `IntegralImage` | IntegralImage is a summed-area representation of a single-channel image that answers rectangle-sum, mean and standard-deviation queries in O(1). |
| `MSEROptions` | MSEROptions configures `DetectMSER`. |
| `MSERPolarity` | MSERPolarity selects whether Maximally Stable Extremal Regions are grown as dark regions on a light background (MSER+) or bright regions on a dark… |
| `MSERRegion` | MSERRegion is one maximally stable extremal region. |
| `Polarity` | Polarity selects which side of a threshold is treated as ink (foreground, rendered 255) by the binarizers in this file. |
| `SWTLetter` | SWTLetter is a candidate glyph found by grouping neighbouring pixels of similar stroke width in an `SWTResult`. |
| `SWTOptions` | SWTOptions configures `StrokeWidthTransform`. |
| `SWTPolarity` | SWTPolarity selects whether the Stroke-Width Transform seeks dark strokes on a light background or the reverse, which fixes the direction in which… |
| `SWTResult` | SWTResult holds the per-pixel stroke widths produced by `StrokeWidthTransform`. |
| `TextComponentOptions` | TextComponentOptions bounds the geometric properties a connected component must satisfy to be considered a text-glyph candidate by… |
| `TextLine` | TextLine is a horizontal grouping of glyph components that share a baseline. |

<details>
<summary><code>Band</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func SegmentBands(prof []int, threshold, minGap, minLength int) ([]Band, error)` | SegmentBands splits a projection profile into bands of consecutive indices whose value is strictly greater than threshold. |
| `func (b Band) Length() int` | Length returns the number of indices covered by the band. |

</details>

<details>
<summary><code>Component</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func FilterByAspectRatio(comps []Component, minRatio, maxRatio float64) []Component` | FilterByAspectRatio returns the components whose bounding-box aspect ratio (width/height) lies in the inclusive range [minRatio, maxRatio]. |
| `func FilterByFillRatio(comps []Component, minFill, maxFill float64) []Component` | FilterByFillRatio returns the components whose foreground fill ratio (area / bounding-box area) lies in the inclusive range [minFill, maxFill]. |
| `func FilterBySize(comps []Component, minArea, maxArea int) []Component` | FilterBySize returns the components whose pixel area lies in the inclusive range [minArea, maxArea]. |
| `func FilterTextComponents(comps []Component, opts TextComponentOptions) []Component` | FilterTextComponents keeps only the components that satisfy every bound in opts, returning the glyph-like candidates in input order. |
| `func (c Component) AspectRatio() float64` | AspectRatio returns the component's bounding-box width divided by its height. |
| `func (c Component) FillRatio() float64` | FillRatio returns the fraction of the component's bounding box occupied by foreground pixels, a value in (0,1]. |

</details>

<details>
<summary><code>ComponentSet</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func LabelComponents(binary *cv.Mat, conn Connectivity) (*ComponentSet, error)` | LabelComponents labels the foreground of a binary single-channel image (any non-zero sample is foreground) and returns the label image and… |
| `func (s *ComponentSet) Count() int` | Count returns the number of foreground components in the set. |
| `func (s *ComponentSet) Mask(label int) (*cv.Mat, error)` | Mask renders a single component as a fresh single-channel 0/255 `cv.Mat` of the full image size. |

</details>

<details>
<summary><code>IntegralImage</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewIntegralImage(src *cv.Mat) (*IntegralImage, error)` | NewIntegralImage builds the summed-area tables of src (reduced to luma if multi-channel). |
| `func (ii *IntegralImage) Mean(rect cv.Rect) float64` | Mean returns the average source sample over rect (clamped to the image). |
| `func (ii *IntegralImage) MeanStdDev(rect cv.Rect) (mean, std float64)` | MeanStdDev returns the mean and population standard deviation of the source samples over rect (clamped to the image). |
| `func (ii *IntegralImage) Sum(rect cv.Rect) float64` | Sum returns the sum of source samples over rect (clamped to the image). |

</details>

<details>
<summary><code>MSEROptions</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func DefaultMSEROptions() MSEROptions` | DefaultMSEROptions returns options suitable for detecting dark glyph-sized regions: Delta 5, MinArea 10, MaxArea 0 (half the image), MaxVariation… |

</details>

<details>
<summary><code>MSERRegion</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func DetectMSER(src *cv.Mat, opts MSEROptions) ([]MSERRegion, error)` | DetectMSER detects Maximally Stable Extremal Regions in src. |
| `func FilterTextRegions(regions []MSERRegion, minAspect, maxAspect float64, minHeight, maxHeight int, minFill float64) []MSERRegion` | FilterTextRegions keeps the MSER regions whose shape is plausible for a text glyph: bounding-box aspect ratio in [minAspect, maxAspect], height in… |
| `func (r MSERRegion) AspectRatio() float64` | AspectRatio returns the region's bounding-box width divided by its height, or 0 for a degenerate zero-height box. |

</details>

<details>
<summary><code>SWTLetter</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func SWTLetters(res *SWTResult, ratio float64, minArea int) ([]SWTLetter, error)` | SWTLetters groups the stroke pixels of res into connected candidates. |

</details>

<details>
<summary><code>SWTOptions</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func DefaultSWTOptions() SWTOptions` | DefaultSWTOptions returns options tuned for dark text on a light background: Canny thresholds 50/150, no explicit stroke cap, and a Pi/2 angle… |

</details>

<details>
<summary><code>SWTResult</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func StrokeWidthTransform(src *cv.Mat, opts SWTOptions) (*SWTResult, error)` | StrokeWidthTransform computes the Stroke-Width Transform of src. |
| `func (r *SWTResult) At(x, y int) float64` | At returns the stroke width at column x, row y, or 0 if the coordinates are out of range. |
| `func (r *SWTResult) Max() float64` | Max returns the largest stroke width in the map (0 if the map is empty of strokes). |
| `func (r *SWTResult) ToMat() *cv.Mat` | ToMat renders the stroke-width map as a single-channel `cv.Mat` for visualization, linearly scaling widths in (0, Max] to grey levels [1,255]; pixels… |

</details>

<details>
<summary><code>TextComponentOptions</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func DefaultTextComponentOptions() TextComponentOptions` | DefaultTextComponentOptions returns loose defaults suitable for medium-sized glyphs: area in [10, 0], aspect in [0.05, 3], fill in [0.1, 0.95]. |

</details>

<details>
<summary><code>TextLine</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func GroupTextLines(comps []Component, vertOverlap float64, maxGap int) ([]TextLine, error)` | GroupTextLines groups glyph-like components into text lines. |

</details>

### Variables

`ErrEmpty`, `ErrInvalidArgument`

Full signatures, doc comments and every runnable example are on
[pkg.go.dev](https://pkg.go.dev/github.com/malcolmston/opencv/textdet).

## License

MIT, as part of [`github.com/malcolmston/opencv`](..).
An independent re-implementation, not affiliated with or endorsed by the
original project.
