# inpaint — standard-library-only image inpainting

[![Go Reference](https://pkg.go.dev/badge/github.com/malcolmston/opencv/inpaint.svg)](https://pkg.go.dev/github.com/malcolmston/opencv/inpaint)

Package inpaint is a standard-library-only image inpainting, restoration and
gradient-domain editing toolkit built on top of the root cv package
(github.com/malcolmston/opencv). It reconstructs missing or corrupted image
regions from their surroundings and blends imagery seamlessly.

The package operates on the root package's `cv.Mat`, a dense row-major matrix
of 8-bit unsigned samples. Single-channel (grayscale) and three-channel (RGB)
images are the common cases. Three-channel data is treated as RGB, matching
Go's image package and the root cv conventions, not OpenCV's native BGR. Masks
— which select the region to fill — are represented by the package's own
`Mask` type (a dense binary grid) that interoperates with `cv.Mat` via
`MaskFromMat` and `Mask.ToMat`. Gradient fields use the `FloatImage` helper.

## Install

```sh
go get github.com/malcolmston/opencv@v0.8.0
```

```go
import "github.com/malcolmston/opencv/inpaint"
```

## Usage

This is the package's own `ExampleDistanceTransform`, so it compiles and its output is
asserted on every `go test ./inpaint/`.

```go
m := inpaint.NewMask(5, 5)
	for y := 1; y < 4; y++ {
		for x := 1; x < 4; x++ {
			m.Set(y, x, true)
		}
	}
	d := inpaint.DistanceTransform(m)
	fmt.Printf("edge=%.0f centre=%.0f outside=%.0f\n", d[1*5+2], d[2*5+2], d[0])
```

```
edge=1 centre=2 outside=0
```

## Exported surface

### Functions

| Function | What it does |
| --- | --- |
| `func ColorChange(src *cv.Mat, mask *Mask, rMul, gMul, bMul float64) *cv.Mat` | ColorChange multiplies the colour of the region selected by mask in src by the per-channel factors (rMul, gMul, bMul) in the gradient domain: the… |
| `func DistanceTransform(mask *Mask) []float64` | DistanceTransform returns, for every pixel, an approximate Euclidean distance to the nearest unselected pixel of mask, computed by the Fast Marching… |
| `func IlluminationChange(src *cv.Mat, mask *Mask, alpha, beta float64) *cv.Mat` | IlluminationChange locally relights the region selected by mask in src by compressing its gradient field with the Fattal-style operator g' =… |
| `func Inpaint(img *cv.Mat, mask *Mask, radius int, method Method) *cv.Mat` | Inpaint reconstructs the pixels of img selected by mask using the chosen method, returning a filled clone (img is not modified). |
| `func InpaintCriminisi(img *cv.Mat, mask *Mask, opts CriminisiOptions) *cv.Mat` | InpaintCriminisi fills the pixels of img selected by mask using Criminisi, Pérez and Toyama's (2004) exemplar-based image completion. |
| `func InpaintDiffusion(img *cv.Mat, mask *Mask) *cv.Mat` | InpaintDiffusion fills the pixels of img selected by mask by solving the Laplace equation over the region with the surrounding pixels as Dirichlet… |
| `func InpaintMat(img, mask *cv.Mat, radius int, method Method) *cv.Mat` | InpaintMat is a convenience wrapper accepting the mask as a `cv.Mat` rather than a `Mask`: any pixel whose first channel is non-zero is filled. |
| `func InpaintNavierStokes(img *cv.Mat, mask *Mask, iterations int) *cv.Mat` | InpaintNavierStokes fills the pixels of img selected by mask with a Bertalmio-style fluid-dynamics scheme: the region is first initialised… |
| `func InpaintPatchMatch(img *cv.Mat, mask *Mask, opts PatchMatchOptions) *cv.Mat` | InpaintPatchMatch fills the pixels of img selected by mask by iterated PatchMatch synthesis: the hole is first initialised harmonically, then each… |
| `func InpaintTelea(img *cv.Mat, mask *Mask, radius int) *cv.Mat` | InpaintTelea reconstructs the pixels of img selected by mask using Telea's (2004) Fast Marching Method inpainting. |
| `func PoissonBlend(dst *cv.Mat, guidanceX, guidanceY *FloatImage, region *Mask, iterations int) *cv.Mat` | PoissonBlend reconstructs an image equal to dst outside region and, inside region, the image whose gradient best matches the guidance vector field… |
| `func SeamlessClone(src, dst *cv.Mat, mask *Mask, center image.Point, mode CloneMode) *cv.Mat` | SeamlessClone imports the masked part of src into dst using Poisson image editing, returning a new image (dst is not modified). |
| `func SolvePoisson(guidance *FloatImage, boundary *cv.Mat, region *Mask, iterations int) *cv.Mat` | SolvePoisson reconstructs an image whose discrete Laplacian equals the guidance field inside the selected region while matching boundary exactly… |
| `func TextureFlattening(src *cv.Mat, mask *Mask, lowThreshold, highThreshold float64) *cv.Mat` | TextureFlattening washes out fine texture inside the region selected by mask in src while preserving strong edges: gradients whose magnitude falls… |

### Types

| Type | What it is |
| --- | --- |
| `BlotchOptions` | BlotchOptions configures `DetectBlotches`. |
| `CloneMode` | CloneMode selects how source guidance gradients are chosen in `SeamlessClone`. |
| `CriminisiOptions` | CriminisiOptions configures `InpaintCriminisi`. |
| `FastMarcher` | FastMarcher solves the eikonal equation \|∇T\| = 1 by the Fast Marching Method, growing the arrival-time field T outward from a set of source… |
| `FloatImage` | FloatImage is a dense row-major matrix of float64 samples with the same channel-interleaved layout as `cv.Mat`. |
| `Mask` | Mask is a dense binary grid marking the pixels an inpainting or editing routine should treat as the region of interest (typically the hole to fill). |
| `Method` | Method selects the region-filling algorithm used by `Inpaint`. |
| `NNF` | NNF is an approximate nearest-neighbour field: for every valid patch centre in a source image it stores the offset to the best-matching patch centre… |
| `PatchMatchOptions` | PatchMatchOptions configures `InpaintPatchMatch`. |
| `ScratchOptions` | ScratchOptions configures `DetectScratches`. |
| `TextOptions` | TextOptions configures `DetectText`. |

<details>
<summary><code>BlotchOptions</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func DefaultBlotchOptions() BlotchOptions` | DefaultBlotchOptions returns settings that detect both bright and dark specks up to radius 2 (dust and dirt on scanned film). |

</details>

<details>
<summary><code>CriminisiOptions</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func DefaultCriminisiOptions() CriminisiOptions` | DefaultCriminisiOptions returns the default exemplar-based settings (PatchRadius 4, whole-image search). |

</details>

<details>
<summary><code>FastMarcher</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewFastMarcher(mask *Mask) *FastMarcher` | NewFastMarcher prepares a march over a mask: the selected (true) pixels of mask are the target region to be reached, and the unselected pixels are… |
| `func (f *FastMarcher) Solve(visit func(y, x int)) []float64` | Solve runs the march and returns the arrival-time field T in row-major order (length Rows*Cols). |

</details>

<details>
<summary><code>FloatImage</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func Divergence(gx, gy *FloatImage) *FloatImage` | Divergence returns the divergence of the vector field (gx, gy) using backward differences: div(x,y) = (gx(x)-gx(x-1)) + (gy(y)-gy(y-1)), the discrete… |
| `func FloatImageFromMat(m *cv.Mat) *FloatImage` | FloatImageFromMat converts a `cv.Mat` into a FloatImage with the same shape, promoting each 8-bit sample to float64. |
| `func GradientX(img *cv.Mat) *FloatImage` | GradientX returns the forward horizontal derivative field I(x+1)-I(x) of img (BORDER_REPLICATE at the right edge). |
| `func GradientY(img *cv.Mat) *FloatImage` | GradientY returns the forward vertical derivative field I(y+1)-I(y) of img (BORDER_REPLICATE at the bottom edge). |
| `func Laplacian(img *cv.Mat) *FloatImage` | Laplacian returns the 5-point discrete Laplacian of img: L = I(x-1)+I(x+1)+I(y-1)+I(y+1)-4*I (BORDER_REPLICATE). |
| `func NewFloatImage(rows, cols, channels int) *FloatImage` | NewFloatImage allocates a zero-filled FloatImage. |
| `func (f *FloatImage) At(y, x, c int) float64` | At returns the sample at row y, column x, channel c. |
| `func (f *FloatImage) Clone() *FloatImage` | Clone returns an independent deep copy. |
| `func (f *FloatImage) Set(y, x, c int, value float64)` | Set stores value at row y, column x, channel c. |
| `func (f *FloatImage) ToMat() *cv.Mat` | ToMat converts the field back to a `cv.Mat`, rounding and clamping every sample into [0,255]. |

</details>

<details>
<summary><code>Mask</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func DetectBlotches(img *cv.Mat, opts BlotchOptions) *Mask` | DetectBlotches locates small isolated dust/dirt specks in img using a square grayscale top-hat: an opening with a square element the size of the… |
| `func DetectScratches(img *cv.Mat, opts ScratchOptions) *Mask` | DetectScratches locates thin line defects (film scratches, pen strokes) in img via a directional grayscale top-hat: opening with a line structuring… |
| `func DetectText(img *cv.Mat, opts TextOptions) *Mask` | DetectText locates high-contrast text or caption overlays in img. |
| `func MaskFromColor(img *cv.Mat, target color.Color, tol uint8) *Mask` | MaskFromColor selects the pixels of img whose colour is within tol (per channel, inclusive) of target — useful for erasing a solid-coloured logo,… |
| `func MaskFromColorRange(img *cv.Mat, lower, upper color.Color) *Mask` | MaskFromColorRange selects the pixels of img whose every channel lies within the inclusive range [lower, upper] (taken as RGB) — an axis-aligned… |
| `func MaskFromMat(m *cv.Mat, threshold uint8) *Mask` | MaskFromMat builds a Mask from the first channel of m: a pixel is selected (true) where the sample is strictly greater than threshold. |
| `func NewMask(rows, cols int) *Mask` | NewMask allocates an all-false mask with the given dimensions. |
| `func (m *Mask) At(y, x int) bool` | At reports whether pixel (x, y) is selected. |
| `func (m *Mask) Boundary() []image.Point` | Boundary returns the selected pixels that touch (4-connected) at least one unselected or out-of-bounds pixel — the inner rim of the region — in… |
| `func (m *Mask) BoundingBox() (image.Rectangle, bool)` | BoundingBox returns the tight bounding rectangle of the selected pixels (Max exclusive) and whether any pixel is selected. |
| `func (m *Mask) Clone() *Mask` | Clone returns an independent deep copy of the mask. |
| `func (m *Mask) Count() int` | Count returns the number of selected (true) pixels. |
| `func (m *Mask) Dilate(radius int) *Mask` | Dilate returns a new mask grown by radius pixels using a square (Chebyshev) structuring element: a pixel is selected if any pixel within the radius… |
| `func (m *Mask) Empty() bool` | Empty reports whether no pixel is selected. |
| `func (m *Mask) Erode(radius int) *Mask` | Erode returns a new mask shrunk by radius pixels using a square structuring element: a pixel stays selected only if every pixel within the radius… |
| `func (m *Mask) FillRect(r image.Rectangle, v bool)` | FillRect selects (v true) or clears every pixel inside the rectangle r, intersected with the mask bounds. |
| `func (m *Mask) Intersect(o *Mask) *Mask` | Intersect returns a new mask selecting pixels selected in both m and o. |
| `func (m *Mask) Invert() *Mask` | Invert returns a new mask with every selection flipped. |
| `func (m *Mask) Set(y, x int, v bool)` | Set marks pixel (x, y) selected (v true) or not. |
| `func (m *Mask) SetAll(v bool)` | SetAll sets every pixel to v. |
| `func (m *Mask) Subtract(o *Mask) *Mask` | Subtract returns a new mask selecting pixels selected in m but not in o. |
| `func (m *Mask) ToMat() *cv.Mat` | ToMat renders the mask as a single-channel `cv.Mat` with 255 where selected and 0 elsewhere, suitable for saving or passing to Mat-based routines. |
| `func (m *Mask) Union(o *Mask) *Mask` | Union returns a new mask selecting pixels selected in either m or o. |

</details>

<details>
<summary><code>Method</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func (m Method) String() string` | String returns the method name. |

</details>

<details>
<summary><code>NNF</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func PatchMatchNNF(a, b *cv.Mat, patchRadius, iterations int) *NNF` | PatchMatchNNF computes an approximate nearest-neighbour field mapping each patch of a to its most similar patch in b using Barnes et al.'s (2009)… |
| `func (n *NNF) At(y, x int) (my, mx int)` | At returns the matched patch-centre coordinates (my, mx) in the target image for the source patch centred at (y, x). |
| `func (n *NNF) Distance(y, x int) float64` | Distance returns the patch match distance (sum of squared differences) at source centre (y, x). |
| `func (n *NNF) Reconstruct(target *cv.Mat) *cv.Mat` | Reconstruct synthesises an image the size of the field by copying, for each pixel, the value of the target-image pixel its best-matching patch centre… |

</details>

<details>
<summary><code>PatchMatchOptions</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func DefaultPatchMatchOptions() PatchMatchOptions` | DefaultPatchMatchOptions returns default synthesis settings (PatchRadius 3, Iterations 5, Passes 4). |

</details>

<details>
<summary><code>ScratchOptions</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func DefaultScratchOptions() ScratchOptions` | DefaultScratchOptions returns settings that detect both bright and dark vertical scratches up to 3 pixels wide (the common film-scratch case). |

</details>

<details>
<summary><code>TextOptions</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func DefaultTextOptions() TextOptions` | DefaultTextOptions returns default text-detection settings (gradient threshold 40, dilate radius 2). |

</details>

Full signatures, doc comments and every runnable example are on
[pkg.go.dev](https://pkg.go.dev/github.com/malcolmston/opencv/inpaint).

## License

MIT, as part of [`github.com/malcolmston/opencv`](..).
An independent re-implementation, not affiliated with or endorsed by the
original project.
