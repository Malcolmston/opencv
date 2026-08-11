# fuzzy — standard-library-only implementation of fuzzy (F-)transform image processing

[![Go Reference](https://pkg.go.dev/badge/github.com/malcolmston/opencv/fuzzy.svg)](https://pkg.go.dev/github.com/malcolmston/opencv/fuzzy)

Package fuzzy is a standard-library-only implementation of fuzzy (F-)transform
image processing built on the root cv package (github.com/malcolmston/opencv).
It ports a useful subset of OpenCV's fuzzy / ft contrib module:
fuzzy-partition kernels (symmetric, single-vector and two-vector forms), the
two-dimensional degree-0 and degree-1 F-transforms (forward components and
inverse reconstruction), a fast separable process variant, F-transform
smoothing, and F-transform image inpainting (one-step, iterative, single-step
and multi-resolution), together with numeric quality reporting.

The degree-0 F-transform covers the image domain with a grid of overlapping
fuzzy basis (membership) functions — a "fuzzy partition". Each basis
function contributes one component: the weighted average of the pixels beneath
it, using the basis as the weight. Those components are a coarse,
node-resolution summary of the image. The inverse transform paints each
component back through its basis and, because the bases form a partition of
unity (they sum to 1 at every pixel), the overlapping contributions blend into
a smooth reconstruction of the original. Skipping unknown pixels while
averaging — and skipping bases that see none — turns the very same
machinery into an inpainter.

## Install

```sh
go get github.com/malcolmston/opencv@v0.8.0
```

```go
import "github.com/malcolmston/opencv/fuzzy"
```

## Usage

This is the package's own `ExampleCreateKernel`, so it compiles and its output is
asserted on every `go test ./fuzzy/`.

```go
k := fuzzy.CreateKernel(fuzzy.LinearBasis, 2)
	fmt.Printf("%dx%d centre=%.1f corner=%.1f\n", k.Rows, k.Cols, k.At(2, 2), k.At(0, 0))
```

```
5x5 centre=1.0 corner=0.0
```

## Exported surface

### Functions

| Function | What it does |
| --- | --- |
| `func CreateKernel(function BasisFunction, radius int) *cv.FloatMat` | CreateKernel builds the two-dimensional fuzzy-partition kernel for the given basis function and radius, mirroring OpenCV's ft::createKernel. |
| `func CreateKernel1D(function BasisFunction, radius int) []float64` | CreateKernel1D returns the sampled one-dimensional fuzzy basis (membership) vector for the given basis function and radius, mirroring the 1-D half of… |
| `func CreateKernelAB(a, b []float64) *cv.FloatMat` | CreateKernelAB builds a two-dimensional fuzzy-partition kernel as the outer product of two one-dimensional membership vectors, mirroring the… |
| `func CreateKernelVec(v []float64) *cv.FloatMat` | CreateKernelVec builds a square two-dimensional fuzzy-partition kernel as the outer product of a one-dimensional membership vector with itself,… |
| `func FT02DFLProcess(img *cv.Mat, function BasisFunction, radius int) *cv.Mat` | FT02DFLProcess smooths img with the degree-0 F-transform using the fast, separable linear-time algorithm, mirroring OpenCV's ft::FT02D_FL_process. |
| `func FT02DFLProcessFloat(img *cv.Mat, function BasisFunction, radius int) *cv.FloatMat` | FT02DFLProcessFloat is the real-valued counterpart of `FT02DFLProcess`, mirroring OpenCV's ft::FT02D_FL_process_float. |
| `func FT02DInverse(c *Components) *cv.Mat` | FT02DInverse reconstructs a smoothed image from F-transform components, mirroring OpenCV's ft::FT02D_inverse. |
| `func FT02DIteration(img *cv.Mat, kernel *cv.FloatMat, unknown *cv.Mat) (out, remaining *cv.Mat, stillUnknown int)` | FT02DIteration performs a single degree-0 F-transform inpainting iteration, mirroring OpenCV's ft::FT02D_iteration. |
| `func FT02DProcess(img *cv.Mat, kernel *cv.FloatMat, mask *cv.Mat) *cv.Mat` | FT02DProcess computes the F-transform components of img with the given kernel and immediately reconstructs them, mirroring OpenCV's ft::FT02D_process. |
| `func FT12DInverse(c *Components1) *cv.Mat` | FT12DInverse reconstructs an image from degree-1 F-transform components, mirroring OpenCV's ft::FT12D_inverse. |
| `func FT12DProcess(img *cv.Mat, kernel *cv.FloatMat, mask *cv.Mat) *cv.Mat` | FT12DProcess computes the degree-1 F-transform components of img with the given kernel and immediately reconstructs them, mirroring OpenCV's… |
| `func Filter(img *cv.Mat, function BasisFunction, radius int) *cv.Mat` | Filter smooths img with the degree-0 F-transform over a fuzzy partition of the given basis function and radius. |
| `func FilterDegree1(img *cv.Mat, function BasisFunction, radius int) *cv.Mat` | FilterDegree1 smooths img with the degree-1 (linear-polynomial) F-transform over a fuzzy partition of the given basis function and radius. |
| `func FilterLinear(img *cv.Mat, radius int) *cv.Mat` | FilterLinear smooths img with the degree-0 F-transform over a triangular (`LinearBasis`) fuzzy partition of the given radius. |
| `func FilterMultiRadius(img *cv.Mat, function BasisFunction, radii []int) []*cv.Mat` | FilterMultiRadius applies the degree-0 F-transform smoother at several radii and returns one reconstruction per radius, in the same order. |
| `func FilterSinus(img *cv.Mat, radius int) *cv.Mat` | FilterSinus smooths img with the degree-0 F-transform over a raised-cosine (`SinusBasis`) fuzzy partition of the given radius — the SINUS variant… |
| `func Inpaint(img, mask *cv.Mat, radius int, function BasisFunction, algorithm InpaintAlgorithm) *cv.Mat` | Inpaint reconstructs the unknown pixels of img using the degree-0 F-transform, mirroring OpenCV's ft::inpaint. |
| `func InpaintMultiStep(img, mask *cv.Mat, radius int, function BasisFunction) *cv.Mat` | InpaintMultiStep reconstructs the unknown pixels of img with a multi-resolution F-transform scheme, mirroring the spirit of OpenCV's ft::MULTI_STEP… |

### Types

| Type | What it is |
| --- | --- |
| `BasisFunction` | BasisFunction selects the shape of the one-dimensional fuzzy basis (membership) function used to build a fuzzy partition of the image domain. |
| `Components` | Components holds the degree-0 F-transform components of an image over a fuzzy partition. |
| `Components1` | Components1 holds the degree-1 (linear) F-transform components of an image over a fuzzy partition, mirroring the FT12D family of OpenCV's ft module. |
| `ErrorStats` | ErrorStats summarises the per-sample discrepancy between a reference image and an F-transform reconstruction of it. |
| `InpaintAlgorithm` | InpaintAlgorithm selects the F-transform inpainting strategy used by `Inpaint`. |

<details>
<summary><code>BasisFunction</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func (f BasisFunction) String() string` | String returns the name of the basis function. |

</details>

<details>
<summary><code>Components</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func FT02DComponents(img *cv.Mat, kernel *cv.FloatMat, mask *cv.Mat) *Components` | FT02DComponents computes the degree-0 F-transform components of img over the fuzzy partition induced by kernel, mirroring OpenCV's… |
| `func (c *Components) At(nodeY, nodeX, ch int) float64` | At returns the component value for node (nodeY, nodeX) and channel c. |
| `func (c *Components) Reconstruct() *cv.Mat` | Reconstruct is a convenience method equivalent to `FT02DInverse`(c); it paints the degree-0 components back into a full-size image. |

</details>

<details>
<summary><code>Components1</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func FT12DComponents(img *cv.Mat, kernel *cv.FloatMat, mask *cv.Mat) *Components1` | FT12DComponents computes the degree-1 F-transform components of img over the fuzzy partition induced by kernel, mirroring OpenCV's… |
| `func FT12DPolynomial(img *cv.Mat, kernel *cv.FloatMat, mask *cv.Mat, channel int) (comps *Components1, c00, c10, c01 *cv.FloatMat)` | FT12DPolynomial computes the degree-1 F-transform components of img and returns the three polynomial-coefficient planes for the requested channel,… |
| `func (c *Components1) At(nodeY, nodeX, ch int) float64` | At returns the constant (degree-0) coefficient c00 for node (nodeY, nodeX) and channel ch. |
| `func (c *Components1) CoeffPlane(which, channel int) *cv.FloatMat` | CoeffPlane extracts one polynomial-coefficient plane at node resolution for a single channel as a Bn x An `cv.FloatMat`. |
| `func (c *Components1) Polynomial(nodeY, nodeX, ch int) (c00, c10, c01 float64)` | Polynomial returns the three coefficients (c00, c10, c01) of the local linear polynomial fitted at node (nodeY, nodeX) for channel ch. |
| `func (c *Components1) Reconstruct() *cv.Mat` | Reconstruct is a convenience method equivalent to `FT12DInverse`(c); it paints the degree-1 components back into a full-size image. |

</details>

<details>
<summary><code>ErrorStats</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func MaskedError(reference, reconstructed, mask *cv.Mat) ErrorStats` | MaskedError is like `TransformError` but restricts the comparison to the pixels where mask is non-zero (in its first channel). |
| `func TransformError(reference, reconstructed *cv.Mat) ErrorStats` | TransformError compares reference against reconstructed sample by sample and returns the aggregate `ErrorStats`. |
| `func (e ErrorStats) String() string` | String renders the statistics compactly, e.g. |

</details>

<details>
<summary><code>InpaintAlgorithm</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func (a InpaintAlgorithm) String() string` | String returns the name of the algorithm. |

</details>

Full signatures, doc comments and every runnable example are on
[pkg.go.dev](https://pkg.go.dev/github.com/malcolmston/opencv/fuzzy).

## License

MIT, as part of [`github.com/malcolmston/opencv`](..).
An independent re-implementation, not affiliated with or endorsed by the
original project.
