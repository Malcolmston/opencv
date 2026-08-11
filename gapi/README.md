# gapi — pure-Go port of OpenCV's G-API (graph API) module built on the standard-library-only

[![Go Reference](https://pkg.go.dev/badge/github.com/malcolmston/opencv/gapi.svg)](https://pkg.go.dev/github.com/malcolmston/opencv/gapi)

Package gapi is a pure-Go port of OpenCV's G-API (graph API) module built on
the standard-library-only OpenCV port github.com/malcolmston/opencv (imported
here as cv). It offers a lazy, graph-based way to express image processing:
operations do not run when you call them but instead record how a result is to
be computed, and the whole pipeline executes later, once, over concrete
images.

The root cv package is eager: cv.GaussianBlur(img, 5, 1.4) allocates and fills
a Mat immediately. G-API instead separates the description of a computation
from its execution. You build a graph out of symbolic nodes, compile it, and
then feed it data. The same compiled graph can be run many times on different
inputs, and because the structure is known ahead of time it can be validated,
reused and (in principle) optimised as a whole.

## Install

```sh
go get github.com/malcolmston/opencv@v0.8.0
```

```go
import "github.com/malcolmston/opencv/gapi"
```

## Usage

This is the package's own `Example`, so it compiles and its output is
asserted on every `go test ./gapi/`.

```go
in := gapi.NewMat()
	gray := gapi.RGB2Gray(in)
	blurred := gapi.GaussianBlur(gray, 5, 1.4)
	edges := gapi.Canny(blurred, 50, 100)

	cc := gapi.NewComputation(in, edges).Compile()

	img := cv.NewMat(16, 16, 3)
	for i := range img.Data {
		img.Data[i] = uint8(i % 256)
	}
	out := cc.Apply(img)[0]

	fmt.Printf("%dx%dx%d\n", out.Rows, out.Cols, out.Channels)
```

```
16x16x1
```

## Exported surface

### Functions

| Function | What it does |
| --- | --- |
| `func Split3(in GMat) (GMat, GMat, GMat)` | Split3 returns three lazy nodes, one per channel, extracting the planes of a three-channel image. |

### Types

| Type | What it is |
| --- | --- |
| `CmpOp` | CmpOp selects the per-sample comparison performed by `Cmp`. |
| `ColorConversionCode` | ColorConversionCode selects a conversion for `CvtColor`; see `cv.ColorConversionCode`. |
| `ComputationT` | ComputationT is a typed, single-image-in single-image-out convenience wrapper mirroring G-API's cv::GComputationT. It captures a build function that… |
| `FlipCode` | FlipCode selects the axis for `Flip`; see `cv.FlipCode`. |
| `GCompiled` | GCompiled is an executable form of a `GComputation`: the graph with a fixed topological evaluation order and a chosen kernel package, mirroring… |
| `GComputation` | GComputation is a captured, compilable image-processing graph, mirroring cv::GComputation. |
| `GKernel` | GKernel is a custom implementation for a named operation, mirroring the role of a G-API kernel. |
| `GKernelPackage` | GKernelPackage is an ordered set of `GKernel` implementations keyed by operation name, mirroring cv::gapi::GKernelPackage. |
| `GMat` | GMat is a symbolic handle to an image node in a `GComputation`'s deferred graph, mirroring OpenCV G-API's cv::GMat. |
| `GScalar` | GScalar is a symbolic handle to a scalar node in a deferred graph, mirroring cv::GScalar. |
| `Inputs` | Inputs bundles the concrete values bound to a computation's protocol inputs when the graph is run. |
| `InterpolationFlag` | InterpolationFlag selects the resampling used by `Resize`; see `cv.InterpolationFlag`. |
| `KernelContext` | KernelContext carries the fully-resolved inputs to a single operation node at graph-execution time. |
| `KernelFunc` | KernelFunc computes the output of one operation node from its resolved context. |
| `MorphShape` | MorphShape selects a structuring-element shape for `Dilate` and `Erode`; see `cv.MorphShape`. |
| `ThresholdType` | ThresholdType selects the behaviour of `Threshold`; see `cv.ThresholdType`. |

<details>
<summary><code>ComputationT</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewComputationT(build func(in GMat) GMat) *ComputationT` | NewComputationT captures build, which must construct the output GMat from the single input GMat it is given, and compiles the resulting graph once. |
| `func (t *ComputationT) Apply(in *cv.Mat) *cv.Mat` | Apply runs the captured pipeline on one image and returns the single result. |

</details>

<details>
<summary><code>GCompiled</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func (c *GCompiled) Apply(mats ...*cv.Mat) []*cv.Mat` | Apply runs the compiled graph on image inputs only (no scalar inputs are bound) and returns every declared output in order. |
| `func (c *GCompiled) Run(in Inputs) ([]*cv.Mat, error)` | Run executes the compiled graph on the bound inputs and returns the outputs. |

</details>

<details>
<summary><code>GComputation</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewComputation(in, out GMat) *GComputation` | NewComputation builds the common single-input, single-output computation from an input `GMat` to an output GMat. |
| `func NewComputationIO(matIns []GMat, scalarIns []GScalar, outs []GMat) *GComputation` | NewComputationIO builds a computation with image inputs, scalar inputs and image outputs, giving full control over the protocol. |
| `func NewComputationMulti(ins, outs []GMat) *GComputation` | NewComputationMulti builds a computation with several image inputs and outputs. |
| `func (c *GComputation) Apply(mats ...*cv.Mat) []*cv.Mat` | Apply is a convenience that compiles the graph and runs it once on the given scalar-free image inputs, returning the outputs. |
| `func (c *GComputation) Compile() *GCompiled` | Compile prepares the graph for execution with the built-in kernels, returning a reusable `GCompiled`. |
| `func (c *GComputation) CompileWith(pkg *GKernelPackage) *GCompiled` | CompileWith prepares the graph for execution, allowing a `GKernelPackage` to override the default implementation of any operation by name. |

</details>

<details>
<summary><code>GKernelPackage</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func Kernels(ks ...GKernel) *GKernelPackage` | Kernels builds a package from the given kernels. |
| `func (p *GKernelPackage) Combine(other *GKernelPackage) *GKernelPackage` | Combine returns a new package containing every kernel from p overlaid with every kernel from other, so entries in other take precedence on conflict. |
| `func (p *GKernelPackage) Include(k GKernel) *GKernelPackage` | Include adds or replaces the kernel for an operation and returns the package for chaining. |
| `func (p *GKernelPackage) Size() int` | Size reports how many operations the package overrides. |

</details>

<details>
<summary><code>GMat</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func Add(a, b GMat) GMat` | Add returns a lazy node computing the saturating per-sample sum of a and b. |
| `func AddC(a GMat, s GScalar) GMat` | AddC returns a lazy node that adds a run-time scalar to every sample of a, saturating to [0,255]. |
| `func AddWeighted(a GMat, alpha float64, b GMat, beta, gamma float64) GMat` | AddWeighted returns a lazy node computing alpha*a + beta*b + gamma. |
| `func BGR2Gray(src GMat) GMat` | BGR2Gray returns a lazy node converting a three-channel BGR image to single-channel grayscale. |
| `func BitwiseAnd(a, b GMat) GMat` | BitwiseAnd returns a lazy node computing the per-sample bitwise AND of a and b. |
| `func BitwiseNot(src GMat) GMat` | BitwiseNot returns a lazy node computing the per-sample complement of src. |
| `func BitwiseOr(a, b GMat) GMat` | BitwiseOr returns a lazy node computing the per-sample bitwise OR of a and b. |
| `func BitwiseXor(a, b GMat) GMat` | BitwiseXor returns a lazy node computing the per-sample bitwise XOR of a and b. |
| `func Blur(src GMat, ksize int) GMat` | Blur returns a lazy node smoothing src with a normalised ksize×ksize box filter. |
| `func Canny(src GMat, lowThresh, highThresh float64) GMat` | Canny returns a lazy node running the Canny edge detector on a single-channel image with the given hysteresis thresholds. |
| `func Cmp(a, b GMat, op CmpOp) GMat` | Cmp returns a lazy node comparing a and b per sample under op, producing 255 where the relation holds and 0 elsewhere, matching OpenCV's cv::compare. |
| `func CmpEQ(a, b GMat) GMat` | CmpEQ is Cmp with `CmpOpEQ`. |
| `func CmpGE(a, b GMat) GMat` | CmpGE is Cmp with `CmpOpGE`. |
| `func CmpGT(a, b GMat) GMat` | CmpGT is Cmp with `CmpOpGT`. |
| `func CmpLE(a, b GMat) GMat` | CmpLE is Cmp with `CmpOpLE`. |
| `func CmpLT(a, b GMat) GMat` | CmpLT is Cmp with `CmpOpLT`. |
| `func CmpNE(a, b GMat) GMat` | CmpNE is Cmp with `CmpOpNE`. |
| `func CvtColor(src GMat, code ColorConversionCode) GMat` | CvtColor returns a lazy node converting src between colour spaces per code. |
| `func Dilate(src GMat, shape MorphShape, ksize, iterations int) GMat` | Dilate returns a lazy node growing bright regions of src with a structuring element of the given shape and size (ksize×ksize), repeated iterations… |
| `func Div(a, b GMat, scale float64) GMat` | Div returns a lazy node computing the per-sample quotient scale*a/b. |
| `func EqualizeHist(src GMat) GMat` | EqualizeHist returns a lazy node performing global histogram equalisation on a single-channel image. |
| `func Erode(src GMat, shape MorphShape, ksize, iterations int) GMat` | Erode returns a lazy node shrinking bright regions of src with a structuring element of the given shape and size (ksize×ksize), repeated iterations… |
| `func Flip(src GMat, code FlipCode) GMat` | Flip returns a lazy node mirroring src along the axis chosen by code. |
| `func GaussianBlur(src GMat, ksize int, sigma float64) GMat` | GaussianBlur returns a lazy node convolving src with a separable Gaussian of size ksize×ksize and standard deviation sigma (sigma <= 0 derives it… |
| `func Laplacian(src GMat, ksize int, scale, delta float64) GMat` | Laplacian returns a lazy node computing the Laplacian of src with the given aperture (1 or 3), scaled by scale, offset by delta and clamped to… |
| `func Mask(src, mask GMat) GMat` | Mask returns a lazy node that keeps samples of src where the single-channel mask is non-zero and zeroes them elsewhere. |
| `func MedianBlur(src GMat, ksize int) GMat` | MedianBlur returns a lazy node replacing each sample with the median of its ksize×ksize neighbourhood. |
| `func Merge3(a, b, c GMat) GMat` | Merge3 returns a lazy node interleaving three single-channel images into one three-channel image. |
| `func Mul(a, b GMat, scale float64) GMat` | Mul returns a lazy node computing the per-sample product a*b*scale. |
| `func MulC(a GMat, s GScalar) GMat` | MulC returns a lazy node that multiplies every sample of a by a run-time scalar, saturating to [0,255]. |
| `func NewMat() GMat` | NewMat creates a protocol image input: a symbolic `GMat` whose pixels are supplied at run time. |
| `func Normalize(src GMat, alpha, beta float64) GMat` | Normalize returns a lazy node linearly rescaling src so its minimum maps to alpha and its maximum to beta (NORM_MINMAX). |
| `func RGB2Gray(src GMat) GMat` | RGB2Gray returns a lazy node converting a three-channel RGB image to single-channel grayscale. |
| `func Resize(src GMat, width, height int, interp InterpolationFlag) GMat` | Resize returns a lazy node scaling src to width×height using interp. |
| `func Sobel(src GMat, dx, dy, ksize int, scale, delta float64) GMat` | Sobel returns a lazy node computing the Sobel derivative of src. |
| `func Sub(a, b GMat) GMat` | Sub returns a lazy node computing the saturating per-sample difference a-b. |
| `func Threshold(src GMat, thresh, maxval float64, typ ThresholdType) GMat` | Threshold returns a lazy node applying a fixed-level threshold to a single-channel image. |
| `func Transpose(src GMat) GMat` | Transpose returns a lazy node swapping the rows and columns of src. |
| `func (g GMat) Valid() bool` | Valid reports whether the GMat refers to a graph node. |

</details>

<details>
<summary><code>GScalar</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func ConstScalar(v float64) GScalar` | ConstScalar creates a compile-time constant scalar node. |
| `func NewScalar() GScalar` | NewScalar creates a protocol scalar input whose value is bound at run time. |
| `func (g GScalar) Valid() bool` | Valid reports whether the GScalar refers to a graph node. |

</details>

### Constants

`ThreshBinary`, `ThreshBinaryInv`, `ThreshTrunc`, `ThreshToZero`, `ThreshToZeroInv`, `ThreshOtsu`, `ColorRGB2Gray`, `ColorBGR2Gray`, `ColorGray2RGB`, `ColorRGB2BGR`, `ColorBGR2RGB`, `ColorRGB2HSV`, `ColorHSV2RGB`, `InterNearest`, `InterLinear`, `FlipVertical`, `FlipHorizontal`, `FlipBoth`, `MorphRect`, `MorphCross`, `MorphEllipse`, `OpAdd`, `OpSub`, `OpMul`, `OpDiv`, `OpAddWeighted`, `OpAddC`, `OpMulC`, `OpMask`, `OpMerge3`, `OpSplit3`, `OpBitwiseAnd`, `OpBitwiseOr`, `OpBitwiseXor`, `OpBitwiseNot`, `OpCmp`, `OpThreshold`, `OpResize`, `OpFlip`, `OpTranspose`, `OpNormalize`, `OpCvtColor`, `OpBlur`, `OpGaussianBlur`, `OpMedianBlur`, `OpSobel`, `OpLaplacian`, `OpCanny`, `OpDilate`, `OpErode`, `OpEqualizeHist`, `OpRGB2Gray`, `OpBGR2Gray`

Full signatures, doc comments and every runnable example are on
[pkg.go.dev](https://pkg.go.dev/github.com/malcolmston/opencv/gapi).

## License

MIT, as part of [`github.com/malcolmston/opencv`](..).
An independent re-implementation, not affiliated with or endorsed by the
original project.
