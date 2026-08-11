# hdr — standard-library-only implementation of the high-dynamic-range imaging pipeline

[![Go Reference](https://pkg.go.dev/badge/github.com/malcolmston/opencv/hdr.svg)](https://pkg.go.dev/github.com/malcolmston/opencv/hdr)

Package hdr is a standard-library-only implementation of the
high-dynamic-range imaging pipeline from OpenCV's photo module, built on top
of the root cv package (github.com/malcolmston/opencv).

It depends only on the Go standard library (math, sort) and the root cv
package: no cgo, no third-party code, and no dependency on any sibling cv/*
subpackage. Small numeric helpers (rounding-and-clamping to [0,255], luma,
separable Gaussian blur, image pyramids) are reimplemented locally rather than
reaching into the internals of other packages.

## Install

```sh
go get github.com/malcolmston/opencv@v0.8.0
```

```go
import "github.com/malcolmston/opencv/hdr"
```

## Usage

This is the package's own `ExampleAlignMTB`, so it compiles and its output is
asserted on every `go test ./hdr/`.

```go
scene := cv.NewMat(24, 24, 1)
	for y := 0; y < 24; y++ {
		for x := 0; x < 24; x++ {
			scene.Data[y*24+x] = uint8((x*13 + y*7) % 256)
		}
	}
	crop := func(oy, ox int) *cv.Mat {
		m := cv.NewMat(16, 16, 1)
		for y := 0; y < 16; y++ {
			for x := 0; x < 16; x++ {
				m.Data[y*16+x] = scene.Data[(oy+y)*24+(ox+x)]
			}
		}
		return m
	}
	ref := crop(4, 4)
	src := crop(5, 6)

	a := hdr.NewAlignMTB(3)
	dx, dy := a.CalculateShift(ref, src)
	fmt.Printf("recovered shift dx=%d dy=%d\n", dx, dy)
```

```
recovered shift dx=2 dy=1
```

## Exported surface

### Functions

| Function | What it does |
| --- | --- |
| `func ApplyColorMap(f *cv.FloatMat, cm ColorMap) *cv.Mat` | ApplyColorMap turns a single-channel `cv.FloatMat` into a three-channel 8-bit pseudo-colour image. |
| `func DetailEnhance(m *cv.Mat, sigmaSpace, sigmaColor, boost float64) *cv.Mat` | DetailEnhance sharpens local detail while leaving large-scale tone untouched, the analogue of OpenCV's detailEnhance and a natural finishing step on… |
| `func EdgePreservingFilter(m *cv.Mat, sigmaSpace, sigmaColor float64) *cv.Mat` | EdgePreservingFilter smooths an 8-bit image while keeping edges crisp, the analogue of OpenCV's edgePreservingFilter. |
| `func GaussianWeight(z int) float64` | GaussianWeight is a Gaussian centred on mid-grey (127.5) with a standard deviation of a quarter of the range, giving a smooth roll-off that strongly… |
| `func HatWeight(z int) float64` | HatWeight is Debevec & Malik's triangular ("hat") weighting: it rises linearly from the extremes to a peak at the middle of the 8-bit range and is… |
| `func MergeMertens(images []*cv.Mat, params MergeMertensParams) (*cv.Mat, error)` | MergeMertens performs Mertens–Kautz–Van Reeth exposure fusion: it blends the LDR stack directly into a single well-exposed, displayable 8-bit… |
| `func TentWeight(z int) float64` | TentWeight is the symmetric tent min(z, 255-z): it reaches zero at both clipping points, fully rejecting saturated and black pixels. |
| `func UniformWeight(z int) float64` | UniformWeight gives every unsaturated pixel equal weight and rejects the two clipping points outright. |
| `func WriteHDR(w io.Writer, r *Radiance) error` | WriteHDR encodes a three-channel radiance map in the classic Radiance RGBE (.hdr / .pic) format using a shared exponent per pixel, the standard… |
| `func WritePFM(w io.Writer, r *Radiance) error` | WritePFM encodes the radiance map as a Portable Float Map (PFM), a simple binary float image format understood by many HDR tools. |

### Types

| Type | What it is |
| --- | --- |
| `AlignMTB` | AlignMTB aligns a bracket of hand-held exposures using Ward's (2003) median-threshold-bitmap (MTB) algorithm, the same method OpenCV exposes as… |
| `Bitmap` | Bitmap is a single-channel binary image whose samples are 0 or 1. |
| `CameraResponse` | CameraResponse is a recovered camera response function (CRF). |
| `ColorMap` | ColorMap selects a pseudo-colour palette for visualising a scalar field such as a radiance or luminance map, mirroring OpenCV's applyColorMap. |
| `MergeMertensParams` | MergeMertensParams configures `MergeMertens`. |
| `MergeMertensProcessor` | MergeMertensProcessor is a reusable, configurable Mertens exposure-fusion operator. |
| `Radiance` | Radiance is a dense, row-major, channel-interleaved image of float64 linear radiance values — the high-dynamic-range counterpart of `cv.Mat`. |
| `Tonemap` | Tonemap is the common interface for operators that compress a linear high-dynamic-range `Radiance` map into a displayable 8-bit `cv.Mat` with samples… |
| `TonemapDrago` | TonemapDrago implements Drago et al. |
| `TonemapDurand` | TonemapDurand implements Durand & Dorsey's (2002) fast bilateral-filtering tone reproduction, the operator OpenCV exposes as createTonemapDurand. |
| `TonemapGamma` | TonemapGamma is the simplest operator: it linearly normalises the radiance to [0,1] and applies a gamma curve. |
| `TonemapMantiuk` | TonemapMantiuk is a local-contrast approximation of Mantiuk et al. |
| `TonemapMantiukGradient` | TonemapMantiukGradient is a gradient-domain HDR compressor in the spirit of Fattal et al. |
| `TonemapReinhard` | TonemapReinhard implements Reinhard et al. |
| `WeightFunc` | WeightFunc maps an 8-bit pixel value to the confidence weight it receives when several exposures are combined into a radiance estimate. |

<details>
<summary><code>AlignMTB</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewAlignMTB(maxBits int) *AlignMTB` | NewAlignMTB returns an aligner with the given pyramid depth and OpenCV-style defaults (exclusion range 4, exclusion enabled). |
| `func (a *AlignMTB) CalculateShift(ref, src *cv.Mat) (dx, dy int)` | CalculateShift returns the integer (dx, dy) translation that best registers src onto ref: applying Shift(src, dx, dy) aligns it with ref. |
| `func (a *AlignMTB) ComputeBitmaps(m *cv.Mat) (tb, eb *Bitmap)` | ComputeBitmaps returns the threshold and exclusion bitmaps of a full resolution image, exactly as used internally by the alignment search. |
| `func (a *AlignMTB) Process(images []*cv.Mat) []*cv.Mat` | Process aligns a bracket of exposures to the middle frame and returns the registered copies in the original order. |
| `func (a *AlignMTB) Shift(m *cv.Mat, dx, dy int) *cv.Mat` | Shift returns a copy of m translated by (dx, dy): the output pixel (x, y) takes the value of the input pixel (x-dx, y-dy), with pixels shifted in… |

</details>

<details>
<summary><code>Bitmap</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func (b *Bitmap) Count() int` | Count returns the number of set (non-zero) samples in the bitmap. |

</details>

<details>
<summary><code>CameraResponse</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func CalibrateDebevec(images []*cv.Mat, times []float64, samples, lambda int) (*CameraResponse, error)` | CalibrateDebevec recovers the camera response function from a bracket of LDR images and their exposure times using the Debevec & Malik least-squares… |
| `func CalibrateRobertson(images []*cv.Mat, times []float64, samples, iterations int) (*CameraResponse, error)` | CalibrateRobertson recovers the camera response with Robertson's iterative maximum-likelihood method: it alternates between estimating per-pixel… |
| `func LinearResponse(channels int) *CameraResponse` | LinearResponse returns a CameraResponse whose curve is the identity (linear) mapping for the given number of channels: Curve`c``z` = z+1. |
| `func (r *CameraResponse) ChannelCurve(c int) []float64` | ChannelCurve returns a copy of the 256-entry response lookup table for channel c, so callers can inspect or plot it without risking mutation of the… |
| `func (r *CameraResponse) EnforceMonotonic()` | EnforceMonotonic repairs each channel's response curve in place so it becomes non-decreasing and strictly positive, forward-filling any non-positive… |
| `func (r *CameraResponse) IsMonotonic(tol float64) bool` | IsMonotonic reports whether every channel's response curve is non-decreasing to within tol (a small negative slope up to tol is tolerated). |
| `func (r *CameraResponse) LogResponse(c int) []float64` | LogResponse returns the natural logarithm of channel c's response curve — the Debevec g(z) function — as a fresh 256-entry slice. |
| `func (r *CameraResponse) Normalize()` | Normalize rescales every channel so that its mid-range entry (value 128) is exactly 1, matching the convention of `CalibrateRobertson`. |
| `func (r *CameraResponse) Response(c, z int) float64` | Response returns the linear scene radiance that channel c maps the 8-bit pixel value z to. |

</details>

<details>
<summary><code>MergeMertensParams</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewMergeMertensParams() MergeMertensParams` | NewMergeMertensParams returns the default Mertens fusion parameters (all exponents equal to 1). |

</details>

<details>
<summary><code>MergeMertensProcessor</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewMergeMertensProcessor(params MergeMertensParams) *MergeMertensProcessor` | NewMergeMertensProcessor returns a processor with the given weighting exponents. |
| `func (m *MergeMertensProcessor) Process(images []*cv.Mat) (*cv.Mat, error)` | Process fuses the bracket with the stored parameters. |
| `func (m *MergeMertensProcessor) ProcessWithExposures(images []*cv.Mat, times []float64) (*cv.Mat, error)` | ProcessWithExposures fuses the bracket, accepting per-image exposure times for API parity with OpenCV. |

</details>

<details>
<summary><code>Radiance</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func MergeDebevec(images []*cv.Mat, times []float64, resp *CameraResponse) (*Radiance, error)` | MergeDebevec merges a bracket of LDR exposures into a linear radiance map using the calibrated camera response and the per-image exposure times. |
| `func MergeDebevecFunc(images []*cv.Mat, times []float64, resp *CameraResponse, w WeightFunc) (*Radiance, error)` | MergeDebevecFunc is `MergeDebevec` with a caller-supplied robustness weighting instead of the fixed hat weight. |
| `func MergeRobertson(images []*cv.Mat, times []float64, resp *CameraResponse) (*Radiance, error)` | MergeRobertson merges an LDR bracket into a linear radiance map with Robertson's estimator, the counterpart of OpenCV's createMergeRobertson. |
| `func NewRadiance(rows, cols, channels int) *Radiance` | NewRadiance allocates a zero-filled Radiance. |
| `func ReadHDR(rd io.Reader) (*Radiance, error)` | ReadHDR decodes an uncompressed Radiance RGBE stream written by `WriteHDR` into a three-channel radiance map. |
| `func ReadPFM(rd io.Reader) (*Radiance, error)` | ReadPFM decodes a Portable Float Map written by `WritePFM` (or any other tool) into a radiance map. |
| `func (r *Radiance) At(y, x, c int) float64` | At returns the radiance sample at row y, column x, channel c. |
| `func (r *Radiance) ChannelFloatMat(c int) *cv.FloatMat` | ChannelFloatMat extracts channel c as a single-channel `cv.FloatMat`. |
| `func (r *Radiance) Clone() *Radiance` | Clone returns a deep copy with independent backing storage. |
| `func (r *Radiance) DynamicRange() float64` | DynamicRange returns the scene's dynamic range in stops (powers of two), log2(maxLum/minLum), computed over the strictly positive luminances. |
| `func (r *Radiance) LogAverageLuminance() float64` | LogAverageLuminance returns the log-average (geometric mean) luminance, the key statistic most tone mappers anchor on. |
| `func (r *Radiance) LuminanceFloatMat() *cv.FloatMat` | LuminanceFloatMat returns the relative luminance of the radiance map as a single-channel `cv.FloatMat` (Rec.709 weights for colour, a copy for a… |
| `func (r *Radiance) Mean() float64` | Mean returns the arithmetic mean of every sample across all channels. |
| `func (r *Radiance) MinMax() (min, max float64)` | MinMax returns the smallest and largest sample across every channel of the radiance map. |
| `func (r *Radiance) Normalized() *Radiance` | Normalized returns a new radiance map divided by its maximum sample so that the peak radiance is 1. |
| `func (r *Radiance) Scale(f float64) *Radiance` | Scale returns a new radiance map with every sample multiplied by f. |
| `func (r *Radiance) Set(y, x, c int, value float64)` | Set stores value at row y, column x, channel c. |
| `func (r *Radiance) Visualize(cm ColorMap) *cv.Mat` | Visualize renders the radiance map's luminance as a false-colour image for inspection. |

</details>

<details>
<summary><code>TonemapDrago</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewTonemapDrago() *TonemapDrago` | NewTonemapDrago returns a Drago tonemapper with the paper's default parameters (gamma 2.2, bias 0.85, unit saturation). |
| `func (t *TonemapDrago) Process(r *Radiance) *cv.Mat` | Process implements `Tonemap`. |

</details>

<details>
<summary><code>TonemapDurand</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewTonemapDurand() *TonemapDurand` | NewTonemapDurand returns a Durand tonemapper with sensible defaults (gamma 2.2, target contrast 4, range sigma 0.4, unit saturation, automatic… |
| `func (t *TonemapDurand) Process(r *Radiance) *cv.Mat` | Process implements `Tonemap`. |

</details>

<details>
<summary><code>TonemapGamma</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewTonemapGamma(gamma float64) *TonemapGamma` | NewTonemapGamma returns a gamma tonemapper. |
| `func (t *TonemapGamma) Process(r *Radiance) *cv.Mat` | Process implements `Tonemap`. |

</details>

<details>
<summary><code>TonemapMantiuk</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewTonemapMantiuk() *TonemapMantiuk` | NewTonemapMantiuk returns a Mantiuk-style tonemapper with default parameters (gamma 2.2, scale 0.7, unit saturation). |
| `func (t *TonemapMantiuk) Process(r *Radiance) *cv.Mat` | Process implements `Tonemap`. |

</details>

<details>
<summary><code>TonemapMantiukGradient</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewTonemapMantiukGradient() *TonemapMantiukGradient` | NewTonemapMantiukGradient returns a gradient-domain tonemapper with default parameters (gamma 2.2, beta 0.85, unit saturation, 400 solver iterations). |
| `func (t *TonemapMantiukGradient) Process(r *Radiance) *cv.Mat` | Process implements `Tonemap`. |

</details>

<details>
<summary><code>TonemapReinhard</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewTonemapReinhard() *TonemapReinhard` | NewTonemapReinhard returns a Reinhard tonemapper with sensible defaults (global operator, gamma 2.2, unit saturation). |
| `func (t *TonemapReinhard) Process(r *Radiance) *cv.Mat` | Process implements `Tonemap`. |

</details>

### Constants

`DefaultLambda`, `DefaultSamples`

Full signatures, doc comments and every runnable example are on
[pkg.go.dev](https://pkg.go.dev/github.com/malcolmston/opencv/hdr).

## License

MIT, as part of [`github.com/malcolmston/opencv`](..).
An independent re-implementation, not affiliated with or endorsed by the
original project.
