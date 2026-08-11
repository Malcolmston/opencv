# dnn_superres — pure-Go, standard-library-only single-image super-resolution toolkit built on top

[![Go Reference](https://pkg.go.dev/badge/github.com/malcolmston/opencv/dnn_superres.svg)](https://pkg.go.dev/github.com/malcolmston/opencv/dnn_superres)

Package dnn_superres is a pure-Go, standard-library-only single-image
super-resolution toolkit built on top of the OpenCV port
github.com/malcolmston/opencv (imported here as cv). It mirrors the surface of
OpenCV's contrib module cv::dnn_superres::DnnSuperResImpl — a small stateful
object configured with SetModel(algorithm, scale) and driven with Upsample —
but it does NOT run any trained neural network. Every algorithm here is a
classical, deterministic, weight-free image-resampling method.

Images are `cv.Mat` values: row-major, channel-interleaved uint8 samples in
the 0–255 range. Grayscale (1 channel) and RGB (3 channels) inputs are both
supported; any positive channel count is in fact handled generically, each
channel resampled independently. The package treats data as RGB (not BGR),
consistent with the root package. Colour order is irrelevant to the maths:
every method is a per-channel spatial resample.

## Install

```sh
go get github.com/malcolmston/opencv@v0.8.0
```

```go
import "github.com/malcolmston/opencv/dnn_superres"
```

## Usage

This is the package's own `ExampleBenchmark`, so it compiles and its output is
asserted on every `go test ./dnn_superres/`.

```go
hi := cv.NewMat(24, 24, 3)
	for y := 0; y < hi.Rows; y++ {
		for x := 0; x < hi.Cols; x++ {
			for c := 0; c < hi.Channels; c++ {
				hi.Set(y, x, c, uint8((x*10+y*5)%256))
			}
		}
	}
	results, err := dnn_superres.Benchmark(hi, 2, nil)
	if err != nil {
		panic(err)
	}
	fmt.Printf("methods=%d named=%t\n", len(results), results[0].Name != "")
```

```
methods=11 named=true
```

## Exported surface

### Functions

| Function | What it does |
| --- | --- |
| `func EdgeGuidedUpscale(src *cv.Mat, scale int, strength float64) (*cv.Mat, error)` | EdgeGuidedUpscale enlarges src by an arbitrary integer scale (>= 2) and sharpens it with gradient-profile sharpening: a bicubic base is enhanced by… |
| `func IterativeBackProjection(src *cv.Mat, scale, iterations int) (*cv.Mat, error)` | IterativeBackProjection enlarges src by an arbitrary integer scale (>= 2) and then refines the result with iterative back-projection (IBP), the… |
| `func MSE(a, b *cv.Mat) (float64, error)` | MSE returns the mean squared error between two images of identical shape, averaged over every sample of every channel. |
| `func PSNR(a, b *cv.Mat) (float64, error)` | PSNR returns the Peak Signal-to-Noise Ratio in decibels between two images of identical shape, the standard reconstruction-quality metric for… |
| `func SSIM(a, b *cv.Mat) (float64, error)` | SSIM returns the Mean Structural Similarity Index between two images of identical shape, the perceptual companion to `PSNR`. |
| `func UpsampleBSpline(src *cv.Mat, scale int) (*cv.Mat, error)` | UpsampleBSpline enlarges src by an arbitrary integer scale (>= 2) using the cubic B-spline kernel, the smoothest, overshoot-free member of the cubic… |
| `func UpsampleBicubic(src *cv.Mat, scale int) (*cv.Mat, error)` | UpsampleBicubic enlarges src by an integer scale (2, 3 or 4) using Keys / Catmull-Rom bicubic convolution (a = -0.5), the 4-tap separable cubic used… |
| `func UpsampleBilinear(src *cv.Mat, scale int) (*cv.Mat, error)` | UpsampleBilinear enlarges src by an integer scale (2, 3 or 4) using bilinear interpolation — a separable 2-tap linear resample. |
| `func UpsampleDCCI(src *cv.Mat, scale int) (*cv.Mat, error)` | UpsampleDCCI enlarges src by an arbitrary integer scale (>= 2) with Directional Cubic Convolution Interpolation. |
| `func UpsampleESPCN(src *cv.Mat, scale int) (*cv.Mat, error)` | UpsampleESPCN enlarges src by an arbitrary integer scale (>= 2) with an ESPCN-style sub-pixel (pixel-shuffle) upscaler. |
| `func UpsampleEdgeDirected(src *cv.Mat, scale int) (*cv.Mat, error)` | UpsampleEdgeDirected enlarges src by an integer scale (2, 3 or 4) with an edge-directed method (a NEDI-lite / edge-guided cubic). |
| `func UpsampleFSRCNN(src *cv.Mat, scale int) (*cv.Mat, error)` | UpsampleFSRCNN enlarges src by an integer scale (2, 3 or 4) with an FSRCNN-STYLE finish: a bicubic base pass followed by a fixed separable… |
| `func UpsampleGaussian(src *cv.Mat, scale int) (*cv.Mat, error)` | UpsampleGaussian enlarges src by an arbitrary integer scale (>= 2) using a truncated Gaussian reconstruction kernel. |
| `func UpsampleGradientProfile(src *cv.Mat, scale int) (*cv.Mat, error)` | UpsampleGradientProfile is a convenience wrapper over `EdgeGuidedUpscale` with a fixed sharpening strength (1.5). |
| `func UpsampleHermite(src *cv.Mat, scale int) (*cv.Mat, error)` | UpsampleHermite enlarges src by an arbitrary integer scale (>= 2) using the two-tap cubic Hermite kernel, a fast smoothstep resample slightly crisper… |
| `func UpsampleIBP(src *cv.Mat, scale int) (*cv.Mat, error)` | UpsampleIBP is a convenience wrapper over `IterativeBackProjection` using a fixed, well-behaved iteration count (10). |
| `func UpsampleLanczos(src *cv.Mat, scale int) (*cv.Mat, error)` | UpsampleLanczos enlarges src by an integer scale (2, 3 or 4) using Lanczos-4 windowed-sinc interpolation (8-tap separable), matching… |
| `func UpsampleLanczos3(src *cv.Mat, scale int) (*cv.Mat, error)` | UpsampleLanczos3 enlarges src by an arbitrary integer scale (>= 2) using Lanczos-3 windowed-sinc interpolation (6-tap separable), a sharpness/ringing… |
| `func UpsampleLapSRN(src *cv.Mat, scale int) (*cv.Mat, error)` | UpsampleLapSRN enlarges src by an arbitrary integer scale (>= 2) with a LapSRN-style progressive (Laplacian-pyramid) upscaler. |
| `func UpsampleLumaOnly(src *cv.Mat, scale int, up UpsampleFunc) (*cv.Mat, error)` | UpsampleLumaOnly enlarges src by upscaling only the luma channel with the high-quality method up, while the chroma channels are enlarged with cheap… |
| `func UpsampleMitchell(src *cv.Mat, scale int) (*cv.Mat, error)` | UpsampleMitchell enlarges src by an arbitrary integer scale (>= 2) using the Mitchell–Netravali cubic (B = C = 1/3), a balanced photographic… |
| `func UpsampleNEDI(src *cv.Mat, scale int) (*cv.Mat, error)` | UpsampleNEDI enlarges src by an arbitrary integer scale (>= 2) with New Edge-Directed Interpolation. |
| `func UpsampleNearest(src *cv.Mat, scale int) (*cv.Mat, error)` | UpsampleNearest enlarges src by an integer scale (2, 3 or 4) using nearest-neighbour sampling. |
| `func UpsamplePerChannel(src *cv.Mat, scale int, up UpsampleFunc) (*cv.Mat, error)` | UpsamplePerChannel enlarges src by splitting it into single-channel planes, running up on each plane independently and merging the results. |
| `func UpsampleScale(src *cv.Mat, scale int) (*cv.Mat, error)` | UpsampleScale enlarges src by an arbitrary integer scale (>= 2) with bicubic (Keys/Catmull-Rom) convolution. |

### Types

| Type | What it is |
| --- | --- |
| `BenchmarkResult` | BenchmarkResult holds the reconstruction-quality scores of one method, measured against the original high-resolution reference. |
| `DnnSuperResImpl` | DnnSuperResImpl is a stateful single-image super-resolution engine mirroring OpenCV's cv::dnn_superres::DnnSuperResImpl. |
| `NamedUpsampler` | NamedUpsampler pairs a human-readable label with an `UpsampleFunc` so that `Benchmark` can report per-method scores. |
| `UpsampleFunc` | UpsampleFunc is the signature shared by every free-function upsampler in this package: it enlarges src by an integer scale and returns a new Mat or… |
| `UpsamplerType` | UpsamplerType enumerates the super-resolution algorithms this package provides. |

<details>
<summary><code>BenchmarkResult</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func Benchmark(hi *cv.Mat, scale int, methods []NamedUpsampler) ([]BenchmarkResult, error)` | Benchmark compares super-resolution methods on a controlled reconstruction task. |

</details>

<details>
<summary><code>DnnSuperResImpl</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewDnnSuperResImpl() *DnnSuperResImpl` | NewDnnSuperResImpl returns a new, unconfigured super-resolution engine. |
| `func (s *DnnSuperResImpl) GetAlgorithm() string` | GetAlgorithm returns the canonical name of the currently configured algorithm, or the empty string if `DnnSuperResImpl.SetModel` has not been called. |
| `func (s *DnnSuperResImpl) GetScale() int` | GetScale returns the currently configured scale factor, or 0 if `DnnSuperResImpl.SetModel` has not been called. |
| `func (s *DnnSuperResImpl) SetModel(algo string, scale int) error` | SetModel selects the algorithm and integer scale factor used by subsequent calls to `DnnSuperResImpl.Upsample`, mirroring the OpenCV method of the… |
| `func (s *DnnSuperResImpl) Upsample(img *cv.Mat) (*cv.Mat, error)` | Upsample enlarges img using the algorithm and scale set by `DnnSuperResImpl.SetModel`, returning a new Mat of size (Rows*scale, Cols*scale). |

</details>

<details>
<summary><code>NamedUpsampler</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func DefaultUpsamplers() []NamedUpsampler` | DefaultUpsamplers returns a representative set of the package's arbitrary-scale methods for use with `Benchmark`. |

</details>

<details>
<summary><code>UpsamplerType</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func ParseUpsamplerType(algo string) (UpsamplerType, error)` | ParseUpsamplerType maps an algorithm name to its `UpsamplerType`. |
| `func (t UpsamplerType) String() string` | String returns the canonical lowercase algorithm name for the type, the same spelling accepted by `DnnSuperResImpl.SetModel`. |
| `func (t UpsamplerType) Upsample(src *cv.Mat, scale int) (*cv.Mat, error)` | Upsample runs the algorithm named by t on src at the given scale (2, 3 or 4), returning a new Mat of size (Rows*scale, Cols*scale). |

</details>

Full signatures, doc comments and every runnable example are on
[pkg.go.dev](https://pkg.go.dev/github.com/malcolmston/opencv/dnn_superres).

## License

MIT, as part of [`github.com/malcolmston/opencv`](..).
An independent re-implementation, not affiliated with or endorsed by the
original project.
