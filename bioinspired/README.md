# bioinspired — from-scratch, standard-library-only port of a useful subset of OpenCV's bioinspired

[![Go Reference](https://pkg.go.dev/badge/github.com/malcolmston/opencv/bioinspired.svg)](https://pkg.go.dev/github.com/malcolmston/opencv/bioinspired)

Package bioinspired is a from-scratch, standard-library-only port of a useful
subset of OpenCV's bioinspired contrib module: biologically-inspired (retina)
vision models. It mirrors the human retina's two main pathways to denoise and
enhance images (the parvocellular channel), extract motion and moving contours
(the magnocellular channel), and compress high dynamic range for display (fast
tone mapping).

Like the parent package `github.com/malcolmston/opencv` (imported here under
the alias cv), bioinspired is written entirely against the Go standard library
(only math, plus fmt for error messages). It uses no cgo and no third-party
dependencies, and it does not import any of the sibling cv/* subpackages:
helper routines such as separable low-pass filtering and colour conversion are
reimplemented locally in floating point.

## Install

```sh
go get github.com/malcolmston/opencv@v0.8.0
```

```go
import "github.com/malcolmston/opencv/bioinspired"
```

## Usage

This is the package's own `ExampleMosaicBayer`, so it compiles and its output is
asserted on every `go test ./bioinspired/`.

```go
rows, cols := 16, 16
	img := cv.NewMat(rows, cols, 3)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			img.Set(y, x, 0, uint8(40+x))
			img.Set(y, x, 1, uint8(50+y))
			img.Set(y, x, 2, uint8(60+(x+y)/2))
		}
	}
	mosaic := bioinspired.MosaicBayer(img, bioinspired.BayerRGGB)
	back := bioinspired.DemosaicBayer(mosaic, bioinspired.BayerRGGB)

	fmt.Printf("mosaic channels: %d\n", mosaic.Channels)
	fmt.Printf("recovered red near-exact: %v\n", abs8(img.At(8, 8, 0), back.At(8, 8, 0)) <= 1)
```

```
mosaic channels: 1
recovered red near-exact: true
```

## Exported surface

### Functions

| Function | What it does |
| --- | --- |
| `func ApplyFastToneMapping(input *cv.Mat) *cv.Mat` | ApplyFastToneMapping is a one-shot, allocation-free convenience wrapper around `RetinaFastToneMapping`: it builds a tone-mapping operator sized to… |
| `func DemosaicBayer(mosaic *cv.Mat, pattern BayerPattern) *cv.Mat` | DemosaicBayer demultiplexes a single-channel Bayer mosaic back into a three-channel RGB image using bilinear interpolation: at every pixel the… |
| `func MosaicBayer(rgb *cv.Mat, pattern BayerPattern) *cv.Mat` | MosaicBayer multiplexes a three-channel RGB image onto a single-channel Bayer mosaic: at each pixel it keeps only the colour dictated by the mosaic… |
| `func WriteRetinaParameters(p RetinaParameters) string` | WriteRetinaParameters serialises a parameter set to a deterministic, plain text format: one "key value" pair per line, keys in a fixed order,… |

### Types

| Type | What it is |
| --- | --- |
| `BayerPattern` | BayerPattern names one of the four 2x2 colour-filter-array (Bayer) mosaic layouts. |
| `IplMagnoParameters` | IplMagnoParameters groups the parameters of the magnocellular IPL pathway — the transient channel that responds to motion and moving contours. |
| `OPLandIplParvoParameters` | OPLandIplParvoParameters groups the parameters of the outer-plexiform-layer (OPL) and parvocellular inner-plexiform-layer (IPL) pathway — the… |
| `OnOffChannels` | OnOffChannels holds the two rectified halves of a centre-surround ganglion-cell response. |
| `Retina` | Retina is a simplified Gipsa-lab retina model. |
| `RetinaFastToneMapping` | RetinaFastToneMapping is a stateless high-dynamic-range tone-mapping operator built from the retina's photoreceptor and ganglion-cell… |
| `RetinaLogSampler` | RetinaLogSampler models the non-uniform, log-polar photoreceptor sampling of the retina: dense sampling near the fovea (image centre) and… |
| `RetinaParameters` | RetinaParameters is the full parameter set of the `Retina` model, split into the parvo (OPL+IPL) and magno (IPL) pathways, mirroring OpenCV's… |
| `RetinaProcessor` | RetinaProcessor wraps a `Retina` and adds OpenCV's channel-activation toggles. |
| `SegmentationParameters` | SegmentationParameters groups the parameters of the `TransientAreasSegmentationModule`. |
| `TransientAreasSegmentationModule` | TransientAreasSegmentationModule segments the moving areas of a scene from a transient (motion) signal, typically the magnocellular output of a… |

<details>
<summary><code>BayerPattern</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func (p BayerPattern) String() string` | String returns a short human-readable name for the pattern. |

</details>

<details>
<summary><code>IplMagnoParameters</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func (p IplMagnoParameters) Validate() error` | Validate reports whether the magno (IPL) parameters are in range: the compression sensitivity must lie in [0,1], the gain must be non-negative and… |

</details>

<details>
<summary><code>OPLandIplParvoParameters</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func (p OPLandIplParvoParameters) Validate() error` | Validate reports whether the parvo (OPL+IPL) parameters are in range: the sensitivity/gain fields must lie in [0,1] and the temporal/spatial… |

</details>

<details>
<summary><code>OnOffChannels</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func SplitOnOffChannels(input *cv.Mat, spatialConstant, gain float64) OnOffChannels` | SplitOnOffChannels models the ON/OFF ganglion-cell split of the parvocellular pathway. |

</details>

<details>
<summary><code>Retina</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewRetina(rows, cols int) *Retina` | NewRetina creates a Retina for frames of the given size using `DefaultRetinaParameters`. |
| `func NewRetinaWithParams(rows, cols int, p RetinaParameters) *Retina` | NewRetinaWithParams is like `NewRetina` but uses the supplied parameters. |
| `func (r *Retina) ApplyFastToneMapping(input *cv.Mat) *cv.Mat` | ApplyFastToneMapping tone-maps a single frame using an operator derived from this retina's photoreceptor and ganglion-cell spatial constants,… |
| `func (r *Retina) ClearBuffers()` | ClearBuffers resets all temporal state and discards the last output, so the next `Retina.Run` behaves as the first frame of a new sequence. |
| `func (r *Retina) GetInputSize() (rows, cols int)` | GetInputSize returns the frame size the retina expects, as (rows, cols), mirroring OpenCV's Retina::getInputSize. |
| `func (r *Retina) GetMagno() *cv.Mat` | GetMagno returns the last magnocellular output as a single-channel `cv.Mat` (the transient/motion response). |
| `func (r *Retina) GetMagnoRAW() *cv.FloatMat` | GetMagnoRAW returns the last magno response as a `cv.FloatMat` without 8-bit quantisation. |
| `func (r *Retina) GetOutputSize() (rows, cols int)` | GetOutputSize returns the size of the retina's output images, as (rows, cols), mirroring OpenCV's Retina::getOutputSize. |
| `func (r *Retina) GetParameters() RetinaParameters` | GetParameters returns a copy of the current parameters. |
| `func (r *Retina) GetParvo() *cv.Mat` | GetParvo returns the last parvocellular output as a `cv.Mat` with the same channel count as the input (denoised, detail- and colour-enhanced). |
| `func (r *Retina) GetParvoRAW() *cv.FloatMat` | GetParvoRAW returns the last parvo luminance response as a `cv.FloatMat` without 8-bit quantisation, useful for measuring small signals in tests. |
| `func (r *Retina) InputSize() (rows, cols int)` | InputSize returns the frame size the Retina was constructed for. |
| `func (r *Retina) PrintSetup() string` | PrintSetup returns a human-readable description of the retina's current parameters, mirroring OpenCV's Retina::printSetup. |
| `func (r *Retina) Run(input *cv.Mat)` | Run processes one frame through both pathways and stores the results. |
| `func (r *Retina) SetParameters(p RetinaParameters)` | SetParameters replaces the model parameters. |
| `func (r *Retina) SetupFromText(text string) error` | SetupFromText parses a parameter text (as produced by `Retina.Write` or `WriteRetinaParameters`) and applies it, mirroring OpenCV's Retina::setup… |
| `func (r *Retina) Write() string` | Write serialises the retina's current parameters to the plain-text format of `WriteRetinaParameters`, mirroring OpenCV's Retina::write. |

</details>

<details>
<summary><code>RetinaFastToneMapping</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewRetinaFastToneMapping(rows, cols int) *RetinaFastToneMapping` | NewRetinaFastToneMapping creates a tone-mapping operator for images of the given size with sensible default neighbourhood radii. |
| `func (t *RetinaFastToneMapping) ProcessFrame(input *cv.Mat) *cv.Mat` | ProcessFrame tone-maps a single image and returns a new `cv.Mat` of the same channel count with values compressed into [0,255]. |
| `func (t *RetinaFastToneMapping) Setup(photoreceptorsNeighborhoodRadius, ganglioncellsNeighborhoodRadius, meanLuminanceModulatorK float64)` | Setup configures the operator. |

</details>

<details>
<summary><code>RetinaLogSampler</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewRetinaLogSampler(rows, cols, rings, sectors int) *RetinaLogSampler` | NewRetinaLogSampler creates a sampler for Cartesian images of size rows x cols that produces log-polar images of size rings x sectors. |
| `func (s *RetinaLogSampler) InputSize() (rows, cols int)` | InputSize returns the Cartesian frame size, as (rows, cols). |
| `func (s *RetinaLogSampler) InverseSample(logPolar *cv.Mat) *cv.Mat` | InverseSample reconstructs a Cartesian image from a log-polar image produced by `RetinaLogSampler.Sample`. |
| `func (s *RetinaLogSampler) OutputSize() (rings, sectors int)` | OutputSize returns the log-polar image size, as (rings, sectors). |
| `func (s *RetinaLogSampler) Sample(input *cv.Mat) *cv.Mat` | Sample transforms a Cartesian image into its log-polar representation. |

</details>

<details>
<summary><code>RetinaParameters</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func DefaultRetinaParameters() RetinaParameters` | DefaultRetinaParameters returns the model's default parameters. |
| `func ReadRetinaParameters(text string) (RetinaParameters, error)` | ReadRetinaParameters parses the text produced by `WriteRetinaParameters`. |
| `func (p RetinaParameters) Validate() error` | Validate reports whether every parameter of both retina pathways is in range. |

</details>

<details>
<summary><code>RetinaProcessor</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewRetinaProcessor(rows, cols int) *RetinaProcessor` | NewRetinaProcessor creates a processor for frames of the given size using `DefaultRetinaParameters`, with both channels active. |
| `func NewRetinaProcessorWithParams(rows, cols int, p RetinaParameters) *RetinaProcessor` | NewRetinaProcessorWithParams is like `NewRetinaProcessor` but uses the supplied parameters. |
| `func (rp *RetinaProcessor) ActivateContoursProcessing(activate bool)` | ActivateContoursProcessing enables or disables the parvocellular (contours / detail) channel. |
| `func (rp *RetinaProcessor) ActivateMovingContoursProcessing(activate bool)` | ActivateMovingContoursProcessing enables or disables the magnocellular (moving contours / motion) channel. |
| `func (rp *RetinaProcessor) ClearBuffers()` | ClearBuffers resets the retina's temporal state. |
| `func (rp *RetinaProcessor) ContoursProcessingActive() bool` | ContoursProcessingActive reports whether the parvo channel is enabled. |
| `func (rp *RetinaProcessor) GetInputSize() (rows, cols int)` | GetInputSize returns the configured frame size, as (rows, cols). |
| `func (rp *RetinaProcessor) GetMagno() *cv.Mat` | GetMagno returns the single-channel magno output, or an all-zero image when moving-contour processing is disabled. |
| `func (rp *RetinaProcessor) GetOutputSize() (rows, cols int)` | GetOutputSize returns the output image size, as (rows, cols). |
| `func (rp *RetinaProcessor) GetParvo() *cv.Mat` | GetParvo returns the parvo output, or an all-zero image of the input's shape when contour processing is disabled. |
| `func (rp *RetinaProcessor) MovingContoursProcessingActive() bool` | MovingContoursProcessingActive reports whether the magno channel is enabled. |
| `func (rp *RetinaProcessor) Retina() *Retina` | Retina returns the underlying `Retina`, for access to parameters or the raw (unquantised) responses. |
| `func (rp *RetinaProcessor) Run(input *cv.Mat)` | Run processes one frame through the underlying retina. |

</details>

<details>
<summary><code>SegmentationParameters</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func DefaultSegmentationParameters() SegmentationParameters` | DefaultSegmentationParameters returns parameters tuned so that a compact, persistently moving region is flagged while a static scene produces an… |
| `func (p SegmentationParameters) Validate() error` | Validate reports whether the segmentation parameters are usable: constants must be non-negative and ThresholdOFF must not exceed ThresholdON (so the… |

</details>

<details>
<summary><code>TransientAreasSegmentationModule</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewTransientAreasSegmentationModule(rows, cols int) *TransientAreasSegmentationModule` | NewTransientAreasSegmentationModule creates a segmentation module for inputs of the given size using `DefaultSegmentationParameters`. |
| `func NewTransientAreasSegmentationModuleWithParams(rows, cols int, p SegmentationParameters) *TransientAreasSegmentationModule` | NewTransientAreasSegmentationModuleWithParams is like `NewTransientAreasSegmentationModule` but uses the supplied parameters. |
| `func (m *TransientAreasSegmentationModule) ClearAllBuffers()` | ClearAllBuffers resets the temporal energy state and discards the last segmentation, so the next `TransientAreasSegmentationModule.Run` behaves as… |
| `func (m *TransientAreasSegmentationModule) GetInputSize() (rows, cols int)` | GetInputSize returns the frame size the module expects, as (rows, cols). |
| `func (m *TransientAreasSegmentationModule) GetOutputSize() (rows, cols int)` | GetOutputSize returns the size of the segmentation picture, as (rows, cols). |
| `func (m *TransientAreasSegmentationModule) GetParameters() SegmentationParameters` | GetParameters returns a copy of the current parameters. |
| `func (m *TransientAreasSegmentationModule) GetSegmentationMask() []bool` | GetSegmentationMask returns the last segmentation as a boolean slice in row-major order (true = moving area). |
| `func (m *TransientAreasSegmentationModule) GetSegmentationPicture() *cv.Mat` | GetSegmentationPicture returns the last segmentation as a single-channel `cv.Mat` whose samples are 255 for moving areas and 0 elsewhere. |
| `func (m *TransientAreasSegmentationModule) PrintSetup() string` | PrintSetup returns a human-readable multi-line description of the current segmentation parameters. |
| `func (m *TransientAreasSegmentationModule) Run(input *cv.Mat)` | Run segments the moving areas of a single-channel transient image (for example the quantised magno output of a `Retina`). |
| `func (m *TransientAreasSegmentationModule) RunFloat(input *cv.FloatMat)` | RunFloat is like `TransientAreasSegmentationModule.Run` but takes the raw, unquantised transient as a `cv.FloatMat` (for example… |
| `func (m *TransientAreasSegmentationModule) Setup(p SegmentationParameters)` | Setup replaces the module parameters. |

</details>

Full signatures, doc comments and every runnable example are on
[pkg.go.dev](https://pkg.go.dev/github.com/malcolmston/opencv/bioinspired).

## License

MIT, as part of [`github.com/malcolmston/opencv`](..).
An independent re-implementation, not affiliated with or endorsed by the
original project.
