# structured_light — standard-library-only implementation of the structured-light pattern generation

[![Go Reference](https://pkg.go.dev/badge/github.com/malcolmston/opencv/structured_light.svg)](https://pkg.go.dev/github.com/malcolmston/opencv/structured_light)

Package structured_light is a standard-library-only implementation of the
structured-light pattern generation and decoding routines from OpenCV's
contrib "structured_light" module, built on top of the root cv package
(github.com/malcolmston/opencv).

It depends only on the Go standard library and the root cv package: no cgo, no
third-party code, and it imports no sibling cv/* subpackage. The root package
supplies the `github.com/malcolmston/opencv.Mat` container (8-bit, row-major,
channel-interleaved); everything specific to structured light is implemented
here from scratch.

## Install

```sh
go get github.com/malcolmston/opencv@v0.8.0
```

```go
import "github.com/malcolmston/opencv/structured_light"
```

## Usage

This is the package's own `ExampleCodePattern`, so it compiles and its output is
asserted on every `go test ./structured_light/`.

```go
c := sl.NewCodePattern(sl.GrayCodeParams{Width: 32, Height: 16}, sl.EncodingBinary)
	fmt.Printf("encoding=%s images=%d\n", c.Encoding, c.NumberOfPatternImages())
```

```
encoding=binary images=18
```

## Exported surface

### Functions

| Function | What it does |
| --- | --- |
| `func CombineGrayAndPhase(fringeOrder []int, wrapped []float64) []float64` | CombineGrayAndPhase fuses a coarse integer fringe order (recovered from a Gray-code or binary stack) with a fine wrapped phase (recovered from a… |
| `func CombineMasks(masks ...[]bool) []bool` | CombineMasks returns the element-wise logical AND of one or more boolean masks of equal length; a pixel is valid only where every input marks it… |
| `func ComputeAmplitude(captured []*cv.Mat, shift float64) []float64` | ComputeAmplitude returns the per-pixel modulation amplitude B = (2/N)·√(S²+C²) of an N-step phase-shifted stack, where S and C are the… |
| `func ComputeBackground(captured []*cv.Mat, shift float64) []float64` | ComputeBackground returns the per-pixel background (DC / bias) intensity of an N-step phase-shifted stack, i.e. |
| `func ComputeDataModulation(captured []*cv.Mat, shift float64) []float64` | ComputeDataModulation returns the per-pixel data modulation (fringe visibility) γ = B/A of an N-step phase-shifted stack, the ratio of the… |
| `func CoordMapToMat(coord []int, rows, cols, maxVal int) *cv.Mat` | CoordMapToMat renders a decoded coordinate map (one int per camera pixel, -1 for invalid) as a single-channel github.com/malcolmston/opencv.Mat of… |
| `func EncodePatternPNG(m *cv.Mat) ([]byte, error)` | EncodePatternPNG returns the PNG encoding of a pattern image as a byte slice, convenient for embedding a generated pattern in memory or a test. |
| `func FTPWrappedPhase(img *cv.Mat, horizontal bool) []float64` | FTPWrappedPhase decodes a single fringe image into a wrapped phase map using Fourier-transform profilometry — a one-shot alternative to multi-image… |
| `func FTPWrappedPhaseBand(img *cv.Mat, horizontal bool, carrier, band int) []float64` | FTPWrappedPhaseBand is `FTPWrappedPhase` with an explicit carrier bin and sideband half-width instead of automatic detection, for callers that know… |
| `func HeterodyneUnwrap(levels []FrequencyPhase, rows, cols int, horizontal bool) ([]float64, error)` | HeterodyneUnwrap unwraps two or three frequencies with the heterodyne (beat) method. |
| `func MaskToMat(mask []bool, rows, cols int) *cv.Mat` | MaskToMat renders a boolean validity mask as a single-channel github.com/malcolmston/opencv.Mat of size rows×cols, with 255 for true and 0 for false. |
| `func ModulationMask(mod []float64, minMod float64) []bool` | ModulationMask thresholds a data-modulation (or amplitude) map: a pixel is valid (true) when its value is at least minMod. |
| `func MultiFrequencyUnwrap(levels []FrequencyPhase, rows, cols int, horizontal bool) ([]float64, error)` | MultiFrequencyUnwrap performs hierarchical (temporal) phase unwrapping across a set of frequencies, resolving a fine, high-frequency phase whose… |
| `func NStepWrappedPhase(captured []*cv.Mat, shift float64) []float64` | NStepWrappedPhase computes the wrapped phase of an N-step phase-shifted stack with an arbitrary, caller-supplied uniform phase step, generalizing… |
| `func OverexposureMask(images []*cv.Mat, sat uint8) []bool` | OverexposureMask flags pixels that are saturated (sample ≥ sat) in any image of a stack; such pixels clip the sinusoid and corrupt the recovered… |
| `func PhaseGradientQuality(wrapped []float64, rows, cols int) []float64` | PhaseGradientQuality builds a quality map for a wrapped phase field, suitable for `QualityGuidedUnwrap`. |
| `func PhaseMapToMat(phase []float64, rows, cols int) *cv.Mat` | PhaseMapToMat renders a phase map (one float per pixel) as a single-channel github.com/malcolmston/opencv.Mat of size rows×cols, min-max normalized… |
| `func PhaseToCoord(abs []float64, freq float64, extent int) []float64` | PhaseToCoord converts an absolute (unwrapped) phase map into projector coordinates. |
| `func QualityGuidedUnwrap(wrapped, quality []float64, rows, cols int) []float64` | QualityGuidedUnwrap unwraps a 2-D wrapped phase map by flood-filling outward from the highest-quality pixel, always extending the unwrapped region… |
| `func SavePatternPNG(path string, m *cv.Mat) error` | SavePatternPNG writes a single pattern image to path as a PNG file, creating or truncating it. |
| `func SavePatternStack(dir, prefix string, stack []*cv.Mat) ([]string, error)` | SavePatternStack writes an entire pattern stack to dir as zero-padded PNG files named "<prefix>NN.png" (for example projector patterns from… |
| `func ShadowMask(white, black *cv.Mat, thresh int) []bool` | ShadowMask builds a boolean lit/shadow mask from an all-white and an all-black reference capture: a pixel is lit (true) when white−black exceeds… |
| `func TriangulatePoint(cam, proj CameraMatrix, camX, camY, projCol, projRow float64) [3]float64` | TriangulatePoint reconstructs the world point observed at camera pixel (camX, camY) and projector pixel (projCol, projRow) from the two projection… |
| `func UnwrapPhaseMap(wrapped []float64, rows, cols int, horizontal bool) []float64` | UnwrapPhaseMap removes the 2π discontinuities of a wrapped phase map by a simple line-by-line spatial unwrap along the fringe direction. |
| `func WrapPhase(a float64) float64` | WrapPhase brings an arbitrary phase angle into the canonical interval (-π, π]. |
| `func WritePatternPNG(w io.Writer, m *cv.Mat) error` | WritePatternPNG encodes a single pattern image as a PNG and writes it to w. |

### Types

| Type | What it is |
| --- | --- |
| `CameraMatrix` | CameraMatrix is a 3×4 pinhole projection matrix P mapping a homogeneous world point [X Y Z 1]ᵀ to homogeneous image coordinates P·[X Y Z 1]ᵀ. |
| `CodePattern` | CodePattern generates and decodes a column/row-encoding binary pattern set under a selectable `Encoding` (reflected Gray or natural binary), letting… |
| `Decoded` | Decoded holds the result of `GrayCodePattern.Decode`. |
| `Encoding` | Encoding selects how a `CodePattern` maps a projector coordinate to the bit sequence projected for it. |
| `FrequencyPhase` | FrequencyPhase pairs a wrapped phase map with the number of full fringe periods (Frequency) its projected pattern spanned across the varying image… |
| `GrayCodeParams` | GrayCodeParams configures the projector resolution a `GrayCodePattern` encodes. |
| `GrayCodePattern` | GrayCodePattern generates and decodes a binary reflected Gray-code pattern set for a projector of the configured resolution. |
| `Params` | Params configures a `SinusoidalPattern` (N-step phase-shifting profilometry). |
| `PointCloud` | PointCloud is the result of triangulating a decoded correspondence field. |
| `SinusoidalPattern` | SinusoidalPattern generates phase-shifted sinusoidal fringe patterns and decodes them into a wrapped phase map. |
| `StereoMatch` | StereoMatch is one camera-to-camera correspondence recovered by decoding the same projector patterns in two cameras: the left and right camera pixels… |

<details>
<summary><code>CameraMatrix</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewPinhole(k, r [3][3]float64, t [3]float64) CameraMatrix` | NewPinhole assembles a `CameraMatrix` as P = K·[R\|t], where K is the 3×3 intrinsic matrix, R the 3×3 world-to-camera rotation, and t the… |
| `func (c CameraMatrix) Project(x [3]float64) (u, v float64)` | Project maps a world point to its pixel coordinates (u, v) under this camera, performing the perspective divide. |

</details>

<details>
<summary><code>CodePattern</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewCodePattern(p GrayCodeParams, enc Encoding) *CodePattern` | NewCodePattern returns a `CodePattern` for the given resolution and encoding with the default robust-bit and shadow thresholds. |
| `func (c *CodePattern) Decode(captured []*cv.Mat, white, black *cv.Mat) (*Decoded, error)` | Decode recovers the camera→projector correspondence from a captured stack, mirroring `GrayCodePattern.Decode` but honouring the pattern's… |
| `func (c *CodePattern) Generate() []*cv.Mat` | Generate returns the pattern stack, column bits first then row bits, each bit immediately followed by its photometric inverse — the same order and… |
| `func (c *CodePattern) NumColBits() int` | NumColBits returns the number of code bits used to encode a projector column. |
| `func (c *CodePattern) NumRowBits() int` | NumRowBits returns the number of code bits used to encode a projector row. |
| `func (c *CodePattern) NumberOfPatternImages() int` | NumberOfPatternImages returns 2*(NumColBits+NumRowBits), the number of images `CodePattern.Generate` produces (a pattern and its inverse per bit),… |
| `func (c *CodePattern) ReferenceImages() (white, black *cv.Mat)` | ReferenceImages returns the all-white and all-black reference patterns, as in `GrayCodePattern.ReferenceImages`. |

</details>

<details>
<summary><code>Decoded</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func (d *Decoded) At(y, x int) (col, row int, ok bool)` | At returns the decoded projector (col, row) for camera pixel (x, y) and whether that pixel is valid. |

</details>

<details>
<summary><code>Encoding</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func (e Encoding) String() string` | String renders the encoding name. |

</details>

<details>
<summary><code>GrayCodePattern</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewGrayCodePattern(p GrayCodeParams) *GrayCodePattern` | NewGrayCodePattern returns a `GrayCodePattern` for the given projector resolution with the default robust-bit and shadow thresholds. |
| `func (g *GrayCodePattern) Decode(captured []*cv.Mat, white, black *cv.Mat) (*Decoded, error)` | Decode recovers the camera→projector correspondence from a captured pattern stack. |
| `func (g *GrayCodePattern) Generate() []*cv.Mat` | Generate returns the full stack of projection pattern images, each a single-channel github.com/malcolmston/opencv.Mat of size Height×Width whose… |
| `func (g *GrayCodePattern) NumColBits() int` | NumColBits returns the number of Gray-code bits used to encode a projector column (before inverses are added). |
| `func (g *GrayCodePattern) NumRowBits() int` | NumRowBits returns the number of Gray-code bits used to encode a projector row (before inverses are added). |
| `func (g *GrayCodePattern) NumberOfPatternImages() int` | NumberOfPatternImages returns the number of pattern images `GrayCodePattern.Generate` produces, namely 2*(NumColBits+NumRowBits): a pattern and an… |
| `func (g *GrayCodePattern) ReferenceImages() (white, black *cv.Mat)` | ReferenceImages returns the fully-lit (white, all 255) and fully-dark (black, all 0) reference patterns of size Height×Width. |

</details>

<details>
<summary><code>PointCloud</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func Triangulate(dec *Decoded, cam, proj CameraMatrix) *PointCloud` | Triangulate reconstructs a `PointCloud` from a Gray-code/phase decoding by triangulating every valid camera pixel against its decoded projector pixel. |
| `func TriangulateStereo(matches []StereoMatch, leftCam, rightCam CameraMatrix) *PointCloud` | TriangulateStereo reconstructs world points from `StereoMatch`es using two calibrated camera projection matrices. |
| `func (pc *PointCloud) Len() int` | Len returns the number of reconstructed points. |

</details>

<details>
<summary><code>SinusoidalPattern</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewSinusoidalPattern(params Params) *SinusoidalPattern` | NewSinusoidalPattern validates params and returns a `SinusoidalPattern`. |
| `func (s *SinusoidalPattern) ComputeWrappedPhase(captured []*cv.Mat) []float64` | ComputeWrappedPhase recovers the wrapped phase map from a captured stack of N phase-shifted images using the standard N-step estimator |
| `func (s *SinusoidalPattern) Generate() []*cv.Mat` | Generate returns NumOfPatternImages single-channel fringe images of size Height×Width. |
| `func (s *SinusoidalPattern) Params() Params` | Params returns a copy of the configuration the pattern was built with. |
| `func (s *SinusoidalPattern) PhaseShift() float64` | PhaseShift returns the effective phase step in radians between consecutive pattern images. |

</details>

<details>
<summary><code>StereoMatch</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func StereoDecode(left, right *Decoded) []StereoMatch` | StereoDecode converts two independently-decoded correspondence fields (from the left and right cameras viewing the same projector sequence) into a… |

</details>

### Constants

`DefaultWhiteThreshold`, `DefaultBlackThreshold`

Full signatures, doc comments and every runnable example are on
[pkg.go.dev](https://pkg.go.dev/github.com/malcolmston/opencv/structured_light).

## License

MIT, as part of [`github.com/malcolmston/opencv`](..).
An independent re-implementation, not affiliated with or endorsed by the
original project.
