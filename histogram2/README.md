# histogram2 — provides a pure-Go, standard-library-only toolkit of classic histogram-based

[![Go Reference](https://pkg.go.dev/badge/github.com/malcolmston/opencv/histogram2.svg)](https://pkg.go.dev/github.com/malcolmston/opencv/histogram2)

Package histogram2 provides a pure-Go, standard-library-only toolkit of
classic histogram-based computer-vision routines built on top of the parent
package's `cv.Mat` image type.

The package never defines its own image type. Every routine that produces an
image returns a `cv.Mat`, and every routine that consumes one accepts a
`cv.Mat`, so results interoperate directly with the rest of the module
(drawing, morphology, thresholding, I/O and so on). The only new types it
introduces are small value-holders for the histograms themselves —
`Histogram1D`, `Histogram2D` and `Histogram3D` — plus the stateful `CLAHE`
and `HOGDescriptor` operators.

## Install

```sh
go get github.com/malcolmston/opencv@v0.8.0
```

```go
import "github.com/malcolmston/opencv/histogram2"
```

## Usage

This is the package's own `ExampleCLAHE_Apply`, so it compiles and its output is
asserted on every `go test ./histogram2/`.

```go
img := cv.NewMat(16, 16, 1)
	for i := range img.Data {
		img.Data[i] = uint8(100 + i%8)
	}
	c := histogram2.NewCLAHE(40, 4, 4)
	out, err := c.Apply(img)
	if err != nil {
		panic(err)
	}
	fmt.Println(out.Rows, out.Cols, out.Channels)
```

```
16 16 1
```

## Exported surface

### Functions

| Function | What it does |
| --- | --- |
| `func Bhattacharyya(a, b []float64) float64` | Bhattacharyya returns the Bhattacharyya distance of two equal-length histograms, matching OpenCV's HISTCMP_BHATTACHARYYA (also the Hellinger… |
| `func BuildSpecificationLUT(srcHist, refHist *Histogram1D) []uint8` | BuildSpecificationLUT builds a lookup table that maps each bin of srcHist to an output intensity in [0,255] such that applying it drives the source… |
| `func CalcBackProject1D(src *cv.Mat, channel int, hist *Histogram1D) (*cv.Mat, error)` | CalcBackProject1D projects a one-dimensional histogram back onto an image: each output pixel receives the histogram weight of the bin its channel… |
| `func CalcBackProject2D(src *cv.Mat, chX, chY int, hist *Histogram2D) (*cv.Mat, error)` | CalcBackProject2D projects a two-dimensional histogram of channels chX and chY back onto an image, rescaling so the largest bin maps to 255. |
| `func ChiSquare(a, b []float64) float64` | ChiSquare returns the asymmetric chi-square distance of two equal-length histograms, matching OpenCV's HISTCMP_CHISQR: the sum over bins of (a-b)^2 /… |
| `func ChiSquareAlt(a, b []float64) float64` | ChiSquareAlt returns the symmetric chi-square distance of two equal-length histograms, matching OpenCV's HISTCMP_CHISQR_ALT: twice the sum over bins… |
| `func CompareHist1D(a, b *Histogram1D, method CompareMethod) (float64, error)` | CompareHist1D compares two one-dimensional histograms with the given method. |
| `func CompareHist2D(a, b *Histogram2D, method CompareMethod) (float64, error)` | CompareHist2D compares two two-dimensional histograms with the given method, flattening them in row-major order. |
| `func ContrastStretch(src *cv.Mat, lowPct, highPct float64) (*cv.Mat, error)` | ContrastStretch rescales each channel of src independently so that the intensity at the lowPct percentile maps to 0 and the intensity at the highPct… |
| `func ContrastStretchRange(src *cv.Mat, inLow, inHigh uint8) (*cv.Mat, error)` | ContrastStretchRange linearly rescales every channel of src so that the input intensity range [inLow, inHigh] is mapped onto the full [0,255] range,… |
| `func Correlation(a, b []float64) float64` | Correlation returns the Pearson correlation coefficient of two equal-length histograms, matching OpenCV's HISTCMP_CORREL. |
| `func CumulativeDistribution(h *Histogram1D) []float64` | CumulativeDistribution returns the normalised cumulative distribution function of h as a slice of length h.BinCount, where entry i is the fraction of… |
| `func EMD1D(a, b []float64) float64` | EMD1D returns the one-dimensional Earth Mover's Distance (Wasserstein-1 distance) between two equal-length histograms with unit ground distance… |
| `func EqualizeHist(src *cv.Mat) (*cv.Mat, error)` | EqualizeHist performs global histogram equalisation on a single-channel image, spreading its intensities across the full [0,255] range to improve… |
| `func EqualizeHistLuminance(src *cv.Mat) (*cv.Mat, error)` | EqualizeHistLuminance equalises the luminance of a three-channel RGB image while preserving its colour ratios: it computes the ITU-R luma of each… |
| `func EqualizeHistPerChannel(src *cv.Mat) (*cv.Mat, error)` | EqualizeHistPerChannel equalises every channel of src independently and returns an image of the same shape. |
| `func GammaCorrect(src *cv.Mat, gamma float64) (*cv.Mat, error)` | GammaCorrect applies a power-law (gamma) transform to every sample of src: out = 255 * (in/255)^gamma. |
| `func HOG(src *cv.Mat, cellSize, bins int) ([]float64, error)` | HOG is a convenience wrapper that computes a Histogram of Oriented Gradients descriptor of src using the given cell size and orientation-bin count… |
| `func Intersection(a, b []float64) float64` | Intersection returns the histogram intersection of two equal-length histograms, matching OpenCV's HISTCMP_INTERSECT: the sum over bins of min(a, b). |
| `func KLDivergence(a, b []float64) float64` | KLDivergence returns the Kullback-Leibler divergence D(a\|\|b) of two equal-length histograms, computed against their normalised distributions. |
| `func MatchHistograms(src, reference *cv.Mat) (*cv.Mat, error)` | MatchHistograms remaps a single-channel source image so that its intensity distribution matches that of a single-channel reference image. |
| `func MatchHistogramsColor(src, reference *cv.Mat) (*cv.Mat, error)` | MatchHistogramsColor remaps each channel of a source image so that its per-channel intensity distribution matches the corresponding channel of a… |
| `func MinMaxStretch(src *cv.Mat) (*cv.Mat, error)` | MinMaxStretch linearly rescales each channel of src independently so that its observed minimum maps to 0 and its maximum to 255 (full-scale contrast… |
| `func SpecifyHistogram(src *cv.Mat, target *Histogram1D) (*cv.Mat, error)` | SpecifyHistogram remaps a single-channel image so that its intensity distribution approximates the target histogram (histogram specification). |

### Types

| Type | What it is |
| --- | --- |
| `CLAHE` | CLAHE is a Contrast Limited Adaptive Histogram Equalisation operator. |
| `CompareMethod` | CompareMethod selects the distance or similarity measure used by `CompareHist1D` and `CompareHist2D`. |
| `HOGDescriptor` | HOGDescriptor computes Histogram of Oriented Gradients feature vectors, the classic dense descriptor of Dalal and Triggs used for pedestrian and… |
| `Histogram1D` | Histogram1D is a one-dimensional histogram of sample values distributed over a half-open value range [Min, Max) split into Bins equal-width bins. |
| `Histogram2D` | Histogram2D is a two-dimensional joint histogram of two channels. |
| `Histogram3D` | Histogram3D is a three-dimensional joint histogram of a three-channel image, with Bins bins per channel over the range [0,256). |

<details>
<summary><code>CLAHE</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewCLAHE(clipLimit float64, tilesX, tilesY int) *CLAHE` | NewCLAHE returns a CLAHE operator. |
| `func (c *CLAHE) Apply(src *cv.Mat) (*cv.Mat, error)` | Apply runs contrast-limited adaptive histogram equalisation on a single-channel image and returns a new single-channel image of the same size. |
| `func (c *CLAHE) SetClipLimit(clipLimit float64)` | SetClipLimit updates the contrast-limiting threshold used by `CLAHE.Apply`. |
| `func (c *CLAHE) SetTilesGridSize(tilesX, tilesY int)` | SetTilesGridSize updates the tile grid used by `CLAHE.Apply`. |

</details>

<details>
<summary><code>HOGDescriptor</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewHOGDescriptor(cellSize, blockSize, bins int) *HOGDescriptor` | NewHOGDescriptor returns a HOG descriptor with the given cell size (pixels), block size (cells) and orientation-bin count. |
| `func (h *HOGDescriptor) Compute(src *cv.Mat) ([]float64, error)` | Compute evaluates the HOG descriptor of src and returns the concatenated, block-normalised feature vector. |
| `func (h *HOGDescriptor) FeatureLength(rows, cols int) int` | FeatureLength returns the length of the descriptor vector `HOGDescriptor.Compute` produces for an image of the given size, which is… |

</details>

<details>
<summary><code>Histogram1D</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func CalcHist1D(src *cv.Mat, channel, binCount int) (*Histogram1D, error)` | CalcHist1D computes a one-dimensional histogram of the given channel of src over the value range [0, 256) split into binCount bins. |
| `func CalcHist1DMasked(src *cv.Mat, channel, binCount int, mask *cv.Mat) (*Histogram1D, error)` | CalcHist1DMasked is like `CalcHist1D` but only counts pixels for which the corresponding sample of mask is non-zero. |
| `func NewHistogram1D(binCount int, min, max float64) *Histogram1D` | NewHistogram1D allocates a zeroed one-dimensional histogram with binCount bins spanning the half-open range [min, max). |
| `func (h *Histogram1D) At(i int) float64` | At returns the count stored in bin i. |
| `func (h *Histogram1D) BinIndex(value float64) int` | BinIndex maps a sample value to its bin index, clamping values below Min to bin 0 and values at or above Max to the final bin. |
| `func (h *Histogram1D) Clone() *Histogram1D` | Clone returns a deep copy of the histogram with its own backing storage. |
| `func (h *Histogram1D) Cumulative() *Histogram1D` | Cumulative returns a new histogram whose bin i holds the sum of bins 0..i of the receiver — the (unnormalised) cumulative distribution. |
| `func (h *Histogram1D) Entropy() float64` | Entropy returns the Shannon entropy of the histogram in bits, treating the normalised bin counts as a probability distribution. |
| `func (h *Histogram1D) Mean() float64` | Mean returns the count-weighted mean of the bin-centre values. |
| `func (h *Histogram1D) Normalize()` | Normalize rescales the histogram in place so that its bin counts sum to one, turning it into a probability distribution. |
| `func (h *Histogram1D) NormalizeMax(scale float64)` | NormalizeMax rescales the histogram in place so that its largest bin equals scale. |
| `func (h *Histogram1D) Peak() int` | Peak returns the index of the bin with the largest count. |
| `func (h *Histogram1D) Total() float64` | Total returns the sum of all bin counts. |
| `func (h *Histogram1D) Variance() float64` | Variance returns the count-weighted variance of the bin-centre values. |

</details>

<details>
<summary><code>Histogram2D</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func CalcHist2D(src *cv.Mat, chX, chY, binsX, binsY int) (*Histogram2D, error)` | CalcHist2D computes the joint histogram of channels chX and chY of src over the range [0,256), using binsX and binsY bins respectively. |
| `func NewHistogram2D(binsX, binsY int) *Histogram2D` | NewHistogram2D allocates a zeroed two-dimensional histogram. |
| `func (h *Histogram2D) At(x, y int) float64` | At returns the count in bin (x, y). |
| `func (h *Histogram2D) Clone() *Histogram2D` | Clone returns a deep copy of the histogram with its own backing storage. |
| `func (h *Histogram2D) Normalize()` | Normalize rescales the histogram in place so that its bin counts sum to one. |
| `func (h *Histogram2D) Set(x, y int, value float64)` | Set stores value in bin (x, y). |
| `func (h *Histogram2D) Total() float64` | Total returns the sum of all bin counts. |

</details>

<details>
<summary><code>Histogram3D</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func CalcHist3D(src *cv.Mat, bins int) (*Histogram3D, error)` | CalcHist3D computes the joint histogram of the first three channels of src (interpreted as an RGB triple) using bins bins per channel over [0,256). |
| `func NewHistogram3D(bins int) *Histogram3D` | NewHistogram3D allocates a zeroed three-dimensional histogram with bins bins per channel. |
| `func (h *Histogram3D) At(x, y, z int) float64` | At returns the count in bin (x, y, z). |
| `func (h *Histogram3D) Normalize()` | Normalize rescales the histogram in place so that its bin counts sum to one. |
| `func (h *Histogram3D) Set(x, y, z int, value float64)` | Set stores value in bin (x, y, z). |
| `func (h *Histogram3D) Total() float64` | Total returns the sum of all bin counts. |

</details>

### Variables

`ErrBinCount`, `ErrChannelRange`, `ErrEmptyImage`, `ErrInvalidArgument`, `ErrSizeMismatch`

Full signatures, doc comments and every runnable example are on
[pkg.go.dev](https://pkg.go.dev/github.com/malcolmston/opencv/histogram2).

## License

MIT, as part of [`github.com/malcolmston/opencv`](..).
An independent re-implementation, not affiliated with or endorsed by the
original project.
