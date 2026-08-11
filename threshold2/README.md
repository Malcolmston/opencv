# threshold2 — implements image thresholding and binarization algorithms on top of the parent

[![Go Reference](https://pkg.go.dev/badge/github.com/malcolmston/opencv/threshold2.svg)](https://pkg.go.dev/github.com/malcolmston/opencv/threshold2)

Package threshold2 implements image thresholding and binarization algorithms
on top of the parent package's `cv.Mat` image type.

The package is a pure-Go, standard-library-only companion to the core opencv
module. It never allocates its own image type: every routine that produces an
image returns a `cv.Mat`, and every routine that consumes one accepts a
`cv.Mat`, so results interoperate directly with the rest of the library
(drawing, morphology, I/O, and so on).

## Install

```sh
go get github.com/malcolmston/opencv@v0.8.0
```

```go
import "github.com/malcolmston/opencv/threshold2"
```

## Usage

This is the package's own `ExampleForegroundRatio`, so it compiles and its output is
asserted on every `go test ./threshold2/`.

```go
img := bimodal(10, 10)
	r, err := threshold2.ForegroundRatio(img, 100, threshold2.ObjectBright)
	if err != nil {
		panic(err)
	}
	fmt.Printf("%.2f\n", r)
```

```
0.50
```

## Exported surface

### Functions

| Function | What it does |
| --- | --- |
| `func AdaptiveGaussian(src *cv.Mat, window int, c float64, p Polarity) (*cv.Mat, error)` | AdaptiveGaussian binarizes src by comparing each pixel to a Gaussian-weighted mean of its neighbourhood minus the constant c. |
| `func AdaptiveMean(src *cv.Mat, window int, c float64, p Polarity) (*cv.Mat, error)` | AdaptiveMean binarizes src by comparing each pixel to the mean of its window-by-window neighbourhood minus the constant c. |
| `func AdaptiveMedian(src *cv.Mat, window int, c float64, p Polarity) (*cv.Mat, error)` | AdaptiveMedian binarizes src by comparing each pixel to the median of its window-by-window neighbourhood minus the constant c. |
| `func Auto(src *cv.Mat, m Method, p Polarity) (*cv.Mat, int, error)` | Auto computes a global threshold for src with the estimator named by m and returns the binarized mask (foreground selected by p) together with the… |
| `func AutoThreshold(src *cv.Mat, m Method) (int, error)` | AutoThreshold computes a global threshold for src using the estimator named by m and returns the grey level in [0,255]. |
| `func Bernsen(src *cv.Mat, window int, contrastThreshold int, p Polarity) (*cv.Mat, error)` | Bernsen binarizes src with Bernsen's method. |
| `func Binarize(src *cv.Mat, thresh int, p Polarity) (*cv.Mat, error)` | Binarize thresholds src at the given grey level and returns a single-channel mask in which foreground pixels (selected by p) are 255 and the rest are… |
| `func ForegroundRatio(src *cv.Mat, thresh int, p Polarity) (float64, error)` | ForegroundRatio reports the fraction of pixels of src that would be labelled foreground when thresholded at thresh with polarity p. |
| `func Hysteresis(src *cv.Mat, low, high int) (*cv.Mat, error)` | Hysteresis performs double-threshold (hysteresis) binarization, the selection rule used by the Canny edge detector. |
| `func InRange(src *cv.Mat, lower, upper []uint8) (*cv.Mat, error)` | InRange produces a single-channel mask that is 255 where every channel of src lies within the inclusive range [lower`c`, upper`c`] and 0 elsewhere,… |
| `func Intermodes(src *cv.Mat, p Polarity) (*cv.Mat, int, error)` | Intermodes binarizes src using the threshold from `IntermodesThreshold`. |
| `func IntermodesThreshold(src *cv.Mat) (int, error)` | IntermodesThreshold returns the average of the two peaks of the histogram after Prewitt-Mendelsohn smoothing (the intermodes method). |
| `func IsoData(src *cv.Mat, p Polarity) (*cv.Mat, int, error)` | IsoData binarizes src using the threshold from `IsoDataThreshold`. |
| `func IsoDataThreshold(src *cv.Mat) (int, error)` | IsoDataThreshold returns the threshold found by the ISODATA (Ridler-Calvard) inter-means iteration: starting from the histogram mean it repeatedly… |
| `func Kapur(src *cv.Mat, p Polarity) (*cv.Mat, int, error)` | Kapur binarizes src using the threshold from `KapurThreshold`. |
| `func KapurThreshold(src *cv.Mat) (int, error)` | KapurThreshold returns the threshold that maximises the sum of the Shannon entropies of the foreground and background classes (Kapur, Sahoo and… |
| `func Kittler(src *cv.Mat, p Polarity) (*cv.Mat, int, error)` | Kittler binarizes src using the threshold from `KittlerThreshold`. |
| `func KittlerThreshold(src *cv.Mat) (int, error)` | KittlerThreshold returns the threshold that minimises the Kittler-Illingworth minimum-error criterion, which models each class as a Gaussian and… |
| `func Li(src *cv.Mat, p Polarity) (*cv.Mat, int, error)` | Li binarizes src using the threshold from `LiThreshold`. |
| `func LiThreshold(src *cv.Mat) (int, error)` | LiThreshold returns the threshold that minimises the cross-entropy between the image and its binarized version, using Li and Tam's iterative method. |
| `func Mean(src *cv.Mat, p Polarity) (*cv.Mat, int, error)` | Mean binarizes src at its mean grey level (see `MeanThreshold`). |
| `func MeanThreshold(src *cv.Mat) (int, error)` | MeanThreshold returns the mean grey level of src, rounded down. |
| `func Median(src *cv.Mat, p Polarity) (*cv.Mat, int, error)` | Median binarizes src at its median grey level (see `MedianThreshold`). |
| `func MedianThreshold(src *cv.Mat) (int, error)` | MedianThreshold returns the median grey level of src, i.e. |
| `func Minimum(src *cv.Mat, p Polarity) (*cv.Mat, int, error)` | Minimum binarizes src using the threshold from `MinimumThreshold`. |
| `func MinimumThreshold(src *cv.Mat) (int, error)` | MinimumThreshold returns the valley between the two peaks of the histogram after Prewitt-Mendelsohn smoothing (the minimum method). |
| `func Moments(src *cv.Mat, p Polarity) (*cv.Mat, int, error)` | Moments binarizes src using the threshold from `MomentsThreshold`. |
| `func MomentsThreshold(src *cv.Mat) (int, error)` | MomentsThreshold returns the threshold chosen by Tsai's moment-preserving method, which selects the level whose binarization preserves the first… |
| `func MultiKapur(src *cv.Mat, classes int) ([]int, error)` | MultiKapur returns classes-1 ascending thresholds that partition the intensity range into the requested number of classes by maximising the total… |
| `func MultiKapurQuantize(src *cv.Mat, classes int) (*cv.Mat, []int, error)` | MultiKapurQuantize applies `MultiKapur` and returns a single-channel image in which every pixel is mapped to the mid-grey of the class it falls into,… |
| `func MultiOtsu(src *cv.Mat, classes int) ([]int, error)` | MultiOtsu returns classes-1 ascending thresholds that split the intensity range into the requested number of classes by maximising the total… |
| `func MultiOtsuQuantize(src *cv.Mat, classes int) (*cv.Mat, []int, error)` | MultiOtsuQuantize applies `MultiOtsu` and returns a single-channel image in which every pixel is mapped to the mid-grey of its class, together with… |
| `func NICK(src *cv.Mat, window int, k float64, p Polarity) (*cv.Mat, error)` | NICK binarizes src with the NICK method, which extends Niblack for low contrast images using the local threshold mean + k*sqrt((sumSq - mean^2)/n),… |
| `func Niblack(src *cv.Mat, window int, k float64, p Polarity) (*cv.Mat, error)` | Niblack binarizes src with Niblack's adaptive method, using the local threshold mean + k*std. |
| `func Otsu(src *cv.Mat, p Polarity) (*cv.Mat, int, error)` | Otsu binarizes src using the threshold from `OtsuThreshold` and returns the mask (foreground selected by p) together with the threshold used. |
| `func Otsu2D(src *cv.Mat, p Polarity) (*cv.Mat, int, int, error)` | Otsu2D binarizes src with the two-dimensional Otsu method. |
| `func Otsu2DThreshold(src *cv.Mat) (s, t int, err error)` | Otsu2DThreshold returns the two-dimensional Otsu threshold pair (s, t): s is the threshold on the pixel grey level and t the threshold on the 3x3… |
| `func OtsuThreshold(src *cv.Mat) (int, error)` | OtsuThreshold returns the global threshold in [0,255] that maximises the between-class variance of src's intensity histogram (Otsu's method). |
| `func PerChannelOtsu(src *cv.Mat, p Polarity) (*cv.Mat, []int, error)` | PerChannelOtsu thresholds each channel of src independently with Otsu's method and returns an image of the same shape whose samples are 0 or 255,… |
| `func PerChannelThreshold(src *cv.Mat, thresholds []int, p Polarity) (*cv.Mat, error)` | PerChannelThreshold binarizes each channel of src with its own grey level from the thresholds slice, whose length must equal the channel count. |
| `func Percentile(src *cv.Mat, fraction float64, p Polarity) (*cv.Mat, int, error)` | Percentile binarizes src at the level returned by `PercentileThreshold` for the given fraction. |
| `func PercentileThreshold(src *cv.Mat, fraction float64) (int, error)` | PercentileThreshold returns the grey level whose cumulative histogram share is closest to the requested fraction, which must lie in [0,1]. |
| `func Phansalkar(src *cv.Mat, window int, k, r float64, p Polarity) (*cv.Mat, error)` | Phansalkar binarizes src with Phansalkar's method, tuned for low-contrast stained images. |
| `func Sauvola(src *cv.Mat, window int, k, r float64, p Polarity) (*cv.Mat, error)` | Sauvola binarizes src with Sauvola's adaptive method, using the local threshold mean * (1 + k*(std/r - 1)). |
| `func ToGray(src *cv.Mat) (*cv.Mat, error)` | ToGray returns a fresh single-channel `cv.Mat` holding the luma of src. |
| `func Triangle(src *cv.Mat, p Polarity) (*cv.Mat, int, error)` | Triangle binarizes src using the threshold from `TriangleThreshold`. |
| `func TriangleThreshold(src *cv.Mat) (int, error)` | TriangleThreshold returns the global threshold chosen by the triangle (Zack) method, which finds the level furthest from the line joining the… |
| `func Wolf(src *cv.Mat, window int, k float64, p Polarity) (*cv.Mat, error)` | Wolf binarizes src with the Wolf-Jolion method, an improvement on Niblack that normalises by the global maximum local contrast. |
| `func Yen(src *cv.Mat, p Polarity) (*cv.Mat, int, error)` | Yen binarizes src using the threshold from `YenThreshold`. |
| `func YenThreshold(src *cv.Mat) (int, error)` | YenThreshold returns the threshold chosen by Yen, Chang and Chang's maximum correlation criterion, an entropic method that favours compact classes. |

### Types

| Type | What it is |
| --- | --- |
| `Histogram` | Histogram is the 256-bin intensity histogram of a single-channel image. |
| `Method` | Method identifies a global threshold-estimation algorithm for the `AutoThreshold` and `Auto` dispatchers. |
| `Polarity` | Polarity selects which side of a threshold is treated as foreground (rendered as 255) when a routine binarizes an image. |

<details>
<summary><code>Histogram</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func ComputeHistogram(src *cv.Mat) (*Histogram, error)` | ComputeHistogram builds the 256-bin intensity histogram of src. |
| `func (h *Histogram) Cumulative() [256]int` | Cumulative returns the cumulative histogram, where entry v is the number of pixels with grey level at most v. |
| `func (h *Histogram) Density() [256]float64` | Density returns the normalised histogram, where each entry is the fraction of pixels at that grey level. |
| `func (h *Histogram) Mean() float64` | Mean returns the mean grey level of the histogram, or 0 for an empty histogram. |
| `func (h *Histogram) Peak() int` | Peak returns the grey level of the most populated bin. |
| `func (h *Histogram) Range() (first, last int)` | Range returns the lowest and highest grey levels that have a non-zero count. |
| `func (h *Histogram) Smoothed(r int) *Histogram` | Smoothed returns a new Histogram whose bins are the moving average of this one over a window of radius r (each output bin averages the 2r+1 input… |
| `func (h *Histogram) Variance() float64` | Variance returns the variance of the grey levels about the mean, or 0 for an empty histogram. |

</details>

<details>
<summary><code>Method</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func (m Method) String() string` | String returns the lowercase name of the method. |

</details>

### Variables

`ErrEmpty`

Full signatures, doc comments and every runnable example are on
[pkg.go.dev](https://pkg.go.dev/github.com/malcolmston/opencv/threshold2).

## License

MIT, as part of [`github.com/malcolmston/opencv`](..).
An independent re-implementation, not affiliated with or endorsed by the
original project.
