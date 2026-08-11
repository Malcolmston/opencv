# bgsegm — from-scratch, standard-library-only port of a useful subset of OpenCV's

[![Go Reference](https://pkg.go.dev/badge/github.com/malcolmston/opencv/bgsegm.svg)](https://pkg.go.dev/github.com/malcolmston/opencv/bgsegm)

Package bgsegm is a from-scratch, standard-library-only port of a useful
subset of OpenCV's background-segmentation module (bgsegm plus the two
subtractors that live in the core video module). It separates the moving
foreground of a frame sequence from the slowly changing background, producing
a per-pixel foreground mask for each frame.

Like the parent package `github.com/malcolmston/opencv` (imported here under
the alias cv), bgsegm is written entirely against the Go standard library
(only math). It uses no cgo and no third-party dependencies, and it does not
import any of the sibling cv/* subpackages.

## Install

```sh
go get github.com/malcolmston/opencv@v0.8.0
```

```go
import "github.com/malcolmston/opencv/bgsegm"
```

## Usage

This is the package's own `ExampleBackgroundSubtractorCNT`, so it compiles and its output is
asserted on every `go test ./bgsegm/`.

```go
sub := bgsegm.NewBackgroundSubtractorCNT(5, true, 900)
	for i := 0; i < 10; i++ {
		sub.Apply(solid(8, 8, 40))
	}
	mask := sub.Apply(blob8())
	fmt.Println("foreground pixels:", countFG(mask))
```

```
foreground pixels: 9
```

## Exported surface

### Functions

| Function | What it does |
| --- | --- |
| `func CleanupMask(mask *cv.Mat, ksize int) *cv.Mat` | CleanupMask returns a morphologically opened copy of mask: an erosion followed by a dilation over a ksize×ksize rectangular structuring element,… |
| `func CloseMask(mask *cv.Mat, ksize int) *cv.Mat` | CloseMask returns a morphologically closed copy of mask: a dilation followed by an erosion over a ksize×ksize rectangular structuring element,… |
| `func RefineMask(mask *cv.Mat, openKsize, closeKsize int) *cv.Mat` | RefineMask returns a cleaned-up copy of a raw foreground mask by first opening it with `CleanupMask` at openKsize (removing isolated speckle) and… |

### Types

| Type | What it is |
| --- | --- |
| `BackgroundSubtractor` | BackgroundSubtractor is the common interface implemented by every model in this package. |
| `BackgroundSubtractorCNT` | BackgroundSubtractorCNT is the counter-based ("CouNT") background model from OpenCV's bgsegm module — the fastest of the family. |
| `BackgroundSubtractorGMG` | BackgroundSubtractorGMG is a Bayesian per-pixel background model after Godbehere, Matsukawa and Goldberg ("Visual Tracking of Human Visitors under… |
| `BackgroundSubtractorGSOC` | BackgroundSubtractorGSOC is the sample-consensus background model contributed to OpenCV's bgsegm module during Google Summer of Code. |
| `BackgroundSubtractorKNN` | BackgroundSubtractorKNN is a non-parametric K-nearest-neighbours background model after Zivkovic and van der Heijden ("Efficient adaptive density… |
| `BackgroundSubtractorLSBP` | BackgroundSubtractorLSBP is the Local SVD Binary Pattern background model of Guo, Wang, Qi, Zhang, Liu and Chang ("Background Subtraction using Local… |
| `BackgroundSubtractorMOG` | BackgroundSubtractorMOG is the original adaptive Gaussian-mixture background model of KaewTraKulPong and Bowden ("An Improved Adaptive Background… |
| `BackgroundSubtractorMOG2` | BackgroundSubtractorMOG2 is an adaptive Gaussian-mixture background model after Zivkovic ("Improved Adaptive Gaussian Mixture Model for Background… |
| `RunningAverage` | RunningAverage is the simplest background model: it keeps an exponential moving average of the frame intensities as the background image and flags a… |
| `ShadowParams` | ShadowParams is the shadow-detection configuration shared, by embedding, among the subtractors that can recognise the moving shadow of a background… |
| `SyntheticSequenceGenerator` | SyntheticSequenceGenerator reproduces OpenCV's bgsegm helper of the same name: it manufactures an endless test video by sliding a small foreground… |

<details>
<summary><code>BackgroundSubtractorCNT</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewBackgroundSubtractorCNT(minPixelStability int, useHistory bool, maxPixelStability int) *BackgroundSubtractorCNT` | NewBackgroundSubtractorCNT creates a CNT subtractor. |
| `func (b *BackgroundSubtractorCNT) Apply(frame *cv.Mat) *cv.Mat` | Apply classifies frame, advances the per-pixel counters and returns the foreground mask. |
| `func (b *BackgroundSubtractorCNT) GetBackgroundImage() *cv.Mat` | GetBackgroundImage returns the per-pixel trusted background value (or, before any value has been trusted, the most recent observation) as a… |

</details>

<details>
<summary><code>BackgroundSubtractorGMG</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewBackgroundSubtractorGMG(numInitFrames int, decisionThreshold float64) *BackgroundSubtractorGMG` | NewBackgroundSubtractorGMG creates a GMG subtractor. |
| `func (b *BackgroundSubtractorGMG) Apply(frame *cv.Mat) *cv.Mat` | Apply learns from or classifies frame and returns the foreground mask. |
| `func (b *BackgroundSubtractorGMG) GetBackgroundImage() *cv.Mat` | GetBackgroundImage returns the centre intensity of each pixel's most-observed histogram bin as a single-channel image, or nil before the first Apply. |
| `func (b *BackgroundSubtractorGMG) GetDecisionThreshold() float64` | GetDecisionThreshold returns the foreground-probability decision threshold. |
| `func (b *BackgroundSubtractorGMG) GetLearningRate() float64` | GetLearningRate returns the histogram aging rate. |
| `func (b *BackgroundSubtractorGMG) GetNumBins() int` | GetNumBins returns the number of intensity quantisation bins per pixel. |
| `func (b *BackgroundSubtractorGMG) GetNumInitFrames() int` | GetNumInitFrames returns the length of the initial learning period. |
| `func (b *BackgroundSubtractorGMG) SetDecisionThreshold(t float64)` | SetDecisionThreshold sets the foreground-probability decision threshold. |
| `func (b *BackgroundSubtractorGMG) SetLearningRate(r float64)` | SetLearningRate sets the histogram aging rate. |
| `func (b *BackgroundSubtractorGMG) SetNumBins(n int)` | SetNumBins sets the number of intensity quantisation bins per pixel. |
| `func (b *BackgroundSubtractorGMG) SetNumInitFrames(n int)` | SetNumInitFrames sets the length of the initial learning period. |

</details>

<details>
<summary><code>BackgroundSubtractorGSOC</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewBackgroundSubtractorGSOC(nSamples int, distThreshold float64, detectShadows bool) *BackgroundSubtractorGSOC` | NewBackgroundSubtractorGSOC creates a GSOC subtractor. |
| `func (b *BackgroundSubtractorGSOC) Apply(frame *cv.Mat) *cv.Mat` | Apply classifies frame, updates the sample banks (including static-foreground absorption) and returns the foreground mask. |
| `func (b *BackgroundSubtractorGSOC) GetBackgroundImage() *cv.Mat` | GetBackgroundImage returns the per-pixel median stored sample as a single-channel image, or nil before the first Apply. |

</details>

<details>
<summary><code>BackgroundSubtractorKNN</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewBackgroundSubtractorKNN(history int, dist2Threshold float64, detectShadows bool) *BackgroundSubtractorKNN` | NewBackgroundSubtractorKNN creates a KNN subtractor. |
| `func (b *BackgroundSubtractorKNN) Apply(frame *cv.Mat) *cv.Mat` | Apply classifies frame, refreshes the sample banks and returns the foreground mask. |
| `func (b *BackgroundSubtractorKNN) GetBackgroundImage() *cv.Mat` | GetBackgroundImage returns the per-pixel median of the sample bank as a single-channel image, or nil before the first Apply. |
| `func (b *BackgroundSubtractorKNN) GetDetectShadows() bool` | GetDetectShadows reports whether shadow classification is enabled. |
| `func (b *BackgroundSubtractorKNN) GetDist2Threshold() float64` | GetDist2Threshold returns the squared intensity distance defining a neighbour. |
| `func (b *BackgroundSubtractorKNN) GetHistory() int` | GetHistory returns the sample-bank refresh horizon. |
| `func (b *BackgroundSubtractorKNN) GetKNNSamples() int` | GetKNNSamples returns the minimum neighbour count for a background classification. |
| `func (b *BackgroundSubtractorKNN) GetNSamples() int` | GetNSamples returns the number of samples stored per pixel. |
| `func (b *BackgroundSubtractorKNN) GetShadowThreshold() float64` | GetShadowThreshold returns the darkest relative intensity still treated as a shadow. |
| `func (b *BackgroundSubtractorKNN) SetDetectShadows(on bool)` | SetDetectShadows enables or disables shadow classification. |
| `func (b *BackgroundSubtractorKNN) SetDist2Threshold(t float64)` | SetDist2Threshold sets the squared intensity distance defining a neighbour. |
| `func (b *BackgroundSubtractorKNN) SetHistory(history int)` | SetHistory sets the sample-bank refresh horizon (History). |
| `func (b *BackgroundSubtractorKNN) SetKNNSamples(k int)` | SetKNNSamples sets the minimum neighbour count for a background classification. |
| `func (b *BackgroundSubtractorKNN) SetNSamples(n int)` | SetNSamples sets the number of samples stored per pixel. |
| `func (b *BackgroundSubtractorKNN) SetShadowThreshold(t float64)` | SetShadowThreshold sets the darkest relative intensity still treated as a shadow. |

</details>

<details>
<summary><code>BackgroundSubtractorLSBP</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewBackgroundSubtractorLSBP(nSamples int, intensityThreshold float64, detectShadows bool) *BackgroundSubtractorLSBP` | NewBackgroundSubtractorLSBP creates an LSBP subtractor. |
| `func (b *BackgroundSubtractorLSBP) Apply(frame *cv.Mat) *cv.Mat` | Apply computes the LSBP descriptors of frame, classifies each pixel against its sample consensus, updates the models and returns the foreground mask. |
| `func (b *BackgroundSubtractorLSBP) GetBackgroundImage() *cv.Mat` | GetBackgroundImage returns the per-pixel median stored intensity as a single-channel image, or nil before the first Apply. |

</details>

<details>
<summary><code>BackgroundSubtractorMOG</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewBackgroundSubtractorMOG(history, nmixtures int, detectShadows bool) *BackgroundSubtractorMOG` | NewBackgroundSubtractorMOG creates an original-MOG subtractor. |
| `func (b *BackgroundSubtractorMOG) Apply(frame *cv.Mat) *cv.Mat` | Apply classifies frame, updates the mixture model and returns the foreground mask. |
| `func (b *BackgroundSubtractorMOG) GetBackgroundImage() *cv.Mat` | GetBackgroundImage returns the per-pixel mean of the most confident background Gaussian as a single-channel image, or nil before the first Apply. |

</details>

<details>
<summary><code>BackgroundSubtractorMOG2</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewBackgroundSubtractorMOG2(history int, varThreshold float64, detectShadows bool) *BackgroundSubtractorMOG2` | NewBackgroundSubtractorMOG2 creates a MOG2 subtractor. |
| `func (b *BackgroundSubtractorMOG2) Apply(frame *cv.Mat) *cv.Mat` | Apply classifies frame, updates the mixture model and returns the foreground mask. |
| `func (b *BackgroundSubtractorMOG2) GetBackgroundImage() *cv.Mat` | GetBackgroundImage returns the per-pixel mean of the most confident background Gaussian as a single-channel image, or nil before the first Apply. |
| `func (b *BackgroundSubtractorMOG2) GetBackgroundRatio() float64` | GetBackgroundRatio returns the cumulative-weight fraction defining the background component set. |
| `func (b *BackgroundSubtractorMOG2) GetDetectShadows() bool` | GetDetectShadows reports whether shadow classification is enabled. |
| `func (b *BackgroundSubtractorMOG2) GetHistory() int` | GetHistory returns the nominal look-back length. |
| `func (b *BackgroundSubtractorMOG2) GetNMixtures() int` | GetNMixtures returns the maximum number of Gaussians kept per pixel. |
| `func (b *BackgroundSubtractorMOG2) GetShadowThreshold() float64` | GetShadowThreshold returns the darkest relative intensity still treated as a shadow. |
| `func (b *BackgroundSubtractorMOG2) GetVarThreshold() float64` | GetVarThreshold returns the squared-Mahalanobis match threshold. |
| `func (b *BackgroundSubtractorMOG2) SetBackgroundRatio(r float64)` | SetBackgroundRatio sets the cumulative-weight fraction defining the background component set. |
| `func (b *BackgroundSubtractorMOG2) SetDetectShadows(on bool)` | SetDetectShadows enables or disables shadow classification. |
| `func (b *BackgroundSubtractorMOG2) SetHistory(history int)` | SetHistory sets the nominal look-back length (History). |
| `func (b *BackgroundSubtractorMOG2) SetNMixtures(n int)` | SetNMixtures sets the maximum number of Gaussians kept per pixel. |
| `func (b *BackgroundSubtractorMOG2) SetShadowThreshold(t float64)` | SetShadowThreshold sets the darkest relative intensity still treated as a shadow. |
| `func (b *BackgroundSubtractorMOG2) SetVarThreshold(t float64)` | SetVarThreshold sets the squared-Mahalanobis match threshold (VarThreshold). |

</details>

<details>
<summary><code>RunningAverage</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewRunningAverage(history int, threshold float64) *RunningAverage` | NewRunningAverage creates a running-average subtractor. |
| `func (r *RunningAverage) Apply(frame *cv.Mat) *cv.Mat` | Apply thresholds frame against the running background, updates the background and returns the foreground mask. |
| `func (r *RunningAverage) GetAlpha() float64` | GetAlpha returns the moving-average learning rate. |
| `func (r *RunningAverage) GetBackgroundImage() *cv.Mat` | GetBackgroundImage returns the current running-average background as a single-channel image, or nil before the first Apply. |
| `func (r *RunningAverage) GetThreshold() float64` | GetThreshold returns the foreground difference threshold. |
| `func (r *RunningAverage) SetAlpha(a float64)` | SetAlpha sets the moving-average learning rate. |
| `func (r *RunningAverage) SetThreshold(t float64)` | SetThreshold sets the absolute intensity difference above which a pixel is foreground. |

</details>

<details>
<summary><code>ShadowParams</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func (s *ShadowParams) GetDetectShadows() bool` | GetDetectShadows reports whether shadow classification is enabled. |
| `func (s *ShadowParams) GetShadowThreshold() float64` | GetShadowThreshold returns the darkest relative intensity still treated as a shadow. |
| `func (s *ShadowParams) GetShadowValue() uint8` | GetShadowValue returns the mask sample emitted for detected shadows. |
| `func (s *ShadowParams) SetDetectShadows(on bool)` | SetDetectShadows enables or disables shadow classification. |
| `func (s *ShadowParams) SetShadowThreshold(t float64)` | SetShadowThreshold sets the darkest relative intensity still treated as a shadow. |
| `func (s *ShadowParams) SetShadowValue(v uint8)` | SetShadowValue sets the mask sample emitted for detected shadows. |

</details>

<details>
<summary><code>SyntheticSequenceGenerator</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewSyntheticSequenceGenerator(background, object *cv.Mat, amplitude, wavelength, wavespeed, objspeed, noiseStdDev float64, seed int64) *SyntheticSequenceGenerator` | NewSyntheticSequenceGenerator creates a generator that slides object over background. |
| `func (g *SyntheticSequenceGenerator) Next() (frame, gtMask *cv.Mat)` | Next produces the next frame of the sequence and its ground-truth foreground mask. |

</details>

### Constants

`BackgroundValue`, `ForegroundValue`, `ShadowValue`

Full signatures, doc comments and every runnable example are on
[pkg.go.dev](https://pkg.go.dev/github.com/malcolmston/opencv/bgsegm).

## License

MIT, as part of [`github.com/malcolmston/opencv`](..).
An independent re-implementation, not affiliated with or endorsed by the
original project.
