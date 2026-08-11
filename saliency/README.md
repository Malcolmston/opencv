# saliency — from-scratch, standard-library-only port of a useful subset of OpenCV's contrib

[![Go Reference](https://pkg.go.dev/badge/github.com/malcolmston/opencv/saliency.svg)](https://pkg.go.dev/github.com/malcolmston/opencv/saliency)

Package saliency is a from-scratch, standard-library-only port of a useful
subset of OpenCV's contrib saliency module: algorithms that highlight the
regions of an image (or video) most likely to draw human visual attention.

The package sits on top of the root module github.com/malcolmston/opencv
(imported as cv) and the Go standard library (only math and math/cmplx). It
uses no cgo and no third-party dependencies, and it does not import any of the
other cv/* subpackages. Every detector operates on the package's central image
type, `cv.Mat` (8-bit unsigned samples, one or three channels), and static
detectors return a single-channel saliency map normalised to the 8-bit range,
where brighter samples mark more salient locations.

## Install

```sh
go get github.com/malcolmston/opencv@v0.8.0
```

```go
import "github.com/malcolmston/opencv/saliency"
```

## Usage

This is the package's own `ExampleAUCJudd`, so it compiles and its output is
asserted on every `go test ./saliency/`.

```go
img := brightDisk()
	sal := saliency.NewStaticSaliencyFineGrained().ComputeSaliency(img)
	fix := cv.NewMat(48, 48, 1)
	fix.Set(24, 24, 0, 255)
	fix.Set(23, 24, 0, 255)
	fix.Set(24, 23, 0, 255)
	fmt.Println(saliency.AUCJudd(sal, fix) > 0.5)
```

```
true
```

## Exported surface

### Functions

| Function | What it does |
| --- | --- |
| `func AUCJudd(salMap, fixation *cv.Mat) float64` | AUCJudd computes the AUC-Judd area-under-ROC score of a saliency map against a binary fixation map (any non-zero sample marks a fixated pixel). |
| `func AdaptiveBinaryMap(saliency *cv.Mat, factor float64) *cv.Mat` | AdaptiveBinaryMap thresholds a single-channel saliency map at factor times its mean value (clamped to the 8-bit range), the adaptive threshold of… |
| `func ApplyCenterBias(saliency *cv.Mat, sigmaFrac float64) *cv.Mat` | ApplyCenterBias multiplies a single-channel saliency map by a Gaussian center-prior (see `CenterBiasPrior`) and renormalises to the 8-bit range,… |
| `func CC(a, b *cv.Mat) float64` | CC computes the linear (Pearson) correlation coefficient between two saliency maps, in [-1,1]. |
| `func CenterBiasPrior(rows, cols int, sigmaFrac float64) *cv.Mat` | CenterBiasPrior returns a rows×cols single-channel Gaussian center-prior map (brightest at the image centre, falling off toward the edges),… |
| `func ComputeBinaryMap(saliency *cv.Mat) *cv.Mat` | ComputeBinaryMap converts a single-channel saliency map into a binary foreground mask (samples are 0 or 255) by thresholding it with Otsu's method,… |
| `func KLDiv(prediction, groundTruth *cv.Mat) float64` | KLDiv computes the Kullback-Leibler divergence of a predicted saliency map from a ground-truth fixation-density map, KL(groundTruth ‖ prediction). |
| `func NSS(salMap, fixation *cv.Mat) float64` | NSS computes the Normalized Scanpath Saliency: the saliency map is normalised to zero mean and unit standard deviation, and NSS is the average of the… |
| `func SIM(a, b *cv.Mat) float64` | SIM computes the similarity (histogram-intersection) metric between two saliency maps: each map is normalised to sum to one and SIM is the sum of the… |
| `func SaliencyToHeatmap(saliency *cv.Mat) *cv.Mat` | SaliencyToHeatmap renders a single-channel saliency map as a three-channel RGB pseudo-colour heatmap using a "jet"-style colormap (low saliency maps… |

### Types

| Type | What it is |
| --- | --- |
| `GMRSaliency` | GMRSaliency implements the Graph-based Manifold Ranking salient object detector of Yang, Zhang, Lu, Ruan & Yang, "Saliency Detection via Graph-Based… |
| `HistogramContrast` | HistogramContrast implements the histogram-based global-contrast salient region detector (HC) of Cheng, Mitra, Huang, Torr & Hu, "Global Contrast… |
| `MinimumBarrierSaliency` | MinimumBarrierSaliency implements the Minimum Barrier Distance (MBD) salient object detector of Zhang, Sclaroff, Lin, Shen, Price & Mech, "Minimum… |
| `MotionSaliencyBinWangApr2014` | MotionSaliencyBinWangApr2014 detects motion saliency — pixels belonging to moving objects — with a per-pixel background model built up over a… |
| `ObjectnessBING` | ObjectnessBING is a lightweight ("BING-lite") objectness detector inspired by Cheng et al., "BING: Binarized Normed Gradients for Objectness… |
| `ObjectnessBox` | ObjectnessBox is a candidate object window returned by `ObjectnessBING`. |
| `ObjectnessCascade` | ObjectnessCascade is a two-stage objectness proposer in the spirit of the BING cascade of Cheng et al. |
| `RegionContrast` | RegionContrast implements the region-based global-contrast salient region detector (RC) of Cheng, Mitra, Huang, Torr & Hu, "Global Contrast based… |
| `StaticSaliency` | StaticSaliency is implemented by the static (single-image) saliency detectors. |
| `StaticSaliencyBooleanMap` | StaticSaliencyBooleanMap implements Boolean Map based Saliency (BMS) after Zhang & Sclaroff, "Saliency Detection: A Boolean Map Approach" (ICCV 2013). |
| `StaticSaliencyContextAware` | StaticSaliencyContextAware implements a context-aware saliency detector after Goferman, Zelnik-Manor & Tal, "Context-Aware Saliency Detection" (CVPR… |
| `StaticSaliencyFineGrained` | StaticSaliencyFineGrained detects saliency with a multi-scale center-surround scheme in the spirit of Montabone & Soto, "Human detection using a… |
| `StaticSaliencyFrequencyTuned` | StaticSaliencyFrequencyTuned implements the frequency-tuned salient region detector of Achanta, Hemami, Estrada & Süsstrunk, "Frequency-tuned… |
| `StaticSaliencyIttiKochNiebur` | StaticSaliencyIttiKochNiebur implements the classical bottom-up visual attention model of Itti, Koch & Niebur, "A Model of Saliency-Based Visual… |
| `StaticSaliencySpectralResidual` | StaticSaliencySpectralResidual detects saliency with the spectral-residual method of Hou & Zhang, "Saliency Detection: A Spectral Residual Approach"… |

<details>
<summary><code>GMRSaliency</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewGMRSaliency() *GMRSaliency` | NewGMRSaliency returns a detector with a 10×10 region grid and the paper's default ranking parameters. |
| `func (s *GMRSaliency) ComputeSaliency(img *cv.Mat) *cv.Mat` | ComputeSaliency returns the manifold-ranking saliency map of img: a single-channel `cv.Mat` the same size as img, normalised to [0,255]. |

</details>

<details>
<summary><code>HistogramContrast</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewHistogramContrast() *HistogramContrast` | NewHistogramContrast returns a detector with 12 quantisation levels per channel. |
| `func (h *HistogramContrast) ComputeSaliency(img *cv.Mat) *cv.Mat` | ComputeSaliency returns the histogram-contrast saliency map of img: a single-channel `cv.Mat` the same size as img, normalised to [0,255]. |

</details>

<details>
<summary><code>MinimumBarrierSaliency</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewMinimumBarrierSaliency() *MinimumBarrierSaliency` | NewMinimumBarrierSaliency returns a detector with three scan passes and a smoothing radius of three. |
| `func (m *MinimumBarrierSaliency) ComputeSaliency(img *cv.Mat) *cv.Mat` | ComputeSaliency returns the minimum-barrier saliency map of img: a single-channel `cv.Mat` the same size as img, normalised to [0,255]. |

</details>

<details>
<summary><code>MotionSaliencyBinWangApr2014</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewMotionSaliencyBinWangApr2014(rows, cols int) *MotionSaliencyBinWangApr2014` | NewMotionSaliencyBinWangApr2014 returns a detector sized for rows×cols frames with sensible defaults (four templates, an intensity tolerance of 40… |
| `func (m *MotionSaliencyBinWangApr2014) ComputeSaliency(img *cv.Mat) *cv.Mat` | ComputeSaliency returns the binary motion-saliency map for one frame: a single-channel `cv.Mat` the same size as the configured frame size, with… |
| `func (m *MotionSaliencyBinWangApr2014) Reset()` | Reset discards the learned background model so the next frame re-seeds it. |

</details>

<details>
<summary><code>ObjectnessBING</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewObjectnessBING() *ObjectnessBING` | NewObjectnessBING returns a detector with a default range of window sizes and a cap of 64 proposals. |
| `func (o *ObjectnessBING) ComputeObjectness(img *cv.Mat) []ObjectnessBox` | ComputeObjectness returns candidate object windows for img ranked by objectness score (highest first). |

</details>

<details>
<summary><code>ObjectnessCascade</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewObjectnessCascade() *ObjectnessCascade` | NewObjectnessCascade returns a cascade with a default range of window sizes, 256 stage-one survivors, an IoU suppression threshold of 0.5 and a cap… |
| `func (o *ObjectnessCascade) ComputeObjectness(img *cv.Mat) []ObjectnessBox` | ComputeObjectness returns candidate object windows for img ranked by objectness score (highest first) after two-stage scoring and non-maximum… |

</details>

<details>
<summary><code>RegionContrast</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewRegionContrast() *RegionContrast` | NewRegionContrast returns a detector with a 12×12 region grid. |
| `func (rc *RegionContrast) ComputeSaliency(img *cv.Mat) *cv.Mat` | ComputeSaliency returns the region-contrast saliency map of img: a single-channel `cv.Mat` the same size as img, normalised to [0,255]. |

</details>

<details>
<summary><code>StaticSaliencyBooleanMap</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewStaticSaliencyBooleanMap() *StaticSaliencyBooleanMap` | NewStaticSaliencyBooleanMap returns a detector with eight threshold levels. |
| `func (s *StaticSaliencyBooleanMap) ComputeSaliency(img *cv.Mat) *cv.Mat` | ComputeSaliency returns the Boolean-map saliency of img: a single-channel `cv.Mat` the same size as img, normalised to [0,255]. |

</details>

<details>
<summary><code>StaticSaliencyContextAware</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewStaticSaliencyContextAware() *StaticSaliencyContextAware` | NewStaticSaliencyContextAware returns a detector with a 48×48 working size and K=32 similar references. |
| `func (s *StaticSaliencyContextAware) ComputeSaliency(img *cv.Mat) *cv.Mat` | ComputeSaliency returns the context-aware saliency map of img: a single-channel `cv.Mat` the same size as img, normalised to [0,255]. |

</details>

<details>
<summary><code>StaticSaliencyFineGrained</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewStaticSaliencyFineGrained() *StaticSaliencyFineGrained` | NewStaticSaliencyFineGrained returns a detector configured with the OpenCV default of six scales. |
| `func (f *StaticSaliencyFineGrained) ComputeSaliency(img *cv.Mat) *cv.Mat` | ComputeSaliency returns the fine-grained saliency map of img, a single-channel `cv.Mat` the same size as img and normalised to [0,255]. |

</details>

<details>
<summary><code>StaticSaliencyFrequencyTuned</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewStaticSaliencyFrequencyTuned() *StaticSaliencyFrequencyTuned` | NewStaticSaliencyFrequencyTuned returns a detector with a 5×5 pre-blur. |
| `func (s *StaticSaliencyFrequencyTuned) ComputeSaliency(img *cv.Mat) *cv.Mat` | ComputeSaliency returns the frequency-tuned saliency map of img: a single-channel `cv.Mat` the same size as img, normalised to [0,255]. |

</details>

<details>
<summary><code>StaticSaliencyIttiKochNiebur</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewStaticSaliencyIttiKochNiebur() *StaticSaliencyIttiKochNiebur` | NewStaticSaliencyIttiKochNiebur returns a detector configured with the classical center scales {2,3}, surround deltas {2,3} and a seven-level pyramid. |
| `func (s *StaticSaliencyIttiKochNiebur) ComputeSaliency(img *cv.Mat) *cv.Mat` | ComputeSaliency returns the Itti-Koch-Niebur saliency map of img: a single-channel `cv.Mat` the same size as img, normalised to [0,255]. |

</details>

<details>
<summary><code>StaticSaliencySpectralResidual</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewStaticSaliencySpectralResidual() *StaticSaliencySpectralResidual` | NewStaticSaliencySpectralResidual returns a detector configured with the OpenCV default 64×64 working resolution. |
| `func (s *StaticSaliencySpectralResidual) ComputeSaliency(img *cv.Mat) *cv.Mat` | ComputeSaliency returns the spectral-residual saliency map of img, a single-channel `cv.Mat` the same size as img and normalised to [0,255]. |

</details>

Full signatures, doc comments and every runnable example are on
[pkg.go.dev](https://pkg.go.dev/github.com/malcolmston/opencv/saliency).

## License

MIT, as part of [`github.com/malcolmston/opencv`](..).
An independent re-implementation, not affiliated with or endorsed by the
original project.
