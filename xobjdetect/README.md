# xobjdetect — pure-Go port of OpenCV's xobjdetect module

[![Go Reference](https://pkg.go.dev/badge/github.com/malcolmston/opencv/xobjdetect.svg)](https://pkg.go.dev/github.com/malcolmston/opencv/xobjdetect)

Package xobjdetect is a pure-Go port of OpenCV's xobjdetect module: a
WaldBoost object detector built on integral-channel (ACF-style) features. It
depends only on the Go standard library and the root
`github.com/malcolmston/opencv` package (imported as cv); there is no cgo and
there are no third-party dependencies.

The module trains a boosted ensemble of decision stumps over integral channel
features and evaluates it with a sequential-probability-ratio-test (SPRT)
early-exit cascade, exactly the recipe behind OpenCV's
cv::xobjdetect::WBDetector and its ICFDetector / ACFFeatureEvaluator
machinery.

## Install

```sh
go get github.com/malcolmston/opencv@v0.8.0
```

```go
import "github.com/malcolmston/opencv/xobjdetect"
```

## Usage

This is the package's own `Example`, so it compiles and its output is
asserted on every `go test ./xobjdetect/`.

```go
rng := rand.New(rand.NewSource(1))
	target := func(r *rand.Rand) *cv.Mat {
		m := cv.NewMat(24, 24, 1)
		for y := 0; y < 24; y++ {
			for x := 0; x < 24; x++ {
				v := 30 + r.Intn(11)
				if x >= 6 && x < 18 && y >= 6 && y < 18 {
					v = 200 + r.Intn(41)
				}
				m.Set(y, x, 0, uint8(v))
			}
		}
		return m
	}
	var pos, neg []*cv.Mat
	for i := 0; i < 30; i++ {
		pos = append(pos, target(rng))
		p := cv.NewMat(24, 24, 1)
		for j := range p.Data {
			p.Data[j] = uint8(90 + rng.Intn(80))
		}
		neg = append(neg, p)
	}

	d := xobjdetect.NewWBDetector()
	d.Rounds = 30
	d.NumFeatures = 150
	if err := d.Train(pos, neg); err != nil {
		panic(err)
	}

	scene := cv.NewMat(80, 80, 1)
	for i := range scene.Data {
		scene.Data[i] = 35
	}
	target(rand.New(rand.NewSource(2))).CopyTo(scene, 30, 40)

	rects, _ := d.Detect(scene)
	fmt.Printf("detections=%d\n", len(rects))
```

```
detections=35
```

## Exported surface

### Types

| Type | What it is |
| --- | --- |
| `ACFFeatureEvaluator` | ACFFeatureEvaluator computes aggregate-channel-feature (ACF) vectors for a `FeaturePool`, mirroring OpenCV's cv::xobjdetect::ACFFeatureEvaluator. |
| `Feature` | Feature is one integral-channel probe: the mean value of channel Channel over the axis-aligned rectangle (X, Y, W, H) measured in detection-window… |
| `FeaturePool` | FeaturePool is a bank of `Feature` probes covering a fixed WinW x WinH detection window. |
| `Stump` | Stump is a confidence-rated decision stump: the atomic weak classifier of a `WaldBoost` ensemble. |
| `WBDetector` | WBDetector is a WaldBoost object detector, the pure-Go counterpart of OpenCV's cv::xobjdetect::WBDetector. |
| `WaldBoost` | WaldBoost is a WaldBoost classifier: a discrete-AdaBoost ensemble of `Stump`s equipped with a sequential-probability-ratio-test (SPRT) early-exit… |

<details>
<summary><code>ACFFeatureEvaluator</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewACFFeatureEvaluator(pool *FeaturePool) *ACFFeatureEvaluator` | NewACFFeatureEvaluator returns an evaluator bound to pool. |
| `func (e *ACFFeatureEvaluator) EvaluateWindow(ox, oy int) []float64` | EvaluateWindow returns the feature vector for the detection window whose top-left corner is (ox, oy) in the image most recently passed to… |
| `func (e *ACFFeatureEvaluator) Pool() *FeaturePool` | Pool returns the feature pool the evaluator was built with. |
| `func (e *ACFFeatureEvaluator) Sample(img *cv.Mat) []float64` | Sample resizes img to the pool's detection window and returns its feature vector. |
| `func (e *ACFFeatureEvaluator) SetImage(img *cv.Mat)` | SetImage precomputes the integral feature channels of img so subsequent `ACFFeatureEvaluator.EvaluateWindow` calls are cheap. |

</details>

<details>
<summary><code>FeaturePool</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewFeaturePool(winW, winH, count int, rng *rand.Rand) *FeaturePool` | NewFeaturePool builds a pool of count random features covering a WinW x WinH window. |
| `func (fp *FeaturePool) Len() int` | Len reports the number of features in the pool, which is the length of every feature vector it produces. |

</details>

<details>
<summary><code>WBDetector</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewWBDetector() *WBDetector` | NewWBDetector returns a detector with the default 24x24 window and default tuning. |
| `func NewWBDetectorSize(winW, winH int) *WBDetector` | NewWBDetectorSize returns a detector whose detection window is winW x winH. It panics if either dimension is not positive. |
| `func (d *WBDetector) Detect(img *cv.Mat) (rects []cv.Rect, confidences []float64)` | Detect scans img and returns the bounding boxes of every detected object together with the confidence score of each, sorted by descending score. |
| `func (d *WBDetector) Evaluator() *ACFFeatureEvaluator` | Evaluator returns an `ACFFeatureEvaluator` bound to the detector's feature pool. |
| `func (d *WBDetector) Read(r io.Reader) error` | Read restores a detector previously written by `WBDetector.Write` from r, replacing the receiver's state. |
| `func (d *WBDetector) Train(posSamples, negSamples []*cv.Mat) error` | Train learns the detector from positive samples (patches containing the object) and negative samples (patches that do not). |
| `func (d *WBDetector) Trained() bool` | Trained reports whether the detector has been trained (directly or by loading a model). |
| `func (d *WBDetector) Write(w io.Writer) error` | Write serialises a trained detector to w with encoding/gob. |

</details>

<details>
<summary><code>WaldBoost</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewWaldBoost(rounds int) *WaldBoost` | NewWaldBoost returns a WaldBoost that fits up to rounds stumps. |
| `func (w *WaldBoost) Predict(feat []float64) (score float64, accepted bool)` | Predict runs the SPRT cascade over feat and returns the accumulated score and whether the sample was accepted. |
| `func (w *WaldBoost) Train(pos, neg [][]float64) error` | Train fits the ensemble from positive and negative feature vectors. |

</details>

### Constants

`NumChannels`, `NumOrientBins`

Full signatures, doc comments and every runnable example are on
[pkg.go.dev](https://pkg.go.dev/github.com/malcolmston/opencv/xobjdetect).

## License

MIT, as part of [`github.com/malcolmston/opencv`](..).
An independent re-implementation, not affiliated with or endorsed by the
original project.
