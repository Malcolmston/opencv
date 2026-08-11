# face — from-scratch, standard-library-only port of a useful subset of OpenCV's contrib face

[![Go Reference](https://pkg.go.dev/badge/github.com/malcolmston/opencv/face.svg)](https://pkg.go.dev/github.com/malcolmston/opencv/face)

Package face is a from-scratch, standard-library-only port of a useful subset
of OpenCV's contrib face module: classic (non-neural) face recognition. It
implements the three canonical recognizers — Eigenfaces, Fisherfaces and
Local Binary Pattern Histograms — plus the underlying Local Binary Pattern
operator, using only the Go standard library and the root
`github.com/malcolmston/opencv` package. There is no cgo and there are no
third-party dependencies.

Beyond recognition the package now also provides a self-contained,
integral-image Haar face *detector* (`GetFacesHAAR`), a trainable facial
*landmark* localiser in the FacemarkLBF spirit (`FacemarkLBF`), a
biologically-inspired feature descriptor (`BIF`) and a Minimum Average
Correlation Energy filter (`MACE`) — so a full pipeline (detect, align by
landmarks, describe, verify or identify) can be built without leaving this
package. The objdetect subpackage still hosts a general Haar cascade
classifier for production detection.

## Install

```sh
go get github.com/malcolmston/opencv@v0.8.0
```

```go
import "github.com/malcolmston/opencv/face"
```

## Usage

This is the package's own `ExampleEigenFaceRecognizer`, so it compiles and its output is
asserted on every `go test ./face/`.

```go
imgs := []*cv.Mat{gradient(0), gradient(0), gradient(1), gradient(1)}
	labels := []int{0, 0, 1, 1}

	r := face.NewEigenFaceRecognizer(0)
	r.Train(imgs, labels)

	label, _ := r.Predict(gradient(0))
	fmt.Println(label)
```

```
0
```

## Exported surface

### Functions

| Function | What it does |
| --- | --- |
| `func GetFacesHAAR(img *cv.Mat, params *HaarParams) []cv.Rect` | GetFacesHAAR detects upright, face-like regions in img and returns their bounding rectangles. |
| `func LBP(img *cv.Mat) *cv.Mat` | LBP computes the basic 3×3 Local Binary Pattern code image of img. |
| `func LBPCircular(img *cv.Mat, radius, neighbours int) *cv.Mat` | LBPCircular computes the extended, circular Local Binary Pattern code image of img with the given radius and neighbour count. |
| `func LBPUniform(img *cv.Mat) *cv.Mat` | LBPUniform computes the uniform-pattern LBP label image of img. |
| `func LBPUniformRotInvariant(img *cv.Mat) *cv.Mat` | LBPUniformRotInvariant computes the rotation-invariant uniform LBP (the "riu2" operator) of img on the classic radius-1, 8-neighbour circle. |

### Types

| Type | What it is |
| --- | --- |
| `BIF` | BIF computes Biologically Inspired Features. |
| `EigenFaceRecognizer` | EigenFaceRecognizer implements the classic Eigenfaces method (Turk & Pentland, 1991). |
| `FaceRecognizer` | FaceRecognizer is the common interface implemented by every recognizer in this package: `EigenFaceRecognizer`, `FisherFaceRecognizer` and… |
| `FacemarkLBF` | FacemarkLBF is a trainable facial-landmark localiser. |
| `FisherFaceRecognizer` | FisherFaceRecognizer implements the Fisherfaces method (Belhumeur, Hespanha & Kriegman, 1997): a principal-component analysis for dimensionality… |
| `HaarParams` | HaarParams tunes the `GetFacesHAAR` sliding-window search. |
| `LBPHFaceRecognizer` | LBPHFaceRecognizer implements Local Binary Pattern Histograms face recognition (Ahonen, Hadid & Pietikäinen, 2004). |
| `MACE` | MACE is a synthesised Minimum Average Correlation Energy correlation filter. |
| `PredictedFace` | PredictedFace is a single scored candidate produced by the PredictCollect family: the training-set Label and the Distance from the query to that… |

<details>
<summary><code>BIF</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewBIF(scales, orients, grid int) *BIF` | NewBIF returns a BIF descriptor with the given number of scales and orientations and a grid×grid spatial pooling grid. |
| `func (b *BIF) Compute(img *cv.Mat) []float64` | Compute reduces img to luma and returns its BIF descriptor. |
| `func (b *BIF) FeatureLength() int` | FeatureLength returns the length of the vector `BIF.Compute` produces: one value per (band, orientation, grid cell), where the number of bands is one… |

</details>

<details>
<summary><code>EigenFaceRecognizer</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewEigenFaceRecognizer(numComponents int) *EigenFaceRecognizer` | NewEigenFaceRecognizer returns an untrained recognizer that keeps at most numComponents eigenfaces. |
| `func (r *EigenFaceRecognizer) Dims() (rows, cols int)` | Dims returns the (rows, cols) geometry every face is resampled to before projection, i.e. |
| `func (r *EigenFaceRecognizer) EigenFaceImage(i int) *cv.Mat` | EigenFaceImage renders the i-th eigenface as a viewable single-channel image of size rows×cols. |
| `func (r *EigenFaceRecognizer) EigenValues() []float64` | EigenValues returns the variance associated with each retained eigenface, in descending order. |
| `func (r *EigenFaceRecognizer) EigenVector(i int) []float64` | EigenVector returns a copy of the i-th eigenface as a flat vector of length rows*cols, ordered by descending variance. |
| `func (r *EigenFaceRecognizer) EigenVectors() [][]float64` | EigenVectors returns the retained eigenfaces as rows: a K×(rows*cols) matrix whose k-th row is the k-th unit-length principal axis in pixel space,… |
| `func (r *EigenFaceRecognizer) GetThreshold() float64` | GetThreshold returns the recognition threshold set by `EigenFaceRecognizer.SetThreshold`, or 0 (unbounded) if none was set. |
| `func (r *EigenFaceRecognizer) Load(rd io.Reader) error` | Load restores a recognizer previously written with `EigenFaceRecognizer.Save`, replacing the receiver's state. |
| `func (r *EigenFaceRecognizer) Mean() []float64` | Mean returns the average face vector learned during training (length rows*cols) as a copy. |
| `func (r *EigenFaceRecognizer) MeanFace() *cv.Mat` | MeanFace renders the learned mean face as a viewable single-channel image of size rows×cols, saturating each averaged sample into [0,255]. |
| `func (r *EigenFaceRecognizer) NumComponents() int` | NumComponents returns the number of eigenfaces retained after training (which may be fewer than requested if the data has lower rank). |
| `func (r *EigenFaceRecognizer) Predict(img *cv.Mat) (int, float64)` | Predict projects the query face into the eigenspace and returns the label of the nearest training projection along with the Euclidean distance to it… |
| `func (r *EigenFaceRecognizer) PredictCollect(img *cv.Mat) []PredictedFace` | PredictCollect returns every training sample scored against img, sorted by ascending distance in the eigenspace. |
| `func (r *EigenFaceRecognizer) PredictThreshold(img *cv.Mat) (int, float64)` | PredictThreshold behaves like Predict but honours the recognition threshold: if the nearest training distance exceeds the value set with… |
| `func (r *EigenFaceRecognizer) Project(img *cv.Mat) []float64` | Project returns the coordinates of img in the trained eigenspace (one coefficient per retained eigenface). |
| `func (r *EigenFaceRecognizer) Reconstruct(coeffs []float64) []float64` | Reconstruct rebuilds a face vector (length rows*cols) from projection coefficients, using the first len(coeffs) eigenfaces. |
| `func (r *EigenFaceRecognizer) ReconstructImage(coeffs []float64) *cv.Mat` | ReconstructImage rebuilds a face from projection coefficients (see `EigenFaceRecognizer.Reconstruct`) and renders it as a viewable single-channel… |
| `func (r *EigenFaceRecognizer) Save(w io.Writer) error` | Save writes the trained recognizer to w using encoding/gob. |
| `func (r *EigenFaceRecognizer) SetThreshold(t float64)` | SetThreshold sets the maximum acceptable prediction distance for this recognizer. |
| `func (r *EigenFaceRecognizer) Train(images []*cv.Mat, labels []int)` | Train fits the eigenspace to the labelled images. |

</details>

<details>
<summary><code>FacemarkLBF</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewFacemarkLBF(numLandmarks int) *FacemarkLBF` | NewFacemarkLBF returns an untrained localiser for numLandmarks landmarks. |
| `func NewFacemarkLBFParams(numLandmarks, stages, radius int) *FacemarkLBF` | NewFacemarkLBFParams returns an untrained localiser with an explicit number of cascade stages and patch radius (the sampling grid is (2*radius+1)… |
| `func (f *FacemarkLBF) Fit(img *cv.Mat, rect cv.Rect) []cv.Point` | Fit places the landmarks in img inside the face rectangle rect and returns them as image-coordinate points. |
| `func (f *FacemarkLBF) MeanShapeAt(rect cv.Rect) []cv.Point` | MeanShapeAt returns the learned mean shape placed inside rect, as image- coordinate points. |
| `func (f *FacemarkLBF) NumLandmarks() int` | NumLandmarks returns the number of landmarks the localiser predicts. |
| `func (f *FacemarkLBF) Train(images []*cv.Mat, rects []cv.Rect, shapes [][]cv.Point)` | Train fits the cascade to labelled faces. |

</details>

<details>
<summary><code>FisherFaceRecognizer</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewFisherFaceRecognizer(numComponents int) *FisherFaceRecognizer` | NewFisherFaceRecognizer returns an untrained recognizer. |
| `func (r *FisherFaceRecognizer) DiscriminantAxes() [][]float64` | DiscriminantAxes returns the retained Fisher discriminant directions mapped back into pixel space, as rows of a K×(rows*cols) matrix. |
| `func (r *FisherFaceRecognizer) FisherFaceImage(k int) *cv.Mat` | FisherFaceImage renders the k-th discriminant axis (see `FisherFaceRecognizer.DiscriminantAxes`) as a contrast-normalised, viewable single-channel… |
| `func (r *FisherFaceRecognizer) GetThreshold() float64` | GetThreshold returns the recognition threshold, or 0 if unset. |
| `func (r *FisherFaceRecognizer) Load(rd io.Reader) error` | Load restores a recognizer previously written with `FisherFaceRecognizer.Save`. |
| `func (r *FisherFaceRecognizer) NumComponents() int` | NumComponents returns the number of discriminant axes retained after training. |
| `func (r *FisherFaceRecognizer) Predict(img *cv.Mat) (int, float64)` | Predict projects the query through the PCA and LDA stages and returns the nearest training label with its Euclidean distance (lower is more… |
| `func (r *FisherFaceRecognizer) PredictCollect(img *cv.Mat) []PredictedFace` | PredictCollect returns every training sample scored against img in the discriminant subspace, sorted by ascending distance and threshold-filtered… |
| `func (r *FisherFaceRecognizer) PredictThreshold(img *cv.Mat) (int, float64)` | PredictThreshold behaves like Predict but returns `Unknown` when the nearest match is beyond the configured threshold. |
| `func (r *FisherFaceRecognizer) Project(img *cv.Mat) []float64` | Project returns the coordinates of img in the trained discriminant subspace. |
| `func (r *FisherFaceRecognizer) Save(w io.Writer) error` | Save writes the trained recognizer to w using encoding/gob. |
| `func (r *FisherFaceRecognizer) SetThreshold(t float64)` | SetThreshold sets the maximum acceptable prediction distance; see `EigenFaceRecognizer.SetThreshold`. |
| `func (r *FisherFaceRecognizer) Train(images []*cv.Mat, labels []int)` | Train fits the PCA-then-LDA pipeline to the labelled images. |

</details>

<details>
<summary><code>HaarParams</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func DefaultHaarParams() HaarParams` | DefaultHaarParams returns the parameters `GetFacesHAAR` uses when passed a nil configuration: a scan from 24-pixel windows up to the image size, a… |

</details>

<details>
<summary><code>LBPHFaceRecognizer</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewLBPHFaceRecognizer() *LBPHFaceRecognizer` | NewLBPHFaceRecognizer returns an untrained recognizer with the OpenCV default geometry: an 8×8 grid and full 256-bin (non-uniform) histograms. |
| `func NewLBPHFaceRecognizerWithParams(gridX, gridY int, uniform bool) *LBPHFaceRecognizer` | NewLBPHFaceRecognizerWithParams returns an untrained recognizer with an explicit grid and histogram type. |
| `func (r *LBPHFaceRecognizer) GetThreshold() float64` | GetThreshold returns the recognition threshold, or 0 if unset. |
| `func (r *LBPHFaceRecognizer) Load(rd io.Reader) error` | Load restores a recognizer previously written with `LBPHFaceRecognizer.Save`. |
| `func (r *LBPHFaceRecognizer) Predict(img *cv.Mat) (int, float64)` | Predict describes the query face and returns the label of the nearest stored histogram together with their chi-square distance (lower is more… |
| `func (r *LBPHFaceRecognizer) PredictCollect(img *cv.Mat) []PredictedFace` | PredictCollect returns every stored histogram scored against img under the chi-square distance, sorted ascending and threshold-filtered when a… |
| `func (r *LBPHFaceRecognizer) PredictThreshold(img *cv.Mat) (int, float64)` | PredictThreshold behaves like Predict but returns `Unknown` when the nearest match is beyond the configured threshold. |
| `func (r *LBPHFaceRecognizer) Save(w io.Writer) error` | Save writes the trained recognizer to w using encoding/gob. |
| `func (r *LBPHFaceRecognizer) SetThreshold(t float64)` | SetThreshold sets the maximum acceptable prediction distance; see `EigenFaceRecognizer.SetThreshold`. |
| `func (r *LBPHFaceRecognizer) Train(images []*cv.Mat, labels []int)` | Train computes and stores the spatial LBP histogram of every labelled image. |

</details>

<details>
<summary><code>MACE</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewMACE(size int) *MACE` | NewMACE returns an untrained MACE filter that operates on size×size images (every training and query image is reduced to luma and resampled to this… |
| `func (m *MACE) PSR(img *cv.Mat) float64` | PSR correlates img with the trained filter and returns the peak-to-sidelobe ratio of the correlation plane: the correlation peak minus the mean of… |
| `func (m *MACE) Same(img *cv.Mat) bool` | Same reports whether img is authentic for this filter, i.e. |
| `func (m *MACE) SetThreshold(psr float64)` | SetThreshold sets the peak-to-sidelobe ratio above which `MACE.Same` reports a query as authentic. |
| `func (m *MACE) Train(images []*cv.Mat)` | Train synthesises the correlation filter from one subject's images. |

</details>

### Constants

`Unknown`

### Variables

`ErrVersion`

Full signatures, doc comments and every runnable example are on
[pkg.go.dev](https://pkg.go.dev/github.com/malcolmston/opencv/face).

## License

MIT, as part of [`github.com/malcolmston/opencv`](..).
An independent re-implementation, not affiliated with or endorsed by the
original project.
