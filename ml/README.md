# ml — from-scratch, standard-library-only port of a useful subset of OpenCV's ml module

[![Go Reference](https://pkg.go.dev/badge/github.com/malcolmston/opencv/ml.svg)](https://pkg.go.dev/github.com/malcolmston/opencv/ml)

Package ml is a from-scratch, standard-library-only port of a useful subset of
OpenCV's ml module: classic (non-neural) machine-learning models for
statistical classification, regression and clustering.

The package operates on plain Go data. A training set is a slice of samples,
where each sample is a feature vector [][]float64 (rows are observations,
columns are features), paired with []int class labels (for classifiers) or
[]float64 responses (for regressors). This keeps the API independent of the
image-oriented `cv.Mat` type; when your data already lives in a Mat, convert
it once with `MatToSamples` and feed the result to any model here.

## Install

```sh
go get github.com/malcolmston/opencv@v0.8.0
```

```go
import "github.com/malcolmston/opencv/ml"
```

## Usage

This is the package's own `ExampleANNMLP`, so it compiles and its output is
asserted on every `go test ./ml/`.

```go
samples := [][]float64{{0, 0}, {1, 1}, {0, 1}, {1, 0}}
	labels := []int{1, 1, 0, 0}

	net := ml.NewANNMLP(8)
	net.Activation = ml.Tanh
	net.LearningRate = 0.5
	net.Epochs = 4000
	_ = net.Train(samples, labels)

	fmt.Println(ml.Accuracy(net.PredictBatch(samples), labels))
```

```
1
```

## Exported surface

### Functions

| Function | What it does |
| --- | --- |
| `func AUC(scores []float64, actual []int, positive int) float64` | AUC returns the area under the ROC curve for a binary scorer, computed directly from the Mann-Whitney U statistic (the probability that a random… |
| `func AUCFromCurve(fpr, tpr []float64) float64` | AUCFromCurve returns the area under an ROC curve given as parallel fpr/tpr slices (as produced by `ROCCurve`), integrated with the trapezoidal rule. |
| `func Accuracy(predicted, actual []int) float64` | Accuracy returns the fraction of predictions that match the ground truth, a value in [0, 1]. |
| `func ConfusionMatrix(predicted, actual []int, numClasses int) [][]int` | ConfusionMatrix returns a numClasses×numClasses table whose entry `a``p` counts the samples whose actual label is a and whose predicted label is p. |
| `func CrossValScore(model Classifier, td *TrainData, k int, seed int64) []float64` | CrossValScore evaluates a classifier by k-fold cross-validation and returns the held-out accuracy on each fold, in fold order. |
| `func F1Score(predicted, actual []int, positive int) float64` | F1Score returns the harmonic mean of `Precision` and `Recall` for the class positive. |
| `func KMeans(data [][]float64, k, maxIter int, seed int64) (labels []int, centers [][]float64)` | KMeans partitions data into k clusters with Lloyd's algorithm using k-means++ seeding. |
| `func Load(r io.Reader, model any) error` | Load gob-decodes a model from r into model, which must be a non-nil pointer. |
| `func LoadFile(path string, model any) error` | LoadFile reads a gob-encoded model from the named file into model, which must be a non-nil pointer of the matching type. |
| `func MAE(predicted, actual []float64) float64` | MAE returns the mean absolute error between predicted and actual regression targets. |
| `func MSE(predicted, actual []float64) float64` | MSE returns the mean squared error between predicted and actual regression targets. |
| `func MacroF1(predicted, actual []int) float64` | MacroF1 returns the unweighted mean of the per-class `F1Score` over every label present in the data. |
| `func MacroPrecision(predicted, actual []int) float64` | MacroPrecision returns the unweighted mean of the per-class `Precision` over every label present in the data. |
| `func MacroRecall(predicted, actual []int) float64` | MacroRecall returns the unweighted mean of the per-class `Recall` over every label present in the data. |
| `func MatToSamples(m *cv.Mat) [][]float64` | MatToSamples converts a single-channel `cv.Mat` whose rows are observations and columns are features into a [][]float64 feature matrix. |
| `func MeanScore(scores []float64) float64` | MeanScore returns the arithmetic mean of scores, a convenience for summarising the per-fold results of `CrossValScore`. |
| `func Precision(predicted, actual []int, positive int) float64` | Precision returns the precision TP/(TP+FP) for the class positive, treating every other label as negative. |
| `func R2Score(predicted, actual []float64) float64` | R2Score returns the coefficient of determination R², the fraction of variance in actual explained by predicted. |
| `func RMSE(predicted, actual []float64) float64` | RMSE returns the square root of `MSE`. |
| `func ROCCurve(scores []float64, actual []int, positive int) (fpr, tpr []float64)` | ROCCurve computes the receiver-operating-characteristic curve for a binary scorer. |
| `func Recall(predicted, actual []int, positive int) float64` | Recall returns the recall TP/(TP+FN) for the class positive, treating every other label as negative. |
| `func Save(w io.Writer, model any) error` | Save gob-encodes model to w. |
| `func SaveFile(path string, model any) error` | SaveFile gob-encodes model and writes it to the named file, truncating any existing contents. |

### Types

| Type | What it is |
| --- | --- |
| `ANNMLP` | ANNMLP is a feed-forward artificial neural network (multilayer perceptron) trained by error back-propagation, mirroring OpenCV's cv::ml::ANN_MLP. |
| `Activation` | Activation selects the non-linearity applied to an `ANNMLP`'s hidden layers. |
| `Boost` | Boost is a boosted ensemble of decision stumps, mirroring OpenCV's cv::ml::Boost. |
| `Classifier` | Classifier is the common interface implemented by every supervised classifier in this package: `KNearest`, `SVM`, `KernelSVM`,… |
| `DecisionTree` | DecisionTree is a CART classification tree. |
| `Fold` | Fold is one train/test partition produced by `TrainData.KFold`. |
| `GaussianMixture` | GaussianMixture is a Gaussian mixture model fitted by the Expectation-Maximization (EM) algorithm, mirroring OpenCV's cv::ml::EM. |
| `KNearest` | KNearest is a k-nearest-neighbours classifier. |
| `KernelSVM` | KernelSVM is a kernelised support-vector machine, extending the linear `SVM` with non-linear decision boundaries via the kernel trick. |
| `KernelType` | KernelType selects the kernel function of a `KernelSVM`. |
| `LogisticRegression` | LogisticRegression is a multinomial (softmax) logistic-regression classifier trained by batch gradient descent on the cross-entropy loss with… |
| `MinMaxScaler` | MinMaxScaler rescales each feature linearly to the range [0, 1] based on the minimum and maximum seen during Fit. |
| `NormalBayesClassifier` | NormalBayesClassifier is a Gaussian naive Bayes classifier. |
| `RTrees` | RTrees is a random forest of CART classification trees, mirroring OpenCV's cv::ml::RTrees. |
| `SVM` | SVM is a linear soft-margin support-vector machine. |
| `StandardScaler` | StandardScaler standardises features to zero mean and unit variance, the exported counterpart of the standardisation the models apply internally. |
| `TrainData` | TrainData bundles a feature matrix with the labels (for classification) and responses (for regression) that go with it, mirroring the role of… |

<details>
<summary><code>ANNMLP</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewANNMLP(hidden ...int) *ANNMLP` | NewANNMLP returns a network with one hidden layer of the given width, tanh activation, and sensible training defaults (learning rate 0.1, 1000… |
| `func (m *ANNMLP) Classes() []int` | Classes returns the sorted set of class labels the network was trained on. |
| `func (m *ANNMLP) GobDecode(data []byte) error` | GobDecode implements gob.GobDecoder. |
| `func (m *ANNMLP) GobEncode() ([]byte, error)` | GobEncode implements gob.GobEncoder. |
| `func (m *ANNMLP) Predict(sample []float64) int` | Predict classifies a single sample. |
| `func (m *ANNMLP) PredictBatch(samples [][]float64) []int` | PredictBatch classifies every sample and returns the predicted labels in order. |
| `func (m *ANNMLP) Probabilities(sample []float64) []float64` | Probabilities returns the softmax class probabilities for a sample, in sorted class order (see `ANNMLP.Classes`). |
| `func (m *ANNMLP) Train(samples [][]float64, labels []int) error` | Train fits the network weights by full-batch back-propagation. |

</details>

<details>
<summary><code>Boost</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewBoost(nEstimators int) *Boost` | NewBoost returns a boosting ensemble that fits up to nEstimators stumps. |
| `func (m *Boost) Classes() []int` | Classes returns the sorted set of class labels the ensemble was trained on. |
| `func (m *Boost) DecisionScores(sample []float64) []float64` | DecisionScores returns the accumulated stump weight backing each class, in sorted class order (see `Boost.Classes`). |
| `func (m *Boost) GobDecode(data []byte) error` | GobDecode implements gob.GobDecoder. |
| `func (m *Boost) GobEncode() ([]byte, error)` | GobEncode implements gob.GobEncoder. |
| `func (m *Boost) Predict(sample []float64) int` | Predict classifies a single sample as the class with the greatest summed stump weight. |
| `func (m *Boost) PredictBatch(samples [][]float64) []int` | PredictBatch classifies every sample and returns the predicted labels in order. |
| `func (m *Boost) Train(samples [][]float64, labels []int) error` | Train fits the boosted ensemble with AdaBoost.SAMME. |

</details>

<details>
<summary><code>DecisionTree</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewDecisionTree(maxDepth int) *DecisionTree` | NewDecisionTree returns a CART tree bounded to maxDepth (0 for unbounded). |
| `func (m *DecisionTree) Predict(sample []float64) int` | Predict classifies a single sample. |
| `func (m *DecisionTree) PredictBatch(samples [][]float64) []int` | PredictBatch classifies every sample and returns the predicted labels in order. |
| `func (m *DecisionTree) Train(samples [][]float64, labels []int) error` | Train grows the tree from samples and labels. |

</details>

<details>
<summary><code>GaussianMixture</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewGaussianMixture(k int) *GaussianMixture` | NewGaussianMixture returns an EM model with k components and sensible defaults (100 iterations, tolerance 1e-6). |
| `func (m *GaussianMixture) Fit(data [][]float64) error` | Fit estimates the mixture parameters from unlabelled data with EM. |
| `func (m *GaussianMixture) GobDecode(data []byte) error` | GobDecode implements gob.GobDecoder. |
| `func (m *GaussianMixture) GobEncode() ([]byte, error)` | GobEncode implements gob.GobEncoder. |
| `func (m *GaussianMixture) LogLikelihood(sample []float64) float64` | LogLikelihood returns the log probability density of sample under the fitted mixture. |
| `func (m *GaussianMixture) Means() [][]float64` | Means returns a copy of the fitted component means. |
| `func (m *GaussianMixture) Predict(sample []float64) int` | Predict returns the index of the component most responsible for sample, a soft-clustering assignment in [0, K). |
| `func (m *GaussianMixture) PredictBatch(samples [][]float64) []int` | PredictBatch assigns every sample to its most probable component. |
| `func (m *GaussianMixture) Responsibilities(sample []float64) []float64` | Responsibilities returns the posterior probability that sample belongs to each component, in component order. |
| `func (m *GaussianMixture) TotalLogLikelihood() float64` | TotalLogLikelihood returns the summed data log-likelihood attained at the end of training. |
| `func (m *GaussianMixture) Weights() []float64` | Weights returns a copy of the fitted mixture weights. |

</details>

<details>
<summary><code>KNearest</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewKNearest(k int) *KNearest` | NewKNearest returns a KNearest that consults k neighbours with a plain majority vote. |
| `func (m *KNearest) FindNearest(sample []float64, k int) (labels []int, dists []float64)` | FindNearest returns the labels and Euclidean distances of the k training samples closest to sample, ordered nearest first. |
| `func (m *KNearest) Predict(sample []float64) int` | Predict classifies a single sample. |
| `func (m *KNearest) PredictBatch(samples [][]float64) []int` | PredictBatch classifies every sample and returns the predicted labels in order. |
| `func (m *KNearest) Train(samples [][]float64, labels []int) error` | Train memorises the training samples and their labels. |

</details>

<details>
<summary><code>KernelSVM</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewKernelSVM(kernel KernelType) *KernelSVM` | NewKernelSVM returns a kernel SVM using the given kernel with sensible defaults (Lambda 1e-3, 50 epochs, degree 3, coef0 1). |
| `func (m *KernelSVM) Classes() []int` | Classes returns the sorted set of class labels the model was trained on. |
| `func (m *KernelSVM) DecisionScores(sample []float64) []float64` | DecisionScores returns the one-vs-rest decision value for every class, in sorted class order (see `KernelSVM.Classes`). |
| `func (m *KernelSVM) GobDecode(data []byte) error` | GobDecode implements gob.GobDecoder. |
| `func (m *KernelSVM) GobEncode() ([]byte, error)` | GobEncode implements gob.GobEncoder. |
| `func (m *KernelSVM) Predict(sample []float64) int` | Predict classifies a single sample as the highest-scoring one-vs-rest class. |
| `func (m *KernelSVM) PredictBatch(samples [][]float64) []int` | PredictBatch classifies every sample and returns the predicted labels in order. |
| `func (m *KernelSVM) Train(samples [][]float64, labels []int) error` | Train fits one kernel-Pegasos binary machine per class in a one-vs-rest scheme. |

</details>

<details>
<summary><code>LogisticRegression</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewLogisticRegression() *LogisticRegression` | NewLogisticRegression returns a classifier with sensible defaults (learning rate 0.1, 500 epochs, no L2 penalty). |
| `func (m *LogisticRegression) Classes() []int` | Classes returns the sorted set of class labels the model was trained on. |
| `func (m *LogisticRegression) Predict(sample []float64) int` | Predict classifies a single sample. |
| `func (m *LogisticRegression) PredictBatch(samples [][]float64) []int` | PredictBatch classifies every sample and returns the predicted labels in order. |
| `func (m *LogisticRegression) Probabilities(sample []float64) []float64` | Probabilities returns the predicted class probabilities for a sample, in sorted class order (see `LogisticRegression.Classes`). |
| `func (m *LogisticRegression) Train(samples [][]float64, labels []int) error` | Train fits the softmax weights by gradient descent. |

</details>

<details>
<summary><code>MinMaxScaler</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func (s *MinMaxScaler) Fit(samples [][]float64) *MinMaxScaler` | Fit learns the per-feature minimum and maximum from samples and returns the receiver for chaining. |
| `func (s *MinMaxScaler) FitTransform(samples [][]float64) [][]float64` | FitTransform fits the scaler on samples and returns the rescaled samples in one call. |
| `func (s *MinMaxScaler) Transform(x []float64) []float64` | Transform rescales a single feature vector to [0, 1] into a fresh slice. |
| `func (s *MinMaxScaler) TransformAll(samples [][]float64) [][]float64` | TransformAll rescales every sample and returns a new matrix. |

</details>

<details>
<summary><code>NormalBayesClassifier</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewNormalBayesClassifier() *NormalBayesClassifier` | NewNormalBayesClassifier returns an untrained Gaussian naive Bayes classifier. |
| `func (m *NormalBayesClassifier) Predict(sample []float64) int` | Predict classifies a single sample. |
| `func (m *NormalBayesClassifier) PredictBatch(samples [][]float64) []int` | PredictBatch classifies every sample and returns the predicted labels in order. |
| `func (m *NormalBayesClassifier) Train(samples [][]float64, labels []int) error` | Train estimates the class priors and the per-class Gaussian parameters. |

</details>

<details>
<summary><code>RTrees</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewRTrees(nTrees int) *RTrees` | NewRTrees returns a random forest with nTrees trees and otherwise sensible defaults (unbounded depth, sqrt feature subsampling). |
| `func (m *RTrees) Classes() []int` | Classes returns the sorted set of class labels the forest was trained on. |
| `func (m *RTrees) GobDecode(data []byte) error` | GobDecode implements gob.GobDecoder. |
| `func (m *RTrees) GobEncode() ([]byte, error)` | GobEncode implements gob.GobEncoder. |
| `func (m *RTrees) OOBError() float64` | OOBError returns the out-of-bag misclassification rate estimated during Train, a value in [0, 1]. |
| `func (m *RTrees) Predict(sample []float64) int` | Predict classifies a single sample by majority vote across all trees. |
| `func (m *RTrees) PredictBatch(samples [][]float64) []int` | PredictBatch classifies every sample and returns the predicted labels in order. |
| `func (m *RTrees) Train(samples [][]float64, labels []int) error` | Train grows the forest. |

</details>

<details>
<summary><code>SVM</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewSVM() *SVM` | NewSVM returns an SVM with sensible defaults (Lambda 1e-3, 200 epochs). |
| `func (m *SVM) Classes() []int` | Classes returns the sorted set of class labels the model was trained on. |
| `func (m *SVM) DecisionScores(sample []float64) []float64` | DecisionScores returns the raw one-vs-rest decision value for every class, in the model's sorted class order (see `SVM.Classes`). |
| `func (m *SVM) Predict(sample []float64) int` | Predict classifies a single sample, returning the label of the highest- scoring one-vs-rest machine. |
| `func (m *SVM) PredictBatch(samples [][]float64) []int` | PredictBatch classifies every sample and returns the predicted labels in order. |
| `func (m *SVM) Train(samples [][]float64, labels []int) error` | Train fits one Pegasos binary machine per class in a one-vs-rest scheme. |

</details>

<details>
<summary><code>StandardScaler</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func (s *StandardScaler) Fit(samples [][]float64) *StandardScaler` | Fit learns the per-feature mean and standard deviation from samples and returns the receiver for chaining. |
| `func (s *StandardScaler) FitTransform(samples [][]float64) [][]float64` | FitTransform fits the scaler on samples and returns the standardised samples in one call. |
| `func (s *StandardScaler) Transform(x []float64) []float64` | Transform standardises a single feature vector into a fresh slice. |
| `func (s *StandardScaler) TransformAll(samples [][]float64) [][]float64` | TransformAll standardises every sample and returns a new matrix. |

</details>

<details>
<summary><code>TrainData</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewRegressionData(samples [][]float64, responses []float64) *TrainData` | NewRegressionData builds a TrainData from samples and real-valued responses. |
| `func NewTrainData(samples [][]float64, labels []int) *TrainData` | NewTrainData builds a TrainData from samples and integer class labels. |
| `func (td *TrainData) KFold(k int, seed int64) []Fold` | KFold partitions the data into k folds for cross-validation. |
| `func (td *TrainData) Len() int` | Len returns the number of samples. |
| `func (td *TrainData) Shuffle(seed int64) *TrainData` | Shuffle returns a new TrainData with the samples (and their labels/responses) randomly permuted by a seeded generator. |
| `func (td *TrainData) Split(trainRatio float64, seed int64) (train, test *TrainData)` | Split partitions the data into a training and a test set. |
| `func (td *TrainData) StratifiedSplit(trainRatio float64, seed int64) (train, test *TrainData)` | StratifiedSplit partitions labelled data into a training and a test set while preserving each class's proportion in both parts. |

</details>

### Variables

`ErrNoSamples`, `ErrLabelMismatch`, `ErrRaggedSamples`, `ErrNotTrained`

Full signatures, doc comments and every runnable example are on
[pkg.go.dev](https://pkg.go.dev/github.com/malcolmston/opencv/ml).

## License

MIT, as part of [`github.com/malcolmston/opencv`](..).
An independent re-implementation, not affiliated with or endorsed by the
original project.
