# ml2 — from-scratch, standard-library-only toolkit of classic (non-neural) machine-learning

[![Go Reference](https://pkg.go.dev/badge/github.com/malcolmston/opencv/ml2.svg)](https://pkg.go.dev/github.com/malcolmston/opencv/ml2)

Package ml2 is a from-scratch, standard-library-only toolkit of classic
(non-neural) machine-learning models aimed at computer-vision workflows:
classification, regression, clustering and dimensionality reduction over
feature vectors extracted from images.

The package operates on plain Go data. A dataset is a slice of feature vectors
[][]float64 (rows are observations, columns are features), paired with []int
class labels for classifiers or used unlabelled for clustering and projection.
This keeps the API independent of the image-oriented `cv.Mat` type while
remaining trivial to bridge to it: convert a single-channel data matrix with
`MatToSamples`, flatten one image into a single feature vector with
`MatToFeatureVector`, or turn a batch of same-sized images into a feature
matrix with `MatsToSamples`.

## Install

```sh
go get github.com/malcolmston/opencv@v0.8.0
```

```go
import "github.com/malcolmston/opencv/ml2"
```

## Usage

This is the package's own `ExampleAccuracy`, so it compiles and its output is
asserted on every `go test ./ml2/`.

```go
yTrue := []int{0, 0, 1, 1}
	yPred := []int{0, 1, 1, 1}
	fmt.Println(ml2.Accuracy(yTrue, yPred))
	fmt.Println(ml2.Precision(yTrue, yPred, 1), ml2.Recall(yTrue, yPred, 1))
	fmt.Println(ml2.ConfusionMatrix(yTrue, yPred, 2))
```

```
0.75
0.6666666666666666 1
[[1 1] [0 2]]
```

## Exported surface

### Functions

| Function | What it does |
| --- | --- |
| `func Accuracy(yTrue, yPred []int) float64` | Accuracy returns the fraction of predictions that match the true labels. |
| `func ConfusionMatrix(yTrue, yPred []int, numClasses int) [][]int` | ConfusionMatrix returns a numClasses-by-numClasses matrix whose entry `t``p` counts samples with true class t predicted as class p. |
| `func CrossValScore(newModel func() Classifier, samples [][]float64, labels []int, k int, seed int64) []float64` | CrossValScore runs k-fold cross-validation and returns the accuracy on each held-out fold. |
| `func F1Score(yTrue, yPred []int, class int) float64` | F1Score returns the harmonic mean of `Precision` and `Recall` for a single class, or zero when both are zero. |
| `func KFoldSplit(n, k int, seed int64) [][]int` | KFoldSplit partitions the indices [0, n) into k folds for cross-validation. |
| `func MacroF1(yTrue, yPred []int, numClasses int) float64` | MacroF1 returns the unweighted mean of the per-class `F1Score` over all numClasses classes. |
| `func MatToFeatureVector(m *cv.Mat) []float64` | MatToFeatureVector flattens an entire `cv.Mat` — every row, column and channel — into a single feature vector in row-major, channel-interleaved… |
| `func MatToSamples(m *cv.Mat) [][]float64` | MatToSamples converts a single-channel `cv.Mat` whose rows are observations and columns are features into a [][]float64 feature matrix. |
| `func MatsToSamples(mats []*cv.Mat) [][]float64` | MatsToSamples flattens each image in a batch into one feature row with `MatToFeatureVector`, producing a feature matrix suitable for training an… |
| `func MeanScore(scores []float64) float64` | MeanScore returns the arithmetic mean of a slice of scores, or zero for an empty slice. |
| `func Normalize(v []float64) []float64` | Normalize returns a copy of v scaled to unit L2 norm. |
| `func NormalizeRows(x [][]float64) [][]float64` | NormalizeRows applies `Normalize` to every row of x, returning a new matrix in which each row has unit L2 norm. |
| `func Precision(yTrue, yPred []int, class int) float64` | Precision returns the precision for a single class: true positives divided by all samples predicted to be that class. |
| `func Recall(yTrue, yPred []int, class int) float64` | Recall returns the recall for a single class: true positives divided by all samples that truly belong to that class. |
| `func Shuffle(samples [][]float64, labels []int, seed int64) (outX [][]float64, outY []int)` | Shuffle reorders samples and their labels together with a reproducible permutation seeded by seed, returning new slices and leaving the inputs… |
| `func TrainTestSplit(samples [][]float64, labels []int, trainRatio float64, seed int64) (trainX [][]float64, trainY []int, testX [][]float64, testY []int)` | TrainTestSplit partitions a labelled dataset into training and test subsets. |

### Types

| Type | What it is |
| --- | --- |
| `Classifier` | Classifier is the common interface implemented by every supervised model in this package. |
| `DecisionTree` | DecisionTree is a CART classification tree that splits to minimise the Gini impurity. |
| `GaussianNB` | GaussianNB is a Gaussian naive Bayes classifier. |
| `KMeans` | KMeans partitions unlabelled data into k clusters with Lloyd's algorithm. |
| `KNN` | KNN is a k-nearest-neighbours classifier. |
| `KernelType` | KernelType selects the kernel function used by an `SVM`. |
| `LDA` | LDA performs Fisher's linear discriminant analysis: a supervised projection that maximises between-class scatter relative to within-class scatter,… |
| `LogisticRegression` | LogisticRegression is a multinomial (softmax) logistic-regression classifier trained by full-batch gradient descent with optional L2 regularisation. |
| `MinMaxScaler` | MinMaxScaler linearly rescales each feature into the closed interval [0, 1] based on the minimum and maximum observed during Fit. |
| `PCA` | PCA performs principal-component analysis: it finds the orthogonal directions of greatest variance in a dataset and projects data onto the leading… |
| `RandomForest` | RandomForest is a bootstrap-aggregated ensemble of CART decision trees. |
| `SVM` | SVM is a kernel support-vector-machine classifier. |
| `SVMParams` | SVMParams configures the training of an `SVM`. |
| `StandardScaler` | StandardScaler standardises features to zero mean and unit variance, column-wise. |

<details>
<summary><code>DecisionTree</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewDecisionTree(maxDepth, minSamplesSplit int) *DecisionTree` | NewDecisionTree returns an untrained tree. |
| `func (t *DecisionTree) Depth() int` | Depth returns the number of edges on the longest root-to-leaf path. |
| `func (t *DecisionTree) Fit(samples [][]float64, labels []int) error` | Fit grows the tree. |
| `func (t *DecisionTree) Predict(sample []float64) int` | Predict returns the class of the leaf that sample falls into. |
| `func (t *DecisionTree) PredictBatch(x [][]float64) []int` | PredictBatch classifies every sample in x. |

</details>

<details>
<summary><code>GaussianNB</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewGaussianNB() *GaussianNB` | NewGaussianNB returns an unfitted Gaussian naive Bayes classifier. |
| `func (m *GaussianNB) Fit(samples [][]float64, labels []int) error` | Fit estimates per-class priors, feature means and feature variances from the training data. |
| `func (m *GaussianNB) Predict(sample []float64) int` | Predict returns the class with the greatest posterior for sample. |
| `func (m *GaussianNB) PredictBatch(x [][]float64) []int` | PredictBatch classifies every sample in x. |
| `func (m *GaussianNB) PredictProba(sample []float64) []float64` | PredictProba returns the posterior class probabilities for sample, normalised to sum to one. |

</details>

<details>
<summary><code>KMeans</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewKMeans(k, maxIter int, seed int64) *KMeans` | NewKMeans returns a KMeans configured for k clusters, at most maxIter refinement iterations and the given random seed for k-means++ initialisation. |
| `func (m *KMeans) Fit(x [][]float64) error` | Fit clusters x, populating Centroids, Labels and Inertia. |
| `func (m *KMeans) Predict(sample []float64) int` | Predict returns the index of the nearest centroid to sample. |
| `func (m *KMeans) PredictBatch(x [][]float64) []int` | PredictBatch assigns every sample in x to its nearest centroid. |

</details>

<details>
<summary><code>KNN</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewKNN(k int) *KNN` | NewKNN returns a KNN classifier using k neighbours. |
| `func (m *KNN) Fit(samples [][]float64, labels []int) error` | Fit records the training samples and labels. |
| `func (m *KNN) KNeighbors(sample []float64) (indices []int, dists []float64)` | KNeighbors returns the indices of the k nearest training samples to the query and their Euclidean distances, sorted nearest first. |
| `func (m *KNN) Predict(sample []float64) int` | Predict returns the majority class among the k nearest neighbours of sample, breaking ties toward the lower class index. |
| `func (m *KNN) PredictBatch(x [][]float64) []int` | PredictBatch classifies every sample in x, returning one label each. |

</details>

<details>
<summary><code>LDA</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewLDA(nComponents int) *LDA` | NewLDA returns an LDA that keeps nComponents discriminant axes. |
| `func (l *LDA) Fit(samples [][]float64, labels []int) error` | Fit learns the discriminant axes from labelled data. |
| `func (l *LDA) FitTransform(samples [][]float64, labels []int) ([][]float64, error)` | FitTransform is Fit followed by Transform on the same data. |
| `func (l *LDA) Transform(samples [][]float64) [][]float64` | Transform projects samples onto the discriminant axes. |

</details>

<details>
<summary><code>LogisticRegression</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewLogisticRegression(lr float64, iters int, l2 float64) *LogisticRegression` | NewLogisticRegression returns a classifier trained with the given learning rate, number of gradient-descent iterations and L2 penalty (l2 == 0… |
| `func (m *LogisticRegression) Fit(samples [][]float64, labels []int) error` | Fit trains the model with batch gradient descent on the softmax cross-entropy loss. |
| `func (m *LogisticRegression) Predict(sample []float64) int` | Predict returns the most probable class for sample. |
| `func (m *LogisticRegression) PredictBatch(x [][]float64) []int` | PredictBatch classifies every sample in x. |
| `func (m *LogisticRegression) PredictProba(sample []float64) []float64` | PredictProba returns the softmax class probabilities for sample. |

</details>

<details>
<summary><code>MinMaxScaler</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewMinMaxScaler() *MinMaxScaler` | NewMinMaxScaler returns an unfitted MinMaxScaler. |
| `func (s *MinMaxScaler) Fit(x [][]float64)` | Fit learns the per-feature minimum and maximum from x. |
| `func (s *MinMaxScaler) FitTransform(x [][]float64) [][]float64` | FitTransform is Fit followed by Transform on the same data. |
| `func (s *MinMaxScaler) Transform(x [][]float64) [][]float64` | Transform rescales x into [0, 1] with the learned bounds and returns a new matrix. |

</details>

<details>
<summary><code>PCA</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewPCA(nComponents int) *PCA` | NewPCA returns a PCA that keeps nComponents axes. |
| `func (p *PCA) ExplainedVarianceRatio() []float64` | ExplainedVarianceRatio returns, per kept component, the fraction of the total dataset variance it explains. |
| `func (p *PCA) Fit(x [][]float64) error` | Fit learns the mean and principal components of x. |
| `func (p *PCA) FitTransform(x [][]float64) ([][]float64, error)` | FitTransform is Fit followed by Transform on the same data. |
| `func (p *PCA) InverseTransform(z [][]float64) [][]float64` | InverseTransform maps projected data back into the original feature space (approximately, since discarded components are lost). |
| `func (p *PCA) Transform(x [][]float64) [][]float64` | Transform projects x onto the principal components, returning a matrix with nComponents columns. |

</details>

<details>
<summary><code>RandomForest</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewRandomForest(nTrees, maxDepth, maxFeatures int, seed int64) *RandomForest` | NewRandomForest returns an untrained forest of nTrees trees, each grown to at most maxDepth (<= 0 for unlimited). |
| `func (f *RandomForest) Fit(samples [][]float64, labels []int) error` | Fit trains the ensemble. |
| `func (f *RandomForest) Predict(sample []float64) int` | Predict returns the majority class across all trees for sample, breaking ties toward the lower class index. |
| `func (f *RandomForest) PredictBatch(x [][]float64) []int` | PredictBatch classifies every sample in x. |
| `func (f *RandomForest) PredictProba(sample []float64) []float64` | PredictProba returns the fraction of trees voting for each class, a simple ensemble probability estimate. |

</details>

<details>
<summary><code>SVM</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewSVM(params SVMParams) *SVM` | NewSVM returns an untrained SVM using the supplied parameters. |
| `func (m *SVM) DecisionFunction(sample []float64) []float64` | DecisionFunction returns the signed decision value(s) for sample. |
| `func (m *SVM) Fit(samples [][]float64, labels []int) error` | Fit trains the SVM. |
| `func (m *SVM) NumSupportVectors() int` | NumSupportVectors returns the total number of support vectors retained across all internal binary machines. |
| `func (m *SVM) Predict(sample []float64) int` | Predict returns the predicted class for sample. |
| `func (m *SVM) PredictBatch(x [][]float64) []int` | PredictBatch classifies every sample in x. |

</details>

<details>
<summary><code>SVMParams</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func DefaultSVMParams(kernel KernelType) SVMParams` | DefaultSVMParams returns reasonable defaults for the given kernel: C = 1, Gamma = 0.5, Degree = 3, Coef0 = 1, Tol = 1e-3, MaxPasses = 5. |

</details>

<details>
<summary><code>StandardScaler</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewStandardScaler() *StandardScaler` | NewStandardScaler returns an unfitted StandardScaler. |
| `func (s *StandardScaler) Fit(x [][]float64)` | Fit learns the per-feature mean and standard deviation from x. |
| `func (s *StandardScaler) FitTransform(x [][]float64) [][]float64` | FitTransform is Fit followed by Transform on the same data. |
| `func (s *StandardScaler) Transform(x [][]float64) [][]float64` | Transform standardises x with the learned statistics and returns a new matrix. |

</details>

Full signatures, doc comments and every runnable example are on
[pkg.go.dev](https://pkg.go.dev/github.com/malcolmston/opencv/ml2).

## License

MIT, as part of [`github.com/malcolmston/opencv`](..).
An independent re-implementation, not affiliated with or endorsed by the
original project.
