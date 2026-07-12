// Package ml is a from-scratch, standard-library-only port of a useful subset
// of OpenCV's ml module: classic (non-neural) machine-learning models for
// statistical classification, regression and clustering.
//
// The package operates on plain Go data. A training set is a slice of samples,
// where each sample is a feature vector [][]float64 (rows are observations,
// columns are features), paired with []int class labels (for classifiers) or
// []float64 responses (for regressors). This keeps the API independent of the
// image-oriented [cv.Mat] type; when your data already lives in a Mat, convert
// it once with [MatToSamples] and feed the result to any model here.
//
// Like the parent package, ml is written entirely against the Go standard
// library (math, sort, math/rand). It uses no cgo and no third-party
// dependencies, and it does not import the other cv/* subpackages.
//
// # Models
//
// Every classifier follows the same shape: construct it, call Train with
// samples and integer labels, then classify with Predict (one sample) or
// PredictBatch (many). The models are:
//
//   - [KNearest] — k-nearest-neighbours with majority vote or inverse-distance
//     weighting, plus a lower-level [KNearest.FindNearest].
//   - [SVM] — a linear soft-margin support-vector machine trained with the
//     Pegasos sub-gradient method, extended to multiple classes one-vs-rest.
//   - [NormalBayesClassifier] — Gaussian naive Bayes.
//   - [LogisticRegression] — multinomial (softmax) logistic regression trained
//     by batch gradient descent.
//   - [DecisionTree] — a CART decision tree using the Gini impurity with a
//     configurable maximum depth.
//   - [RTrees] — a random forest of CART trees with bootstrap aggregation,
//     per-split feature subsampling and a free out-of-bag error estimate.
//   - [Boost] — a boosted ensemble of decision stumps trained with multiclass
//     AdaBoost.SAMME.
//   - [ANNMLP] — a feed-forward multilayer perceptron with configurable hidden
//     layers, sigmoid or tanh activations and a softmax output, trained by
//     back-propagation.
//   - [KernelSVM] — a kernelised support-vector machine (linear, Gaussian RBF or
//     polynomial kernel) trained with kernel Pegasos, extended one-vs-rest.
//
// Clustering and density estimation are provided by [KMeans], an unsupervised
// Lloyd's-algorithm implementation with k-means++ seeding, and by
// [GaussianMixture], a Gaussian mixture model fitted with Expectation-
// Maximization that also reports the data log-likelihood.
//
// # Determinism
//
// Models whose training draws on randomness (SVM's sample shuffling and
// KMeans's k-means++ seeding) take an explicit int64 seed so that repeated runs
// on the same data produce identical results. There is no hidden global state.
//
// # Helpers and metrics
//
// [TrainData] bundles samples with their labels and offers a reproducible
// train/test [TrainData.Split], a [TrainData.StratifiedSplit] that preserves
// class balance, and a [TrainData.KFold] partition for cross-validation.
// [CrossValScore] runs k-fold validation of any [Classifier], and [StandardScaler]
// and [MinMaxScaler] expose the feature-scaling utilities.
//
// Evaluate classifiers with [Accuracy], [ConfusionMatrix], per-class [Precision],
// [Recall] and [F1Score] (plus their macro averages), and for binary scorers the
// [ROCCurve] and [AUC]. Regression fits are scored with [MSE], [RMSE], [MAE] and
// [R2Score]. Convert a feature matrix stored as a single-channel [cv.Mat] (one
// row per sample) into [][]float64 with [MatToSamples].
//
// # Persistence
//
// Every ensemble and network model can be serialised with [Save]/[Load] (or the
// [SaveFile]/[LoadFile] convenience wrappers), which are thin helpers over
// encoding/gob; the models implement gob.GobEncoder and gob.GobDecoder so a
// fitted model round-trips losslessly.
//
// # Errors and panics
//
// Train methods return an error for malformed input (empty data, ragged
// feature vectors or a sample/label count mismatch). Prediction helpers favour
// speed and panic on a feature-length mismatch or when called before Train,
// mirroring a Go slice index error.
package ml
