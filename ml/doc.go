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
//
// Clustering is provided by [KMeans], an unsupervised Lloyd's-algorithm
// implementation with k-means++ seeding.
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
// train/test [TrainData.Split]. Evaluate predictions with [Accuracy] and
// [ConfusionMatrix]. Convert a feature matrix stored as a single-channel
// [cv.Mat] (one row per sample) into [][]float64 with [MatToSamples].
//
// # Errors and panics
//
// Train methods return an error for malformed input (empty data, ragged
// feature vectors or a sample/label count mismatch). Prediction helpers favour
// speed and panic on a feature-length mismatch or when called before Train,
// mirroring a Go slice index error.
package ml
