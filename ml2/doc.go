// Package ml2 is a from-scratch, standard-library-only toolkit of classic
// (non-neural) machine-learning models aimed at computer-vision workflows:
// classification, regression, clustering and dimensionality reduction over
// feature vectors extracted from images.
//
// The package operates on plain Go data. A dataset is a slice of feature
// vectors [][]float64 (rows are observations, columns are features), paired
// with []int class labels for classifiers or used unlabelled for clustering
// and projection. This keeps the API independent of the image-oriented
// [cv.Mat] type while remaining trivial to bridge to it: convert a
// single-channel data matrix with [MatToSamples], flatten one image into a
// single feature vector with [MatToFeatureVector], or turn a batch of
// same-sized images into a feature matrix with [MatsToSamples].
//
// Like the parent package, ml2 is written entirely against the Go standard
// library (math, sort, math/rand). It uses no cgo and no third-party
// dependencies, and it does not import the other cv/* subpackages. All
// randomised routines take an explicit seed and are fully deterministic.
//
// # Supervised models
//
// Every classifier implements the [Classifier] interface — Fit to train and
// Predict for a single sample — so they are interchangeable in [CrossValScore]
// and the metrics helpers. The models are:
//
//   - [KNN] — k-nearest-neighbours with majority vote.
//   - [GaussianNB] — Gaussian naive Bayes.
//   - [LogisticRegression] — multinomial (softmax) logistic regression.
//   - [SVM] — a kernel support-vector machine trained with Platt's SMO,
//     extended to multiple classes one-vs-rest, with linear, polynomial and
//     RBF kernels.
//   - [DecisionTree] — a CART decision tree using the Gini impurity.
//   - [RandomForest] — a bootstrap-aggregated ensemble of randomised trees.
//
// # Unsupervised models
//
//   - [KMeans] — Lloyd's k-means clustering with deterministic k-means++
//     seeding.
//   - [PCA] — principal-component analysis for dimensionality reduction.
//   - [LDA] — Fisher's linear discriminant analysis for supervised projection.
//
// # Preprocessing and evaluation
//
// [StandardScaler] and [MinMaxScaler] rescale features; [Accuracy],
// [ConfusionMatrix], [Precision], [Recall], [F1Score] and [MacroF1] measure
// classifier quality; [KFoldSplit] and [CrossValScore] perform k-fold
// cross-validation.
package ml2
