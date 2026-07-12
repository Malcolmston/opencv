package ml

// Classifier is the common interface implemented by every supervised classifier
// in this package: [KNearest], [SVM], [KernelSVM], [NormalBayesClassifier],
// [LogisticRegression], [DecisionTree], [RTrees], [Boost] and [ANNMLP]. It lets
// generic tooling such as [CrossValScore] work with any model.
type Classifier interface {
	// Train fits the model to samples and their integer class labels.
	Train(samples [][]float64, labels []int) error
	// PredictBatch returns the predicted label for each sample, in order.
	PredictBatch(samples [][]float64) []int
}

// CrossValScore evaluates a classifier by k-fold cross-validation and returns
// the held-out accuracy on each fold, in fold order. The same model instance is
// re-trained from scratch on every fold's training split (each Train call fully
// overwrites the previous fit) and scored on that fold's test split. td must
// carry Labels. It panics if k is out of range (see [TrainData.KFold]) or a fold
// fails to train.
func CrossValScore(model Classifier, td *TrainData, k int, seed int64) []float64 {
	if td.Labels == nil {
		panic("ml: CrossValScore requires labelled data")
	}
	folds := td.KFold(k, seed)
	scores := make([]float64, len(folds))
	for i, f := range folds {
		if err := model.Train(f.Train.Samples, f.Train.Labels); err != nil {
			panic("ml: CrossValScore training failed: " + err.Error())
		}
		pred := model.PredictBatch(f.Test.Samples)
		scores[i] = Accuracy(pred, f.Test.Labels)
	}
	return scores
}

// MeanScore returns the arithmetic mean of scores, a convenience for summarising
// the per-fold results of [CrossValScore]. An empty input yields 0.
func MeanScore(scores []float64) float64 {
	if len(scores) == 0 {
		return 0
	}
	var sum float64
	for _, s := range scores {
		sum += s
	}
	return sum / float64(len(scores))
}
