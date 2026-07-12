package ml

import (
	"testing"

	cv "github.com/malcolmston/opencv"
)

func expectPanic(t *testing.T, name string, fn func()) {
	t.Helper()
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("%s: expected panic", name)
		}
	}()
	fn()
}

func TestSVMDecisionScoresAndClasses(t *testing.T) {
	s, l := threeClass()
	m := NewSVM()
	_ = m.Train(s, l)
	scores := m.DecisionScores([]float64{6, 0})
	if len(scores) != 3 {
		t.Fatalf("DecisionScores len = %d", len(scores))
	}
	if argmax(scores) != 1 {
		t.Errorf("expected class index 1 to win near (6,0), got %d", argmax(scores))
	}
}

func TestLogisticClasses(t *testing.T) {
	s, l := threeClass()
	m := NewLogisticRegression()
	_ = m.Train(s, l)
	c := m.Classes()
	if len(c) != 3 || c[0] != 0 || c[2] != 2 {
		t.Errorf("Classes() = %v", c)
	}
}

func TestQueryPanics(t *testing.T) {
	s, l := linearlySeparable()
	svm := NewSVM()
	_ = svm.Train(s, l)
	expectPanic(t, "SVM wrong dim", func() { svm.Predict([]float64{1}) })

	nb := NewNormalBayesClassifier()
	_ = nb.Train(s, l)
	expectPanic(t, "NB wrong dim", func() { nb.Predict([]float64{1, 2, 3}) })

	dt := NewDecisionTree(3)
	_ = dt.Train(s, l)
	expectPanic(t, "DT wrong dim", func() { dt.Predict([]float64{1}) })

	knn := NewKNearest(1)
	expectPanic(t, "KNN untrained", func() { knn.Predict([]float64{1, 2}) })
	expectPanic(t, "NewKNearest bad k", func() { NewKNearest(0) })
}

func TestMetricsPanics(t *testing.T) {
	expectPanic(t, "Accuracy len", func() { Accuracy([]int{0}, []int{0, 1}) })
	expectPanic(t, "CM len", func() { ConfusionMatrix([]int{0}, []int{0, 1}, 2) })
	expectPanic(t, "CM numClasses", func() { ConfusionMatrix([]int{0}, []int{0}, 0) })
	expectPanic(t, "CM range", func() { ConfusionMatrix([]int{5}, []int{0}, 2) })
	if Accuracy(nil, nil) != 0 {
		t.Error("Accuracy of empty should be 0")
	}
}

func TestKMeansPanics(t *testing.T) {
	expectPanic(t, "empty", func() { KMeans(nil, 1, 10, 0) })
	expectPanic(t, "bad k", func() { KMeans([][]float64{{1}}, 5, 10, 0) })
	expectPanic(t, "ragged", func() { KMeans([][]float64{{1, 2}, {3}}, 1, 10, 0) })
}

func TestRegressionTrainData(t *testing.T) {
	samples := [][]float64{{1}, {2}, {3}, {4}}
	resp := []float64{1, 2, 3, 4}
	td := NewRegressionData(samples, resp)
	train, test := td.Split(0.5, 3)
	if len(train.Responses) != train.Len() || len(test.Responses) != test.Len() {
		t.Error("responses not carried through Split")
	}
	if train.Labels != nil {
		t.Error("expected nil labels for regression split")
	}
}

func TestMatToSamplesPanics(t *testing.T) {
	expectPanic(t, "empty mat", func() { MatToSamples(&cv.Mat{}) })
	expectPanic(t, "multichannel", func() { MatToSamples(cv.NewMat(2, 2, 3)) })
}
