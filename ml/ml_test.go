package ml

import (
	"math"
	"testing"
)

// linearlySeparable returns a small, deterministic 2-D data set with two
// linearly separable classes: class 0 clustered near (1,1) and class 1 near
// (5,5).
func linearlySeparable() (samples [][]float64, labels []int) {
	class0 := [][]float64{
		{1.0, 1.0}, {1.2, 0.8}, {0.8, 1.1}, {1.1, 1.3}, {0.9, 0.9},
		{1.3, 1.0}, {0.7, 1.2}, {1.0, 0.7},
	}
	class1 := [][]float64{
		{5.0, 5.0}, {5.2, 4.8}, {4.8, 5.1}, {5.1, 5.3}, {4.9, 4.9},
		{5.3, 5.0}, {4.7, 5.2}, {5.0, 4.7},
	}
	for _, s := range class0 {
		samples = append(samples, s)
		labels = append(labels, 0)
	}
	for _, s := range class1 {
		samples = append(samples, s)
		labels = append(labels, 1)
	}
	return samples, labels
}

// threeClass returns a deterministic 2-D data set with three well-separated
// classes for multiclass tests.
func threeClass() (samples [][]float64, labels []int) {
	centres := [][]float64{{0, 0}, {6, 0}, {3, 6}}
	offsets := [][]float64{{0, 0}, {0.3, 0.2}, {-0.2, 0.3}, {0.1, -0.3}, {-0.3, -0.1}}
	for c, ctr := range centres {
		for _, o := range offsets {
			samples = append(samples, []float64{ctr[0] + o[0], ctr[1] + o[1]})
			labels = append(labels, c)
		}
	}
	return samples, labels
}

func trainAccuracy(t *testing.T, pred, labels []int, name string, want float64) {
	t.Helper()
	acc := Accuracy(pred, labels)
	if acc < want {
		t.Errorf("%s: train accuracy %.3f < %.3f", name, acc, want)
	}
}

func TestKNearest(t *testing.T) {
	samples, labels := linearlySeparable()
	m := NewKNearest(3)
	if err := m.Train(samples, labels); err != nil {
		t.Fatalf("Train: %v", err)
	}
	trainAccuracy(t, m.PredictBatch(samples), labels, "KNearest", 1.0)

	// Obvious points.
	if got := m.Predict([]float64{1.0, 1.0}); got != 0 {
		t.Errorf("Predict near class 0 = %d, want 0", got)
	}
	if got := m.Predict([]float64{5.0, 5.0}); got != 1 {
		t.Errorf("Predict near class 1 = %d, want 1", got)
	}

	// Weighted variant and FindNearest.
	m.Weighted = true
	if got := m.Predict([]float64{4.9, 5.1}); got != 1 {
		t.Errorf("weighted Predict = %d, want 1", got)
	}
	lbls, dists := m.FindNearest([]float64{1.0, 1.0}, 3)
	if len(lbls) != 3 || len(dists) != 3 {
		t.Fatalf("FindNearest returned %d labels, %d dists", len(lbls), len(dists))
	}
	for i := 1; i < len(dists); i++ {
		if dists[i] < dists[i-1] {
			t.Errorf("FindNearest distances not sorted: %v", dists)
		}
	}
}

func TestSVM(t *testing.T) {
	samples, labels := linearlySeparable()
	m := NewSVM()
	if err := m.Train(samples, labels); err != nil {
		t.Fatalf("Train: %v", err)
	}
	trainAccuracy(t, m.PredictBatch(samples), labels, "SVM", 1.0)

	// Multiclass one-vs-rest.
	s3, l3 := threeClass()
	m3 := NewSVM()
	if err := m3.Train(s3, l3); err != nil {
		t.Fatalf("Train multiclass: %v", err)
	}
	trainAccuracy(t, m3.PredictBatch(s3), l3, "SVM-3class", 1.0)
	if len(m3.Classes()) != 3 {
		t.Errorf("Classes() = %v, want 3 classes", m3.Classes())
	}
}

func TestSVMDeterministic(t *testing.T) {
	samples, labels := linearlySeparable()
	a := NewSVM()
	b := NewSVM()
	_ = a.Train(samples, labels)
	_ = b.Train(samples, labels)
	for _, s := range samples {
		if a.Predict(s) != b.Predict(s) {
			t.Fatal("SVM not deterministic across identical runs")
		}
	}
}

func TestLogisticRegression(t *testing.T) {
	samples, labels := linearlySeparable()
	m := NewLogisticRegression()
	if err := m.Train(samples, labels); err != nil {
		t.Fatalf("Train: %v", err)
	}
	trainAccuracy(t, m.PredictBatch(samples), labels, "LogisticRegression", 1.0)

	// Probabilities sum to 1.
	p := m.Probabilities([]float64{1, 1})
	var sum float64
	for _, v := range p {
		sum += v
	}
	if math.Abs(sum-1) > 1e-9 {
		t.Errorf("probabilities sum to %.6f, want 1", sum)
	}

	// Multiclass softmax.
	s3, l3 := threeClass()
	m3 := NewLogisticRegression()
	if err := m3.Train(s3, l3); err != nil {
		t.Fatalf("Train multiclass: %v", err)
	}
	trainAccuracy(t, m3.PredictBatch(s3), l3, "LogReg-3class", 1.0)
}

func TestNormalBayes(t *testing.T) {
	samples, labels := linearlySeparable()
	m := NewNormalBayesClassifier()
	if err := m.Train(samples, labels); err != nil {
		t.Fatalf("Train: %v", err)
	}
	trainAccuracy(t, m.PredictBatch(samples), labels, "NormalBayes", 1.0)

	s3, l3 := threeClass()
	m3 := NewNormalBayesClassifier()
	if err := m3.Train(s3, l3); err != nil {
		t.Fatalf("Train multiclass: %v", err)
	}
	trainAccuracy(t, m3.PredictBatch(s3), l3, "NormalBayes-3class", 1.0)
}

func TestDecisionTree(t *testing.T) {
	samples, labels := linearlySeparable()
	m := NewDecisionTree(5)
	if err := m.Train(samples, labels); err != nil {
		t.Fatalf("Train: %v", err)
	}
	trainAccuracy(t, m.PredictBatch(samples), labels, "DecisionTree", 1.0)

	s3, l3 := threeClass()
	m3 := NewDecisionTree(6)
	if err := m3.Train(s3, l3); err != nil {
		t.Fatalf("Train multiclass: %v", err)
	}
	trainAccuracy(t, m3.PredictBatch(s3), l3, "DecisionTree-3class", 1.0)
}

func TestKMeans(t *testing.T) {
	// Two well-separated clusters.
	samples, _ := linearlySeparable()
	labels, centers := KMeans(samples, 2, 100, 42)
	if len(centers) != 2 {
		t.Fatalf("KMeans returned %d centers, want 2", len(centers))
	}
	// Every sample of the same true cluster should share a label. Check via
	// centre proximity to the ground-truth means (1,1) and (5,5).
	truth := [][]float64{{1, 1}, {5, 5}}
	for _, tc := range truth {
		matched := false
		for _, c := range centers {
			if math.Hypot(c[0]-tc[0], c[1]-tc[1]) < 0.5 {
				matched = true
			}
		}
		if !matched {
			t.Errorf("no recovered center near ground truth %v (centers=%v)", tc, centers)
		}
	}
	// Labels should partition into exactly two groups.
	seen := map[int]bool{}
	for _, l := range labels {
		seen[l] = true
	}
	if len(seen) != 2 {
		t.Errorf("KMeans produced %d distinct labels, want 2", len(seen))
	}
}

func TestKMeansThreeClusters(t *testing.T) {
	samples, _ := threeClass()
	_, centers := KMeans(samples, 3, 100, 7)
	truth := [][]float64{{0, 0}, {6, 0}, {3, 6}}
	for _, tc := range truth {
		matched := false
		for _, c := range centers {
			if math.Hypot(c[0]-tc[0], c[1]-tc[1]) < 0.6 {
				matched = true
			}
		}
		if !matched {
			t.Errorf("no recovered center near %v (centers=%v)", tc, centers)
		}
	}
}

func TestKMeansDeterministic(t *testing.T) {
	samples, _ := threeClass()
	l1, c1 := KMeans(samples, 3, 100, 99)
	l2, c2 := KMeans(samples, 3, 100, 99)
	for i := range l1 {
		if l1[i] != l2[i] {
			t.Fatalf("KMeans labels differ at %d: %d vs %d", i, l1[i], l2[i])
		}
	}
	for i := range c1 {
		for j := range c1[i] {
			if c1[i][j] != c2[i][j] {
				t.Fatalf("KMeans centers differ at [%d][%d]", i, j)
			}
		}
	}
}

func TestMetrics(t *testing.T) {
	pred := []int{0, 1, 1, 0, 2}
	actual := []int{0, 1, 0, 0, 2}
	if got := Accuracy(pred, actual); math.Abs(got-0.8) > 1e-9 {
		t.Errorf("Accuracy = %.3f, want 0.8", got)
	}
	cm := ConfusionMatrix(pred, actual, 3)
	if cm[0][0] != 2 || cm[0][1] != 1 || cm[1][1] != 1 || cm[2][2] != 1 {
		t.Errorf("ConfusionMatrix wrong: %v", cm)
	}
}

func TestTrainDataSplit(t *testing.T) {
	samples, labels := threeClass()
	td := NewTrainData(samples, labels)
	train, test := td.Split(0.7, 5)
	if train.Len()+test.Len() != td.Len() {
		t.Errorf("split sizes %d+%d != %d", train.Len(), test.Len(), td.Len())
	}
	if train.Len() != int(float64(td.Len())*0.7) {
		t.Errorf("train size = %d", train.Len())
	}
	if len(train.Labels) != train.Len() {
		t.Errorf("train labels not carried along")
	}
	// Reproducible.
	tr2, _ := td.Split(0.7, 5)
	for i := range train.Labels {
		if train.Labels[i] != tr2.Labels[i] {
			t.Fatal("Split not reproducible for a fixed seed")
		}
	}
}

func TestTrainErrors(t *testing.T) {
	m := NewKNearest(1)
	if err := m.Train(nil, nil); err != ErrNoSamples {
		t.Errorf("empty: got %v, want ErrNoSamples", err)
	}
	if err := m.Train([][]float64{{1, 2}}, []int{0, 1}); err != ErrLabelMismatch {
		t.Errorf("mismatch: got %v, want ErrLabelMismatch", err)
	}
	if err := m.Train([][]float64{{1, 2}, {3}}, []int{0, 1}); err != ErrRaggedSamples {
		t.Errorf("ragged: got %v, want ErrRaggedSamples", err)
	}
}

func TestNotTrainedPanics(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic predicting on untrained model")
		}
	}()
	m := NewLogisticRegression()
	m.Predict([]float64{1, 2})
}
