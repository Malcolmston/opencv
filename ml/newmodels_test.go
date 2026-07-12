package ml

import (
	"bytes"
	"math"
	"math/rand"
	"testing"
)

// gaussianBlobs returns a deterministic 2-D data set of two Gaussian clusters
// (class 0 around (0,0), class 1 around (4,4)) with the given per-class count and
// spread, generated from a fixed seed for reproducibility.
func gaussianBlobs(perClass int, spread float64, seed int64) (samples [][]float64, labels []int) {
	rng := rand.New(rand.NewSource(seed))
	centres := [][]float64{{0, 0}, {4, 4}}
	for c, ctr := range centres {
		for i := 0; i < perClass; i++ {
			samples = append(samples, []float64{
				ctr[0] + rng.NormFloat64()*spread,
				ctr[1] + rng.NormFloat64()*spread,
			})
			labels = append(labels, c)
		}
	}
	return samples, labels
}

// xorData returns the classic non-linearly-separable XOR problem: the four
// corners of the unit square, class 1 on the main diagonal and class 0 on the
// anti-diagonal.
func xorData() (samples [][]float64, labels []int) {
	return [][]float64{{0, 0}, {1, 1}, {0, 1}, {1, 0}}, []int{1, 1, 0, 0}
}

// xorBlobs returns many noisy samples around the XOR corners for models that
// need more than four points.
func xorBlobs(perCorner int, spread float64, seed int64) (samples [][]float64, labels []int) {
	rng := rand.New(rand.NewSource(seed))
	corners := [][]float64{{0, 0}, {1, 1}, {0, 1}, {1, 0}}
	lbl := []int{1, 1, 0, 0}
	for c, ctr := range corners {
		for i := 0; i < perCorner; i++ {
			samples = append(samples, []float64{
				ctr[0] + rng.NormFloat64()*spread,
				ctr[1] + rng.NormFloat64()*spread,
			})
			labels = append(labels, lbl[c])
		}
	}
	return samples, labels
}

func heldOutAccuracy(t *testing.T, model Classifier, train, test *TrainData, name string, want float64) {
	t.Helper()
	if err := model.Train(train.Samples, train.Labels); err != nil {
		t.Fatalf("%s Train: %v", name, err)
	}
	acc := Accuracy(model.PredictBatch(test.Samples), test.Labels)
	if acc < want {
		t.Errorf("%s: held-out accuracy %.3f < %.3f", name, acc, want)
	}
}

func TestRTrees(t *testing.T) {
	samples, labels := gaussianBlobs(60, 0.7, 1)
	td := NewTrainData(samples, labels)
	train, test := td.Split(0.7, 42)

	m := NewRTrees(25)
	m.MaxDepth = 6
	heldOutAccuracy(t, m, train, test, "RTrees", 0.9)

	if oob := m.OOBError(); oob < 0 || oob > 1 {
		t.Errorf("OOBError out of range: %f", oob)
	}
	if oob := m.OOBError(); oob > 0.2 {
		t.Errorf("OOBError %.3f unexpectedly high on separable data", oob)
	}
	if len(m.Classes()) != 2 {
		t.Errorf("Classes() = %v", m.Classes())
	}

	// Deterministic across identical runs.
	a := NewRTrees(15)
	b := NewRTrees(15)
	_ = a.Train(samples, labels)
	_ = b.Train(samples, labels)
	for _, s := range samples {
		if a.Predict(s) != b.Predict(s) {
			t.Fatal("RTrees not deterministic")
		}
	}

	// Multiclass.
	s3, l3 := threeClass()
	m3 := NewRTrees(30)
	if err := m3.Train(s3, l3); err != nil {
		t.Fatalf("Train multiclass: %v", err)
	}
	trainAccuracy(t, m3.PredictBatch(s3), l3, "RTrees-3class", 1.0)
}

func TestBoost(t *testing.T) {
	samples, labels := linearlySeparable()
	m := NewBoost(20)
	if err := m.Train(samples, labels); err != nil {
		t.Fatalf("Train: %v", err)
	}
	trainAccuracy(t, m.PredictBatch(samples), labels, "Boost", 1.0)

	// Held-out on noisy blobs.
	bs, bl := gaussianBlobs(50, 0.8, 7)
	td := NewTrainData(bs, bl)
	tr, te := td.Split(0.7, 3)
	heldOutAccuracy(t, NewBoost(30), tr, te, "Boost-heldout", 0.9)

	// Multiclass SAMME.
	s3, l3 := threeClass()
	m3 := NewBoost(40)
	if err := m3.Train(s3, l3); err != nil {
		t.Fatalf("Train multiclass: %v", err)
	}
	trainAccuracy(t, m3.PredictBatch(s3), l3, "Boost-3class", 1.0)
	if len(m3.DecisionScores(s3[0])) != 3 {
		t.Error("DecisionScores wrong length")
	}
}

func TestANNMLPXor(t *testing.T) {
	samples, labels := xorData()
	m := NewANNMLP(8)
	m.Activation = Tanh
	m.LearningRate = 0.5
	m.Epochs = 4000
	m.Seed = 1
	if err := m.Train(samples, labels); err != nil {
		t.Fatalf("Train: %v", err)
	}
	// A linear model cannot solve XOR; the MLP must classify all four corners.
	trainAccuracy(t, m.PredictBatch(samples), labels, "ANNMLP-XOR", 1.0)

	p := m.Probabilities([]float64{0, 0})
	var sum float64
	for _, v := range p {
		sum += v
	}
	if math.Abs(sum-1) > 1e-9 {
		t.Errorf("probabilities sum to %.6f, want 1", sum)
	}

	// Sigmoid activation on noisy XOR blobs, held out.
	bs, bl := xorBlobs(40, 0.15, 11)
	td := NewTrainData(bs, bl)
	tr, te := td.Split(0.7, 5)
	sig := NewANNMLP(10, 6)
	sig.Activation = Sigmoid
	sig.LearningRate = 0.3
	sig.Epochs = 3000
	heldOutAccuracy(t, sig, tr, te, "ANNMLP-sigmoid", 0.9)
}

func TestGaussianMixture(t *testing.T) {
	samples, labels := gaussianBlobs(80, 0.6, 3)
	m := NewGaussianMixture(2)
	if err := m.Fit(samples); err != nil {
		t.Fatalf("Fit: %v", err)
	}
	// Each true class must map (consistently) to a single component: the
	// clustering should recover the two blobs, so grouping by component index
	// must agree with the labels up to relabelling.
	pred := m.PredictBatch(samples)
	if clusteringAgreement(pred, labels, 2) < 0.95 {
		t.Errorf("GMM clustering agreement %.3f < 0.95", clusteringAgreement(pred, labels, 2))
	}
	if !isFinite(m.TotalLogLikelihood()) {
		t.Errorf("log-likelihood not finite: %v", m.TotalLogLikelihood())
	}
	if len(m.Means()) != 2 || len(m.Weights()) != 2 {
		t.Error("Means/Weights wrong length")
	}
	ws := m.Weights()
	if math.Abs(ws[0]+ws[1]-1) > 1e-6 {
		t.Errorf("weights sum to %.6f, want 1", ws[0]+ws[1])
	}

	// Deterministic.
	a := NewGaussianMixture(2)
	b := NewGaussianMixture(2)
	_ = a.Fit(samples)
	_ = b.Fit(samples)
	for _, s := range samples {
		if a.Predict(s) != b.Predict(s) {
			t.Fatal("GMM not deterministic")
		}
	}
}

// clusteringAgreement returns the best label-matching accuracy between predicted
// cluster indices and true labels over all relabellings (fine for k=2).
func clusteringAgreement(pred, labels []int, k int) float64 {
	best := 0.0
	// Try both identity and swapped mapping for k==2; general case tries the two
	// obvious permutations which suffices for the tests here.
	maps := [][]int{{0, 1}, {1, 0}}
	if k != 2 {
		maps = [][]int{{0, 1, 2}, {0, 2, 1}, {1, 0, 2}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0}}
	}
	for _, mp := range maps {
		var correct int
		for i := range pred {
			if mp[pred[i]] == labels[i] {
				correct++
			}
		}
		if acc := float64(correct) / float64(len(pred)); acc > best {
			best = acc
		}
	}
	return best
}

func isFinite(f float64) bool { return !math.IsNaN(f) && !math.IsInf(f, 0) }

func TestKernelSVMRBF(t *testing.T) {
	// XOR-style blobs are not linearly separable; the RBF kernel must still
	// separate them, while the plain linear SVM should not.
	samples, labels := xorBlobs(50, 0.2, 9)
	td := NewTrainData(samples, labels)
	tr, te := td.Split(0.7, 21)

	rbf := NewKernelSVM(RBFKernel)
	rbf.Gamma = 0.5
	rbf.Epochs = 100
	heldOutAccuracy(t, rbf, tr, te, "KernelSVM-RBF", 0.9)

	// Deterministic.
	a := NewKernelSVM(RBFKernel)
	b := NewKernelSVM(RBFKernel)
	_ = a.Train(tr.Samples, tr.Labels)
	_ = b.Train(tr.Samples, tr.Labels)
	for _, s := range te.Samples {
		if a.Predict(s) != b.Predict(s) {
			t.Fatal("KernelSVM not deterministic")
		}
	}
}

func TestKernelSVMPolyAndLinear(t *testing.T) {
	samples, labels := linearlySeparable()

	poly := NewKernelSVM(PolyKernel)
	poly.Degree = 2
	poly.Epochs = 100
	if err := poly.Train(samples, labels); err != nil {
		t.Fatalf("poly Train: %v", err)
	}
	trainAccuracy(t, poly.PredictBatch(samples), labels, "KernelSVM-Poly", 1.0)

	lin := NewKernelSVM(LinearKernel)
	lin.Epochs = 100
	if err := lin.Train(samples, labels); err != nil {
		t.Fatalf("linear Train: %v", err)
	}
	trainAccuracy(t, lin.PredictBatch(samples), labels, "KernelSVM-Linear", 1.0)

	// Multiclass one-vs-rest with the RBF kernel.
	s3, l3 := threeClass()
	m3 := NewKernelSVM(RBFKernel)
	m3.Gamma = 0.2
	if err := m3.Train(s3, l3); err != nil {
		t.Fatalf("Train multiclass: %v", err)
	}
	trainAccuracy(t, m3.PredictBatch(s3), l3, "KernelSVM-3class", 1.0)
	if len(m3.DecisionScores(s3[0])) != 3 {
		t.Error("DecisionScores wrong length")
	}
}

func TestKFold(t *testing.T) {
	samples, labels := gaussianBlobs(30, 0.7, 2)
	td := NewTrainData(samples, labels)
	folds := td.KFold(5, 13)
	if len(folds) != 5 {
		t.Fatalf("KFold returned %d folds", len(folds))
	}
	// Every sample is tested exactly once, and train+test covers all samples.
	testTotal := 0
	for _, f := range folds {
		if f.Train.Len()+f.Test.Len() != td.Len() {
			t.Errorf("fold sizes %d+%d != %d", f.Train.Len(), f.Test.Len(), td.Len())
		}
		if len(f.Test.Labels) != f.Test.Len() {
			t.Error("labels not carried into fold")
		}
		testTotal += f.Test.Len()
	}
	if testTotal != td.Len() {
		t.Errorf("test folds cover %d samples, want %d", testTotal, td.Len())
	}
}

func TestStratifiedSplit(t *testing.T) {
	// Imbalanced classes: 30 of class 0, 10 of class 1.
	var samples [][]float64
	var labels []int
	for i := 0; i < 30; i++ {
		samples = append(samples, []float64{float64(i)})
		labels = append(labels, 0)
	}
	for i := 0; i < 10; i++ {
		samples = append(samples, []float64{float64(i)})
		labels = append(labels, 1)
	}
	td := NewTrainData(samples, labels)
	train, test := td.StratifiedSplit(0.7, 4)
	// Class proportions preserved: train should hold ~21 of class 0 and ~7 of 1.
	c0, c1 := 0, 0
	for _, l := range train.Labels {
		if l == 0 {
			c0++
		} else {
			c1++
		}
	}
	if c0 != 21 || c1 != 7 {
		t.Errorf("stratified train class counts = (%d,%d), want (21,7)", c0, c1)
	}
	if train.Len()+test.Len() != td.Len() {
		t.Errorf("split sizes %d+%d != %d", train.Len(), test.Len(), td.Len())
	}
}

func TestExtraMetrics(t *testing.T) {
	pred := []int{1, 1, 0, 1, 0, 0}
	actual := []int{1, 0, 0, 1, 1, 0}
	// For positive class 1: TP=2, FP=1, FN=1.
	if p := Precision(pred, actual, 1); math.Abs(p-2.0/3.0) > 1e-9 {
		t.Errorf("Precision = %.4f, want %.4f", p, 2.0/3.0)
	}
	if r := Recall(pred, actual, 1); math.Abs(r-2.0/3.0) > 1e-9 {
		t.Errorf("Recall = %.4f, want %.4f", r, 2.0/3.0)
	}
	if f := F1Score(pred, actual, 1); math.Abs(f-2.0/3.0) > 1e-9 {
		t.Errorf("F1 = %.4f, want %.4f", f, 2.0/3.0)
	}
	if mp := MacroPrecision(pred, actual); mp <= 0 || mp > 1 {
		t.Errorf("MacroPrecision out of range: %f", mp)
	}
	if mf := MacroF1(pred, actual); mf <= 0 || mf > 1 {
		t.Errorf("MacroF1 out of range: %f", mf)
	}
	if MacroRecall(pred, actual) <= 0 {
		t.Error("MacroRecall should be positive")
	}
}

func TestROCAndAUC(t *testing.T) {
	// A perfect scorer: all positives score above all negatives.
	scores := []float64{0.9, 0.8, 0.7, 0.3, 0.2, 0.1}
	actual := []int{1, 1, 1, 0, 0, 0}
	if auc := AUC(scores, actual, 1); math.Abs(auc-1.0) > 1e-9 {
		t.Errorf("perfect AUC = %.4f, want 1", auc)
	}
	fpr, tpr := ROCCurve(scores, actual, 1)
	if len(fpr) != len(tpr) {
		t.Fatal("ROC fpr/tpr length mismatch")
	}
	if fpr[0] != 0 || tpr[0] != 0 {
		t.Errorf("ROC should start at (0,0), got (%f,%f)", fpr[0], tpr[0])
	}
	last := len(fpr) - 1
	if math.Abs(fpr[last]-1) > 1e-9 || math.Abs(tpr[last]-1) > 1e-9 {
		t.Errorf("ROC should end at (1,1), got (%f,%f)", fpr[last], tpr[last])
	}
	if area := AUCFromCurve(fpr, tpr); math.Abs(area-1.0) > 1e-9 {
		t.Errorf("AUCFromCurve = %.4f, want 1", area)
	}

	// A reversed scorer has AUC 0; a random split near 0.5.
	if auc := AUC(scores, []int{0, 0, 0, 1, 1, 1}, 1); math.Abs(auc-0.0) > 1e-9 {
		t.Errorf("reversed AUC = %.4f, want 0", auc)
	}
	// Ties: all equal scores give AUC 0.5.
	if auc := AUC([]float64{1, 1, 1, 1}, []int{1, 0, 1, 0}, 1); math.Abs(auc-0.5) > 1e-9 {
		t.Errorf("tied AUC = %.4f, want 0.5", auc)
	}
}

func TestRegressionMetrics(t *testing.T) {
	pred := []float64{2, 4, 6, 8}
	actual := []float64{1, 4, 5, 9}
	// Residuals: 1,0,1,1 -> MSE=0.75, MAE=0.75.
	if got := MSE(pred, actual); math.Abs(got-0.75) > 1e-9 {
		t.Errorf("MSE = %.4f, want 0.75", got)
	}
	if got := RMSE(pred, actual); math.Abs(got-math.Sqrt(0.75)) > 1e-9 {
		t.Errorf("RMSE = %.4f", got)
	}
	if got := MAE(pred, actual); math.Abs(got-0.75) > 1e-9 {
		t.Errorf("MAE = %.4f, want 0.75", got)
	}
	if r2 := R2Score(actual, actual); math.Abs(r2-1) > 1e-9 {
		t.Errorf("R2 of perfect fit = %.4f, want 1", r2)
	}
	if r2 := R2Score(pred, actual); r2 <= 0 || r2 >= 1 {
		t.Errorf("R2 = %.4f, expected in (0,1)", r2)
	}
}

func TestScalers(t *testing.T) {
	samples := [][]float64{{0, 10}, {2, 20}, {4, 30}}
	ss := (&StandardScaler{}).Fit(samples)
	scaled := ss.TransformAll(samples)
	// Column means should be ~0 after standardisation.
	var m0, m1 float64
	for _, s := range scaled {
		m0 += s[0]
		m1 += s[1]
	}
	if math.Abs(m0) > 1e-9 || math.Abs(m1) > 1e-9 {
		t.Errorf("standardised means not zero: %f %f", m0, m1)
	}

	mm := (&MinMaxScaler{}).Fit(samples)
	sc := mm.TransformAll(samples)
	if sc[0][0] != 0 || sc[2][0] != 1 {
		t.Errorf("minmax range wrong: %v", sc)
	}
	if got := (&MinMaxScaler{}).FitTransform(samples); got[1][1] != 0.5 {
		t.Errorf("FitTransform midpoint = %f, want 0.5", got[1][1])
	}
}

func TestCrossValScore(t *testing.T) {
	samples, labels := gaussianBlobs(40, 0.6, 6)
	td := NewTrainData(samples, labels)
	scores := CrossValScore(NewRTrees(20), td, 5, 8)
	if len(scores) != 5 {
		t.Fatalf("got %d scores", len(scores))
	}
	if mean := MeanScore(scores); mean < 0.9 {
		t.Errorf("cross-val mean accuracy %.3f < 0.9", mean)
	}
}

func TestPersistence(t *testing.T) {
	samples, labels := gaussianBlobs(40, 0.6, 4)

	t.Run("RTrees", func(t *testing.T) {
		m := NewRTrees(15)
		_ = m.Train(samples, labels)
		roundTripClassifier(t, m, &RTrees{}, samples)
	})
	t.Run("Boost", func(t *testing.T) {
		m := NewBoost(20)
		_ = m.Train(samples, labels)
		roundTripClassifier(t, m, &Boost{}, samples)
	})
	t.Run("ANNMLP", func(t *testing.T) {
		m := NewANNMLP(8)
		m.Epochs = 500
		_ = m.Train(samples, labels)
		roundTripClassifier(t, m, &ANNMLP{}, samples)
	})
	t.Run("KernelSVM", func(t *testing.T) {
		m := NewKernelSVM(RBFKernel)
		m.Epochs = 30
		_ = m.Train(samples, labels)
		roundTripClassifier(t, m, &KernelSVM{}, samples)
	})
	t.Run("GaussianMixture", func(t *testing.T) {
		m := NewGaussianMixture(2)
		_ = m.Fit(samples)
		var buf bytes.Buffer
		if err := Save(&buf, m); err != nil {
			t.Fatalf("Save: %v", err)
		}
		loaded := &GaussianMixture{}
		if err := Load(&buf, loaded); err != nil {
			t.Fatalf("Load: %v", err)
		}
		for _, s := range samples {
			if m.Predict(s) != loaded.Predict(s) {
				t.Fatal("GMM prediction changed after round-trip")
			}
		}
	})
}

// roundTripClassifier gob-encodes model, decodes into fresh, and checks that
// predictions are identical.
func roundTripClassifier(t *testing.T, model, fresh Classifier, samples [][]float64) {
	t.Helper()
	var buf bytes.Buffer
	if err := Save(&buf, model); err != nil {
		t.Fatalf("Save: %v", err)
	}
	if err := Load(&buf, fresh); err != nil {
		t.Fatalf("Load: %v", err)
	}
	before := model.PredictBatch(samples)
	after := fresh.PredictBatch(samples)
	for i := range before {
		if before[i] != after[i] {
			t.Fatalf("prediction %d changed after round-trip: %d vs %d", i, before[i], after[i])
		}
	}
}
