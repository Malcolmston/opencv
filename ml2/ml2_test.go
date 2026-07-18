package ml2

import (
	"math"
	"math/rand"
	"testing"

	cv "github.com/malcolmston/opencv"
)

// genBlobs builds a deterministic labelled dataset: perClass Gaussian points
// scattered by spread around each centre; label i for centre i.
func genBlobs(centers [][]float64, perClass int, spread float64, seed int64) ([][]float64, []int) {
	rng := rand.New(rand.NewSource(seed))
	var x [][]float64
	var y []int
	for c, centre := range centers {
		for p := 0; p < perClass; p++ {
			pt := make([]float64, len(centre))
			for j := range centre {
				pt[j] = centre[j] + rng.NormFloat64()*spread
			}
			x = append(x, pt)
			y = append(y, c)
		}
	}
	return x, y
}

func approx(a, b, tol float64) bool { return math.Abs(a-b) <= tol }

// --- Interop with cv.Mat ---

func TestMatToSamples(t *testing.T) {
	m := cv.NewMat(2, 3, 1)
	vals := []uint8{1, 2, 3, 4, 5, 6}
	copy(m.Data, vals)
	got := MatToSamples(m)
	want := [][]float64{{1, 2, 3}, {4, 5, 6}}
	for i := range want {
		for j := range want[i] {
			if got[i][j] != want[i][j] {
				t.Fatalf("MatToSamples[%d][%d]=%v want %v", i, j, got[i][j], want[i][j])
			}
		}
	}
}

func TestMatToFeatureVectorAndBatch(t *testing.T) {
	a := cv.NewMat(1, 2, 3)
	copy(a.Data, []uint8{10, 20, 30, 40, 50, 60})
	fv := MatToFeatureVector(a)
	if len(fv) != 6 || fv[0] != 10 || fv[5] != 60 {
		t.Fatalf("MatToFeatureVector = %v", fv)
	}
	b := cv.NewMat(1, 2, 3)
	copy(b.Data, []uint8{1, 1, 1, 1, 1, 1})
	s := MatsToSamples([]*cv.Mat{a, b})
	if len(s) != 2 || len(s[0]) != 6 || s[1][0] != 1 {
		t.Fatalf("MatsToSamples = %v", s)
	}
}

// --- Preprocessing ---

func TestStandardScaler(t *testing.T) {
	x := [][]float64{{0, 100}, {2, 100}, {4, 100}}
	s := NewStandardScaler()
	z := s.FitTransform(x)
	// Column 0 has mean 2, std sqrt(8/3); column 1 constant -> unchanged.
	if !approx(s.Mean[0], 2, 1e-9) {
		t.Fatalf("mean0 = %v", s.Mean[0])
	}
	// Standardised column-0 mean must be ~0 and unit variance.
	var sum, sumsq float64
	for _, r := range z {
		sum += r[0]
		sumsq += r[0] * r[0]
	}
	if !approx(sum/3, 0, 1e-9) || !approx(sumsq/3, 1, 1e-9) {
		t.Fatalf("standardised col0 mean=%v var=%v", sum/3, sumsq/3)
	}
	// Constant column must map to 0 (std forced to 1, minus mean).
	if !approx(z[0][1], 0, 1e-9) {
		t.Fatalf("constant col not zeroed: %v", z[0][1])
	}
}

func TestMinMaxScaler(t *testing.T) {
	x := [][]float64{{0}, {5}, {10}}
	s := NewMinMaxScaler()
	z := s.FitTransform(x)
	if !approx(z[0][0], 0, 1e-9) || !approx(z[1][0], 0.5, 1e-9) || !approx(z[2][0], 1, 1e-9) {
		t.Fatalf("minmax = %v", z)
	}
}

func TestNormalize(t *testing.T) {
	v := Normalize([]float64{3, 4})
	if !approx(v[0], 0.6, 1e-9) || !approx(v[1], 0.8, 1e-9) {
		t.Fatalf("Normalize = %v", v)
	}
	if z := Normalize([]float64{0, 0}); z[0] != 0 || z[1] != 0 {
		t.Fatalf("zero norm = %v", z)
	}
}

// --- KNN ---

func TestKNN(t *testing.T) {
	x, y := genBlobs([][]float64{{0, 0}, {6, 6}}, 15, 0.4, 1)
	m := NewKNN(3)
	if err := m.Fit(x, y); err != nil {
		t.Fatal(err)
	}
	if got := m.Predict([]float64{0.2, -0.1}); got != 0 {
		t.Fatalf("near-origin predicted %d", got)
	}
	if got := m.Predict([]float64{5.8, 6.1}); got != 1 {
		t.Fatalf("near-(6,6) predicted %d", got)
	}
	idx, dists := m.KNeighbors([]float64{0, 0})
	if len(idx) != 3 || len(dists) != 3 {
		t.Fatalf("KNeighbors returned %d neighbours", len(idx))
	}
	for i := 1; i < len(dists); i++ {
		if dists[i] < dists[i-1] {
			t.Fatalf("neighbours not sorted: %v", dists)
		}
	}
}

// --- KMeans ---

func TestKMeans(t *testing.T) {
	x, y := genBlobs([][]float64{{0, 0}, {10, 10}}, 20, 0.3, 7)
	km := NewKMeans(2, 100, 42)
	if err := km.Fit(x); err != nil {
		t.Fatal(err)
	}
	// Every point in a true class should share a cluster label.
	c0 := km.Labels[0]
	for i := 0; i < 20; i++ {
		if km.Labels[i] != c0 {
			t.Fatalf("class-0 point %d in cluster %d not %d", i, km.Labels[i], c0)
		}
	}
	c1 := km.Labels[20]
	if c0 == c1 {
		t.Fatalf("both blobs collapsed into one cluster")
	}
	for i := 20; i < 40; i++ {
		if km.Labels[i] != c1 {
			t.Fatalf("class-1 point %d in cluster %d not %d", i, km.Labels[i], c1)
		}
	}
	_ = y
	if km.Inertia <= 0 {
		t.Fatalf("inertia should be positive, got %v", km.Inertia)
	}
	// A far point clusters with the (10,10) blob.
	if km.Predict([]float64{11, 9}) != c1 {
		t.Fatalf("predict near (10,10) went to wrong cluster")
	}
}

// --- Gaussian Naive Bayes ---

func TestGaussianNB(t *testing.T) {
	x, y := genBlobs([][]float64{{0, 0}, {5, 5}, {0, 5}}, 30, 0.5, 3)
	m := NewGaussianNB()
	if err := m.Fit(x, y); err != nil {
		t.Fatal(err)
	}
	acc := Accuracy(y, m.PredictBatch(x))
	if acc < 0.98 {
		t.Fatalf("GaussianNB train accuracy %v", acc)
	}
	p := m.PredictProba([]float64{0, 0})
	var sum float64
	for _, v := range p {
		sum += v
	}
	if !approx(sum, 1, 1e-9) {
		t.Fatalf("proba sums to %v", sum)
	}
	if ml2argmax(p) != 0 {
		t.Fatalf("proba argmax %d != 0", ml2argmax(p))
	}
}

// --- Logistic Regression ---

func TestLogisticRegression(t *testing.T) {
	x, y := genBlobs([][]float64{{0, 0}, {5, 5}}, 40, 0.5, 9)
	m := NewLogisticRegression(0.1, 500, 0)
	if err := m.Fit(x, y); err != nil {
		t.Fatal(err)
	}
	acc := Accuracy(y, m.PredictBatch(x))
	if acc < 0.98 {
		t.Fatalf("logistic train accuracy %v", acc)
	}
	if m.Predict([]float64{0, 0}) != 0 || m.Predict([]float64{5, 5}) != 1 {
		t.Fatalf("logistic misclassified centres")
	}
}

// --- SVM ---

func TestSVMLinear(t *testing.T) {
	x, y := genBlobs([][]float64{{0, 0}, {6, 6}}, 30, 0.5, 11)
	m := NewSVM(DefaultSVMParams(LinearKernel))
	if err := m.Fit(x, y); err != nil {
		t.Fatal(err)
	}
	acc := Accuracy(y, m.PredictBatch(x))
	if acc < 0.98 {
		t.Fatalf("linear SVM train accuracy %v", acc)
	}
	if m.NumSupportVectors() == 0 {
		t.Fatalf("no support vectors")
	}
}

func TestSVMMulticlass(t *testing.T) {
	centers := [][]float64{{0, 0}, {8, 0}, {0, 8}}
	x, y := genBlobs(centers, 25, 0.5, 5)
	p := DefaultSVMParams(LinearKernel)
	p.C = 5
	m := NewSVM(p)
	if err := m.Fit(x, y); err != nil {
		t.Fatal(err)
	}
	acc := Accuracy(y, m.PredictBatch(x))
	if acc < 0.97 {
		t.Fatalf("multiclass SVM accuracy %v", acc)
	}
	if len(m.DecisionFunction([]float64{0, 0})) != 3 {
		t.Fatalf("decision function should have 3 outputs")
	}
}

func TestSVMKernelXOR(t *testing.T) {
	// XOR is not linearly separable; an RBF kernel must still fit it.
	x := [][]float64{{0, 0}, {1, 1}, {0, 1}, {1, 0}}
	y := []int{0, 0, 1, 1}
	p := DefaultSVMParams(RBFKernel)
	p.C = 100
	p.Gamma = 1
	p.MaxPasses = 20
	m := NewSVM(p)
	if err := m.Fit(x, y); err != nil {
		t.Fatal(err)
	}
	if acc := Accuracy(y, m.PredictBatch(x)); acc != 1 {
		t.Fatalf("RBF SVM XOR accuracy %v", acc)
	}
}

// --- Decision Tree ---

func TestDecisionTree(t *testing.T) {
	x, y := genBlobs([][]float64{{0, 0}, {5, 5}}, 30, 0.5, 4)
	m := NewDecisionTree(6, 2)
	if err := m.Fit(x, y); err != nil {
		t.Fatal(err)
	}
	acc := Accuracy(y, m.PredictBatch(x))
	if acc < 0.98 {
		t.Fatalf("tree train accuracy %v", acc)
	}
	if m.Depth() < 1 {
		t.Fatalf("tree depth should be >= 1")
	}
}

func TestDecisionTreeExactSplit(t *testing.T) {
	// A single clean axis-aligned threshold at x0 ≈ 2.5.
	x := [][]float64{{1, 0}, {2, 5}, {3, 1}, {4, 9}}
	y := []int{0, 0, 1, 1}
	m := NewDecisionTree(2, 2)
	if err := m.Fit(x, y); err != nil {
		t.Fatal(err)
	}
	if m.Predict([]float64{1.5, 100}) != 0 || m.Predict([]float64{3.5, -100}) != 1 {
		t.Fatalf("tree did not learn the x0 split")
	}
}

// --- Random Forest ---

func TestRandomForest(t *testing.T) {
	x, y := genBlobs([][]float64{{0, 0}, {5, 5}, {5, 0}}, 30, 0.6, 8)
	m := NewRandomForest(25, 6, 0, 123)
	if err := m.Fit(x, y); err != nil {
		t.Fatal(err)
	}
	acc := Accuracy(y, m.PredictBatch(x))
	if acc < 0.95 {
		t.Fatalf("forest train accuracy %v", acc)
	}
	pr := m.PredictProba([]float64{0, 0})
	var sum float64
	for _, v := range pr {
		sum += v
	}
	if !approx(sum, 1, 1e-9) {
		t.Fatalf("forest proba sum %v", sum)
	}
}

// --- PCA ---

func TestPCA(t *testing.T) {
	x := [][]float64{{-2, -2}, {-1, -1}, {0, 0}, {1, 1}, {2, 2}}
	p := NewPCA(2)
	z, err := p.FitTransform(x)
	if err != nil {
		t.Fatal(err)
	}
	// First component points along the (1,1)/√2 diagonal.
	c := p.Components[0]
	if !approx(math.Abs(c[0]), math.Sqrt2/2, 1e-6) || !approx(math.Abs(c[1]), math.Sqrt2/2, 1e-6) {
		t.Fatalf("first component %v", c)
	}
	// All variance is on the first axis.
	ratio := p.ExplainedVarianceRatio()
	if !approx(ratio[0], 1, 1e-6) {
		t.Fatalf("explained variance ratio[0] = %v", ratio[0])
	}
	// Second projected coordinate is ~0 everywhere.
	for _, row := range z {
		if !approx(row[1], 0, 1e-6) {
			t.Fatalf("second projection not zero: %v", row[1])
		}
	}
	// Inverse transform reconstructs the input (no info lost on the diagonal).
	rec := p.InverseTransform(z)
	for i := range x {
		if !approx(rec[i][0], x[i][0], 1e-6) || !approx(rec[i][1], x[i][1], 1e-6) {
			t.Fatalf("reconstruction mismatch at %d: %v vs %v", i, rec[i], x[i])
		}
	}
}

// --- LDA ---

func TestLDA(t *testing.T) {
	// Two classes separated along x; within-class spread is along y.
	x := [][]float64{
		{0, -1}, {0, 0}, {0, 1},
		{4, -1}, {4, 0}, {4, 1},
	}
	y := []int{0, 0, 0, 1, 1, 1}
	l := NewLDA(1)
	z, err := l.FitTransform(x, y)
	if err != nil {
		t.Fatal(err)
	}
	// The discriminant axis is the x direction.
	c := l.Components[0]
	if !approx(math.Abs(c[0]), 1, 1e-6) || !approx(math.Abs(c[1]), 0, 1e-6) {
		t.Fatalf("LDA component %v not along x", c)
	}
	// Projected class means must be well separated.
	m0 := (z[0][0] + z[1][0] + z[2][0]) / 3
	m1 := (z[3][0] + z[4][0] + z[5][0]) / 3
	if math.Abs(m1-m0) < 3 {
		t.Fatalf("LDA projected means too close: %v vs %v", m0, m1)
	}
}

// --- Metrics ---

func TestMetrics(t *testing.T) {
	yTrue := []int{0, 0, 1, 1, 2, 2}
	yPred := []int{0, 1, 1, 1, 2, 0}
	if !approx(Accuracy(yTrue, yPred), 4.0/6.0, 1e-12) {
		t.Fatalf("accuracy = %v", Accuracy(yTrue, yPred))
	}
	cm := ConfusionMatrix(yTrue, yPred, 3)
	if cm[0][0] != 1 || cm[0][1] != 1 || cm[1][1] != 2 || cm[2][2] != 1 || cm[2][0] != 1 {
		t.Fatalf("confusion matrix = %v", cm)
	}
	// Class 1: predicted 3 times, 2 correct -> precision 2/3; actual twice, both
	// found -> recall 1.
	if !approx(Precision(yTrue, yPred, 1), 2.0/3.0, 1e-12) {
		t.Fatalf("precision = %v", Precision(yTrue, yPred, 1))
	}
	if !approx(Recall(yTrue, yPred, 1), 1, 1e-12) {
		t.Fatalf("recall = %v", Recall(yTrue, yPred, 1))
	}
	f1 := F1Score(yTrue, yPred, 1)
	if !approx(f1, 2*(2.0/3.0)/(1+2.0/3.0), 1e-12) {
		t.Fatalf("f1 = %v", f1)
	}
	_ = MacroF1(yTrue, yPred, 3)
}

// --- Cross-validation ---

func TestKFoldSplit(t *testing.T) {
	folds := KFoldSplit(10, 5, 1)
	if len(folds) != 5 {
		t.Fatalf("expected 5 folds, got %d", len(folds))
	}
	seen := make(map[int]bool)
	total := 0
	for _, f := range folds {
		total += len(f)
		for _, idx := range f {
			if seen[idx] {
				t.Fatalf("index %d appears twice", idx)
			}
			seen[idx] = true
		}
	}
	if total != 10 {
		t.Fatalf("folds cover %d indices, want 10", total)
	}
}

func TestCrossValScore(t *testing.T) {
	x, y := genBlobs([][]float64{{0, 0}, {8, 8}}, 20, 0.4, 2)
	scores := CrossValScore(func() Classifier { return NewKNN(3) }, x, y, 4, 1)
	if len(scores) != 4 {
		t.Fatalf("expected 4 scores, got %d", len(scores))
	}
	if MeanScore(scores) < 0.98 {
		t.Fatalf("cross-val mean accuracy %v", MeanScore(scores))
	}
}

// Verify every model satisfies the Classifier interface at compile time.
var (
	_ Classifier = (*KNN)(nil)
	_ Classifier = (*GaussianNB)(nil)
	_ Classifier = (*LogisticRegression)(nil)
	_ Classifier = (*SVM)(nil)
	_ Classifier = (*DecisionTree)(nil)
	_ Classifier = (*RandomForest)(nil)
)
