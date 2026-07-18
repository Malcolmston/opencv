package ml2

import (
	"math/rand"

	cv "github.com/malcolmston/opencv"
)

// Classifier is the common interface implemented by every supervised model in
// this package. Fit trains the model on a feature matrix and matching integer
// class labels (dense in [0, k)); Predict returns the predicted class of one
// feature vector. The interface lets models be used interchangeably with
// [CrossValScore] and the metrics helpers.
type Classifier interface {
	// Fit trains the model on samples and labels, returning an error for
	// malformed input.
	Fit(samples [][]float64, labels []int) error
	// Predict returns the predicted class label for a single feature vector.
	Predict(sample []float64) int
}

// MatToSamples converts a single-channel [cv.Mat] whose rows are observations
// and columns are features into a [][]float64 feature matrix. Sample i is row i
// of the Mat with one entry per column. It panics if the Mat is empty or has
// more than one channel, since the row-per-sample interpretation only makes
// sense for a plain 2-D data matrix.
func MatToSamples(m *cv.Mat) [][]float64 {
	if m.Empty() {
		panic("ml2: MatToSamples given an empty Mat")
	}
	if m.Channels != 1 {
		panic("ml2: MatToSamples requires a single-channel Mat")
	}
	out := make([][]float64, m.Rows)
	for y := 0; y < m.Rows; y++ {
		row := make([]float64, m.Cols)
		base := y * m.Cols
		for x := 0; x < m.Cols; x++ {
			row[x] = float64(m.Data[base+x])
		}
		out[y] = row
	}
	return out
}

// MatToFeatureVector flattens an entire [cv.Mat] — every row, column and
// channel — into a single feature vector in row-major, channel-interleaved
// order. This is the natural way to turn one small image or image patch into a
// fixed-length descriptor for the classifiers in this package. It panics if the
// Mat is empty.
func MatToFeatureVector(m *cv.Mat) []float64 {
	if m.Empty() {
		panic("ml2: MatToFeatureVector given an empty Mat")
	}
	out := make([]float64, len(m.Data))
	for i, v := range m.Data {
		out[i] = float64(v)
	}
	return out
}

// MatsToSamples flattens each image in a batch into one feature row with
// [MatToFeatureVector], producing a feature matrix suitable for training an
// image classifier. Every Mat must have identical dimensions and channel count.
// It panics on an empty batch or on a size mismatch between images.
func MatsToSamples(mats []*cv.Mat) [][]float64 {
	if len(mats) == 0 {
		panic("ml2: MatsToSamples given no images")
	}
	first := mats[0]
	if first.Empty() {
		panic("ml2: MatsToSamples given an empty image")
	}
	dim := len(first.Data)
	out := make([][]float64, len(mats))
	for i, m := range mats {
		if m.Empty() {
			panic("ml2: MatsToSamples given an empty image")
		}
		if len(m.Data) != dim || m.Rows != first.Rows || m.Cols != first.Cols || m.Channels != first.Channels {
			panic("ml2: MatsToSamples requires all images to share the same shape")
		}
		out[i] = MatToFeatureVector(m)
	}
	return out
}

// TrainTestSplit partitions a labelled dataset into training and test subsets.
// trainRatio (0..1) is the fraction assigned to the training set; the shuffle
// is reproducible for a given seed. The returned slices reference the original
// sample vectors (they are not deep-copied), and labels travel with their
// samples. It panics if len(samples) != len(labels).
func TrainTestSplit(samples [][]float64, labels []int, trainRatio float64, seed int64) (trainX [][]float64, trainY []int, testX [][]float64, testY []int) {
	if len(samples) != len(labels) {
		panic("ml2: TrainTestSplit requires len(samples) == len(labels)")
	}
	n := len(samples)
	perm := rand.New(rand.NewSource(seed)).Perm(n)
	nTrain := int(float64(n) * trainRatio)
	if nTrain < 0 {
		nTrain = 0
	}
	if nTrain > n {
		nTrain = n
	}
	for i, idx := range perm {
		if i < nTrain {
			trainX = append(trainX, samples[idx])
			trainY = append(trainY, labels[idx])
		} else {
			testX = append(testX, samples[idx])
			testY = append(testY, labels[idx])
		}
	}
	return trainX, trainY, testX, testY
}

// Shuffle reorders samples and their labels together with a reproducible
// permutation seeded by seed, returning new slices and leaving the inputs
// untouched. It panics if len(samples) != len(labels).
func Shuffle(samples [][]float64, labels []int, seed int64) (outX [][]float64, outY []int) {
	if len(samples) != len(labels) {
		panic("ml2: Shuffle requires len(samples) == len(labels)")
	}
	n := len(samples)
	perm := rand.New(rand.NewSource(seed)).Perm(n)
	outX = make([][]float64, n)
	outY = make([]int, n)
	for i, idx := range perm {
		outX[i] = samples[idx]
		outY[i] = labels[idx]
	}
	return outX, outY
}
