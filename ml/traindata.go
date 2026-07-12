package ml

import (
	"math/rand"

	cv "github.com/malcolmston/opencv"
)

// TrainData bundles a feature matrix with the labels (for classification) and
// responses (for regression) that go with it, mirroring the role of OpenCV's
// cv::ml::TrainData. Samples is a slice of feature vectors; Labels[i] and
// Responses[i] describe Samples[i]. Either target slice may be nil when it is
// not relevant to the task at hand.
type TrainData struct {
	// Samples holds one feature vector per observation.
	Samples [][]float64
	// Labels holds an integer class label per observation, or nil.
	Labels []int
	// Responses holds a real-valued regression target per observation, or nil.
	Responses []float64
}

// NewTrainData builds a TrainData from samples and integer class labels.
func NewTrainData(samples [][]float64, labels []int) *TrainData {
	return &TrainData{Samples: samples, Labels: labels}
}

// NewRegressionData builds a TrainData from samples and real-valued responses.
func NewRegressionData(samples [][]float64, responses []float64) *TrainData {
	return &TrainData{Samples: samples, Responses: responses}
}

// Len returns the number of samples.
func (td *TrainData) Len() int { return len(td.Samples) }

// Split partitions the data into a training and a test set. The fraction
// (0..1) of samples assigned to the training set is trainRatio; the remainder
// form the test set. The assignment is a reproducible shuffle seeded by seed,
// and Labels and Responses (when present) are carried along with their samples.
// Split shares the underlying feature-vector slices with the receiver; it does
// not deep-copy each sample.
func (td *TrainData) Split(trainRatio float64, seed int64) (train, test *TrainData) {
	n := len(td.Samples)
	perm := rand.New(rand.NewSource(seed)).Perm(n)
	nTrain := int(float64(n) * trainRatio)
	if nTrain < 0 {
		nTrain = 0
	}
	if nTrain > n {
		nTrain = n
	}
	train = &TrainData{}
	test = &TrainData{}
	hasLabels := td.Labels != nil
	hasResp := td.Responses != nil
	for i, idx := range perm {
		dst := train
		if i >= nTrain {
			dst = test
		}
		dst.Samples = append(dst.Samples, td.Samples[idx])
		if hasLabels {
			dst.Labels = append(dst.Labels, td.Labels[idx])
		}
		if hasResp {
			dst.Responses = append(dst.Responses, td.Responses[idx])
		}
	}
	return train, test
}

// MatToSamples converts a single-channel [cv.Mat] whose rows are observations
// and columns are features into a [][]float64 feature matrix. Sample i is row i
// of the Mat and has one entry per column. It panics if the Mat is empty or has
// more than one channel, since the row-major-per-sample interpretation only
// makes sense for a plain 2-D data matrix.
func MatToSamples(m *cv.Mat) [][]float64 {
	if m.Empty() {
		panic("ml: MatToSamples given an empty Mat")
	}
	if m.Channels != 1 {
		panic("ml: MatToSamples requires a single-channel Mat")
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
