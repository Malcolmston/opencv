package face

import (
	"errors"
	"sort"

	cv "github.com/malcolmston/opencv"
)

// Unknown is the label returned by the threshold-aware prediction methods
// ([EigenFaceRecognizer.PredictThreshold] and friends) when the nearest match
// lies beyond the configured recognition threshold, indicating the query face
// was not confidently recognised as any known subject. It mirrors OpenCV's
// convention of returning -1 for a rejected prediction.
const Unknown = -1

// ErrVersion is returned by the Load methods when the gob stream was written by
// an incompatible version of the serialisation format.
var ErrVersion = errors.New("face: incompatible serialized model version")

// PredictedFace is a single scored candidate produced by the PredictCollect
// family: the training-set Label and the Distance from the query to that
// sample in the recognizer's feature space (lower is more similar).
type PredictedFace struct {
	// Label is the subject label of the matched training sample.
	Label int
	// Distance is the query-to-sample distance; smaller is a better match.
	Distance float64
}

// collectResults scores query against every stored sample and returns the
// (label, distance) pairs sorted by ascending distance. When threshold > 0,
// pairs beyond it are dropped. Ordering is stable, so ties keep training order.
func collectResults(db [][]float64, labels []int, query []float64, dist func(a, b []float64) float64, threshold float64) []PredictedFace {
	out := make([]PredictedFace, 0, len(db))
	for i, sample := range db {
		d := dist(sample, query)
		if threshold > 0 && d > threshold {
			continue
		}
		out = append(out, PredictedFace{Label: labels[i], Distance: d})
	}
	sort.SliceStable(out, func(i, j int) bool { return out[i].Distance < out[j].Distance })
	return out
}

// bestOrUnknown collapses a sorted result list into a single prediction. It
// returns the nearest label, or [Unknown] with the best available distance when
// the list is empty (every candidate was beyond the threshold).
func bestOrUnknown(results []PredictedFace) (int, float64) {
	if len(results) == 0 {
		return Unknown, 0
	}
	return results[0].Label, results[0].Distance
}

// PredictCollect returns every training sample scored against img, sorted by
// ascending distance in the eigenspace. When a threshold has been set with
// [EigenFaceRecognizer.SetThreshold], samples beyond it are omitted. This is the
// analogue of OpenCV's predict_collect with a StandardCollector: it exposes the
// full ranking behind a plain Predict, which is useful for k-nearest-neighbour
// voting or confidence analysis. It panics if the recognizer is untrained.
func (r *EigenFaceRecognizer) PredictCollect(img *cv.Mat) []PredictedFace {
	if !r.trained {
		panic("face: EigenFaceRecognizer.PredictCollect before Train")
	}
	q := r.pca.project(imageVector(img, r.rows, r.cols))
	return collectResults(r.projections, r.labels, q, euclidean, loadThreshold(r))
}

// PredictThreshold behaves like Predict but honours the recognition threshold:
// if the nearest training distance exceeds the value set with
// [EigenFaceRecognizer.SetThreshold], it returns ([Unknown], distance) instead
// of a spurious label. With no threshold set it is equivalent to Predict.
func (r *EigenFaceRecognizer) PredictThreshold(img *cv.Mat) (int, float64) {
	return bestOrUnknown(r.PredictCollect(img))
}

// PredictCollect returns every training sample scored against img in the
// discriminant subspace, sorted by ascending distance and threshold-filtered
// when a threshold is set. It panics if the recognizer is untrained.
func (r *FisherFaceRecognizer) PredictCollect(img *cv.Mat) []PredictedFace {
	if !r.trained {
		panic("face: FisherFaceRecognizer.PredictCollect before Train")
	}
	p := r.pca.project(imageVector(img, r.rows, r.cols))
	q := ldaProject(r.lda, p)
	return collectResults(r.projections, r.labels, q, euclidean, loadThreshold(r))
}

// PredictThreshold behaves like Predict but returns [Unknown] when the nearest
// match is beyond the configured threshold.
func (r *FisherFaceRecognizer) PredictThreshold(img *cv.Mat) (int, float64) {
	return bestOrUnknown(r.PredictCollect(img))
}

// PredictCollect returns every stored histogram scored against img under the
// chi-square distance, sorted ascending and threshold-filtered when a threshold
// is set. It panics if the recognizer is untrained.
func (r *LBPHFaceRecognizer) PredictCollect(img *cv.Mat) []PredictedFace {
	if !r.trained {
		panic("face: LBPHFaceRecognizer.PredictCollect before Train")
	}
	q := r.spatialHistogram(img)
	return collectResults(r.histograms, r.labels, q, chiSquareDistance, loadThreshold(r))
}

// PredictThreshold behaves like Predict but returns [Unknown] when the nearest
// match is beyond the configured threshold.
func (r *LBPHFaceRecognizer) PredictThreshold(img *cv.Mat) (int, float64) {
	return bestOrUnknown(r.PredictCollect(img))
}
