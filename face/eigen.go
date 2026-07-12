package face

import (
	cv "github.com/malcolmston/opencv"
)

// EigenFaceRecognizer implements the classic Eigenfaces method (Turk &
// Pentland, 1991). Training flattens every face into a vector, computes their
// mean, and performs a principal-component analysis on the mean-centred data;
// the leading principal axes are the "eigenfaces" spanning the face subspace.
// Each training face is projected onto that subspace and remembered. A query
// face is projected the same way and classified by nearest neighbour (Euclidean
// distance) among the stored projections.
//
// Eigenfaces are holistic: every training and query image is reduced to luma
// and resampled to a common geometry (that of the first training image), so the
// method is sensitive to alignment and illumination. Construct with
// [NewEigenFaceRecognizer]; the zero value is not usable.
type EigenFaceRecognizer struct {
	numComponents int
	rows, cols    int
	pca           *pcaModel
	projections   [][]float64
	labels        []int
	trained       bool
}

// NewEigenFaceRecognizer returns an untrained recognizer that keeps at most
// numComponents eigenfaces. Pass numComponents <= 0 to keep every component
// with non-negligible variance (at most one fewer than the number of training
// images).
func NewEigenFaceRecognizer(numComponents int) *EigenFaceRecognizer {
	return &EigenFaceRecognizer{numComponents: numComponents}
}

// Train fits the eigenspace to the labelled images. Every image is reduced to
// luma and resampled to the first image's dimensions before flattening. It
// panics on malformed input (see [FaceRecognizer]).
func (r *EigenFaceRecognizer) Train(images []*cv.Mat, labels []int) {
	validateTraining(images, labels)

	g0 := toGrayMat(images[0])
	r.rows, r.cols = g0.Rows, g0.Cols

	vectors := make([][]float64, len(images))
	for i, im := range images {
		vectors[i] = imageVector(im, r.rows, r.cols)
	}

	// Keep at most numComponents, but never more than the intrinsic rank
	// (n-1 for n mean-centred samples); computePCA enforces the rank bound.
	maxComp := r.numComponents
	if maxComp <= 0 || maxComp > len(images)-1 {
		if len(images) > 1 {
			maxComp = len(images) - 1
		} else {
			maxComp = 1
		}
	}

	r.pca = computePCA(vectors, maxComp)
	r.projections = make([][]float64, len(vectors))
	for i, v := range vectors {
		r.projections[i] = r.pca.project(v)
	}
	r.labels = append([]int(nil), labels...)
	r.trained = true
}

// Predict projects the query face into the eigenspace and returns the label of
// the nearest training projection along with the Euclidean distance to it
// (lower is more confident). It panics if the recognizer is untrained.
func (r *EigenFaceRecognizer) Predict(img *cv.Mat) (int, float64) {
	if !r.trained {
		panic("face: EigenFaceRecognizer.Predict before Train")
	}
	v := imageVector(img, r.rows, r.cols)
	q := r.pca.project(v)
	idx, dist := nearestNeighbor(r.projections, q, euclidean)
	return r.labels[idx], dist
}

// NumComponents returns the number of eigenfaces retained after training (which
// may be fewer than requested if the data has lower rank). It returns 0 before
// training.
func (r *EigenFaceRecognizer) NumComponents() int {
	if !r.trained {
		return 0
	}
	return r.pca.dim()
}

// EigenValues returns the variance associated with each retained eigenface, in
// descending order. The returned slice is a copy.
func (r *EigenFaceRecognizer) EigenValues() []float64 {
	if !r.trained {
		return nil
	}
	return append([]float64(nil), r.pca.eigenvalues...)
}

// Mean returns the average face vector learned during training (length
// rows*cols) as a copy. It returns nil before training.
func (r *EigenFaceRecognizer) Mean() []float64 {
	if !r.trained {
		return nil
	}
	return append([]float64(nil), r.pca.mean...)
}

// Project returns the coordinates of img in the trained eigenspace (one
// coefficient per retained eigenface). It panics if the recognizer is
// untrained.
func (r *EigenFaceRecognizer) Project(img *cv.Mat) []float64 {
	if !r.trained {
		panic("face: EigenFaceRecognizer.Project before Train")
	}
	return r.pca.project(imageVector(img, r.rows, r.cols))
}

// Reconstruct rebuilds a face vector (length rows*cols) from projection
// coefficients, using the first len(coeffs) eigenfaces. Supplying fewer
// coefficients yields a lower-rank approximation; this is the knob behind
// eigenface reconstruction-quality experiments. It panics if the recognizer is
// untrained.
func (r *EigenFaceRecognizer) Reconstruct(coeffs []float64) []float64 {
	if !r.trained {
		panic("face: EigenFaceRecognizer.Reconstruct before Train")
	}
	return r.pca.reconstruct(coeffs)
}
