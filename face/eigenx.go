package face

import (
	cv "github.com/malcolmston/opencv"
)

// This file exposes the learned subspace of the holistic recognizers as vectors
// and as renderable images, matching the getEigenVectors / getEigenValues /
// getMean accessors and the "eigenface" visualisations that OpenCV's Eigenfaces
// implementation provides.

// EigenVectors returns the retained eigenfaces as rows: a K×(rows*cols) matrix
// whose k-th row is the k-th unit-length principal axis in pixel space, ordered
// by descending variance. The result is a deep copy; it is nil before training.
func (r *EigenFaceRecognizer) EigenVectors() [][]float64 {
	if !r.trained {
		return nil
	}
	out := make([][]float64, len(r.pca.components))
	for i, c := range r.pca.components {
		out[i] = append([]float64(nil), c...)
	}
	return out
}

// EigenVector returns a copy of the i-th eigenface as a flat vector of length
// rows*cols, ordered by descending variance. It panics if the recognizer is
// untrained or i is out of range.
func (r *EigenFaceRecognizer) EigenVector(i int) []float64 {
	if !r.trained {
		panic("face: EigenFaceRecognizer.EigenVector before Train")
	}
	if i < 0 || i >= len(r.pca.components) {
		panic("face: EigenFaceRecognizer.EigenVector index out of range")
	}
	return append([]float64(nil), r.pca.components[i]...)
}

// Dims returns the (rows, cols) geometry every face is resampled to before
// projection, i.e. the shape of the mean face and each eigenface. It returns
// (0,0) before training.
func (r *EigenFaceRecognizer) Dims() (rows, cols int) {
	if !r.trained {
		return 0, 0
	}
	return r.rows, r.cols
}

// MeanFace renders the learned mean face as a viewable single-channel image of
// size rows×cols, saturating each averaged sample into [0,255]. It panics if the
// recognizer is untrained.
func (r *EigenFaceRecognizer) MeanFace() *cv.Mat {
	if !r.trained {
		panic("face: EigenFaceRecognizer.MeanFace before Train")
	}
	return vectorToImage(r.pca.mean, r.rows, r.cols, false)
}

// EigenFaceImage renders the i-th eigenface as a viewable single-channel image
// of size rows×cols. Eigenfaces contain signed coefficients, so the values are
// contrast-normalised to fill [0,255] (the minimum maps to 0 and the maximum to
// 255) purely for display. It panics if the recognizer is untrained or i is out
// of range.
func (r *EigenFaceRecognizer) EigenFaceImage(i int) *cv.Mat {
	if !r.trained {
		panic("face: EigenFaceRecognizer.EigenFaceImage before Train")
	}
	if i < 0 || i >= len(r.pca.components) {
		panic("face: EigenFaceRecognizer.EigenFaceImage index out of range")
	}
	return vectorToImage(r.pca.components[i], r.rows, r.cols, true)
}

// ReconstructImage rebuilds a face from projection coefficients (see
// [EigenFaceRecognizer.Reconstruct]) and renders it as a viewable single-channel
// image of size rows×cols. It panics if the recognizer is untrained.
func (r *EigenFaceRecognizer) ReconstructImage(coeffs []float64) *cv.Mat {
	if !r.trained {
		panic("face: EigenFaceRecognizer.ReconstructImage before Train")
	}
	return vectorToImage(r.pca.reconstruct(coeffs), r.rows, r.cols, false)
}

// DiscriminantAxes returns the retained Fisher discriminant directions mapped
// back into pixel space, as rows of a K×(rows*cols) matrix. Each axis is the
// composition of the PCA basis with one LDA direction, i.e. the pixel-space
// "Fisherface" along which that discriminant projects. The result is a deep
// copy; it is nil before training.
func (r *FisherFaceRecognizer) DiscriminantAxes() [][]float64 {
	if !r.trained {
		return nil
	}
	d := len(r.pca.mean)
	out := make([][]float64, len(r.lda))
	for k, axis := range r.lda {
		vec := make([]float64, d)
		for j, coeff := range axis {
			comp := r.pca.components[j]
			for p := 0; p < d; p++ {
				vec[p] += coeff * comp[p]
			}
		}
		out[k] = vec
	}
	return out
}

// FisherFaceImage renders the k-th discriminant axis (see
// [FisherFaceRecognizer.DiscriminantAxes]) as a contrast-normalised, viewable
// single-channel image of size rows×cols. It panics if the recognizer is
// untrained or k is out of range.
func (r *FisherFaceRecognizer) FisherFaceImage(k int) *cv.Mat {
	if !r.trained {
		panic("face: FisherFaceRecognizer.FisherFaceImage before Train")
	}
	if k < 0 || k >= len(r.lda) {
		panic("face: FisherFaceRecognizer.FisherFaceImage index out of range")
	}
	return vectorToImage(r.DiscriminantAxes()[k], r.rows, r.cols, true)
}

// vectorToImage packs a length rows*cols vector into a single-channel Mat. When
// normalise is true the values are linearly stretched so the minimum maps to 0
// and the maximum to 255 (used for signed eigenface/fisherface display);
// otherwise they are saturated directly into the byte range.
func vectorToImage(v []float64, rows, cols int, normalise bool) *cv.Mat {
	out := cv.NewMat(rows, cols, 1)
	if !normalise {
		for i := 0; i < rows*cols; i++ {
			out.Data[i] = clampByte(v[i])
		}
		return out
	}
	lo, hi := v[0], v[0]
	for _, x := range v {
		if x < lo {
			lo = x
		}
		if x > hi {
			hi = x
		}
	}
	span := hi - lo
	if span < 1e-12 {
		return out // constant vector -> all zero
	}
	scale := 255 / span
	for i := 0; i < rows*cols; i++ {
		out.Data[i] = clampByte((v[i] - lo) * scale)
	}
	return out
}
