package face

import (
	"encoding/gob"
	"io"
	"sync"
)

// This file adds model persistence and the recognition-threshold controls that
// OpenCV's cv::face::FaceRecognizer exposes through save/load and
// setThreshold/getThreshold.
//
// # Persistence
//
// [EigenFaceRecognizer.Save], [FisherFaceRecognizer.Save] and
// [LBPHFaceRecognizer.Save] serialise a trained model to an [io.Writer] with
// [encoding/gob]; the matching Load methods restore it. A round trip reproduces
// predictions exactly. Because the recognizers hold unexported state, each Save
// copies that state into an exported snapshot struct that gob can encode, and
// Load rebuilds the recognizer from the snapshot. The wire format is versioned
// so a future change can be detected rather than silently mis-decoded.
//
// # Thresholds
//
// SetThreshold sets a maximum acceptable prediction distance. Because the three
// recognizer methods added here live outside the original struct definitions,
// the per-recognizer threshold is held in an internal side table keyed by the
// recognizer pointer rather than in a struct field; this keeps the existing
// public types unchanged. The stored value is honoured by
// [EigenFaceRecognizer.PredictThreshold] and friends and by the PredictCollect
// family, and is carried through Save/Load. A threshold of zero (the default)
// means "unbounded": every match is accepted, mirroring OpenCV's DBL_MAX
// default.

// snapshotVersion is written first in every gob stream so an incompatible
// format can be rejected on load.
const snapshotVersion = 1

// thresholdTable holds the recognition threshold for recognizers on which
// SetThreshold has been called. Keys are the recognizer pointers; the table is
// safe for concurrent use.
var thresholdTable sync.Map

func storeThreshold(key any, t float64) { thresholdTable.Store(key, t) }

func loadThreshold(key any) float64 {
	if v, ok := thresholdTable.Load(key); ok {
		return v.(float64)
	}
	return 0
}

// pcaSnapshot is the gob-encodable form of the internal pcaModel.
type pcaSnapshot struct {
	Mean        []float64
	Components  [][]float64
	Eigenvalues []float64
}

func snapshotPCA(p *pcaModel) pcaSnapshot {
	return pcaSnapshot{Mean: p.mean, Components: p.components, Eigenvalues: p.eigenvalues}
}

func (s pcaSnapshot) model() *pcaModel {
	return &pcaModel{mean: s.Mean, components: s.Components, eigenvalues: s.Eigenvalues}
}

// eigenSnapshot is the gob wire format of an [EigenFaceRecognizer].
type eigenSnapshot struct {
	Version       int
	NumComponents int
	Rows, Cols    int
	PCA           pcaSnapshot
	Projections   [][]float64
	Labels        []int
	Threshold     float64
}

// Save writes the trained recognizer to w using [encoding/gob]. It panics if
// the recognizer is untrained and returns any encoding error from w.
func (r *EigenFaceRecognizer) Save(w io.Writer) error {
	if !r.trained {
		panic("face: EigenFaceRecognizer.Save before Train")
	}
	snap := eigenSnapshot{
		Version:       snapshotVersion,
		NumComponents: r.numComponents,
		Rows:          r.rows,
		Cols:          r.cols,
		PCA:           snapshotPCA(r.pca),
		Projections:   r.projections,
		Labels:        r.labels,
		Threshold:     loadThreshold(r),
	}
	return gob.NewEncoder(w).Encode(&snap)
}

// Load restores a recognizer previously written with [EigenFaceRecognizer.Save],
// replacing the receiver's state. It returns a decoding error, including
// [ErrVersion] when the stream was produced by an incompatible version.
func (r *EigenFaceRecognizer) Load(rd io.Reader) error {
	var snap eigenSnapshot
	if err := gob.NewDecoder(rd).Decode(&snap); err != nil {
		return err
	}
	if snap.Version != snapshotVersion {
		return ErrVersion
	}
	r.numComponents = snap.NumComponents
	r.rows, r.cols = snap.Rows, snap.Cols
	r.pca = snap.PCA.model()
	r.projections = snap.Projections
	r.labels = snap.Labels
	r.trained = true
	if snap.Threshold != 0 {
		storeThreshold(r, snap.Threshold)
	}
	return nil
}

// fisherSnapshot is the gob wire format of a [FisherFaceRecognizer].
type fisherSnapshot struct {
	Version       int
	NumComponents int
	Rows, Cols    int
	PCA           pcaSnapshot
	LDA           [][]float64
	Projections   [][]float64
	Labels        []int
	Threshold     float64
}

// Save writes the trained recognizer to w using [encoding/gob]. It panics if
// the recognizer is untrained.
func (r *FisherFaceRecognizer) Save(w io.Writer) error {
	if !r.trained {
		panic("face: FisherFaceRecognizer.Save before Train")
	}
	snap := fisherSnapshot{
		Version:       snapshotVersion,
		NumComponents: r.numComponents,
		Rows:          r.rows,
		Cols:          r.cols,
		PCA:           snapshotPCA(r.pca),
		LDA:           r.lda,
		Projections:   r.projections,
		Labels:        r.labels,
		Threshold:     loadThreshold(r),
	}
	return gob.NewEncoder(w).Encode(&snap)
}

// Load restores a recognizer previously written with [FisherFaceRecognizer.Save].
func (r *FisherFaceRecognizer) Load(rd io.Reader) error {
	var snap fisherSnapshot
	if err := gob.NewDecoder(rd).Decode(&snap); err != nil {
		return err
	}
	if snap.Version != snapshotVersion {
		return ErrVersion
	}
	r.numComponents = snap.NumComponents
	r.rows, r.cols = snap.Rows, snap.Cols
	r.pca = snap.PCA.model()
	r.lda = snap.LDA
	r.projections = snap.Projections
	r.labels = snap.Labels
	r.trained = true
	if snap.Threshold != 0 {
		storeThreshold(r, snap.Threshold)
	}
	return nil
}

// lbphSnapshot is the gob wire format of an [LBPHFaceRecognizer].
type lbphSnapshot struct {
	Version    int
	GridX      int
	GridY      int
	Uniform    bool
	Histograms [][]float64
	Labels     []int
	Threshold  float64
}

// Save writes the trained recognizer to w using [encoding/gob]. It panics if
// the recognizer is untrained.
func (r *LBPHFaceRecognizer) Save(w io.Writer) error {
	if !r.trained {
		panic("face: LBPHFaceRecognizer.Save before Train")
	}
	snap := lbphSnapshot{
		Version:    snapshotVersion,
		GridX:      r.GridX,
		GridY:      r.GridY,
		Uniform:    r.Uniform,
		Histograms: r.histograms,
		Labels:     r.labels,
		Threshold:  loadThreshold(r),
	}
	return gob.NewEncoder(w).Encode(&snap)
}

// Load restores a recognizer previously written with [LBPHFaceRecognizer.Save].
func (r *LBPHFaceRecognizer) Load(rd io.Reader) error {
	var snap lbphSnapshot
	if err := gob.NewDecoder(rd).Decode(&snap); err != nil {
		return err
	}
	if snap.Version != snapshotVersion {
		return ErrVersion
	}
	r.GridX = snap.GridX
	r.GridY = snap.GridY
	r.Uniform = snap.Uniform
	r.histograms = snap.Histograms
	r.labels = snap.Labels
	r.trained = true
	if snap.Threshold != 0 {
		storeThreshold(r, snap.Threshold)
	}
	return nil
}

// SetThreshold sets the maximum acceptable prediction distance for this
// recognizer. Predictions whose nearest distance exceeds t are reported as
// "unknown" (label [Unknown]) by the threshold-aware prediction methods. A
// non-positive t clears the threshold, restoring the default unbounded
// behaviour.
func (r *EigenFaceRecognizer) SetThreshold(t float64) { storeThreshold(r, maxNonNeg(t)) }

// GetThreshold returns the recognition threshold set by
// [EigenFaceRecognizer.SetThreshold], or 0 (unbounded) if none was set.
func (r *EigenFaceRecognizer) GetThreshold() float64 { return loadThreshold(r) }

// SetThreshold sets the maximum acceptable prediction distance; see
// [EigenFaceRecognizer.SetThreshold].
func (r *FisherFaceRecognizer) SetThreshold(t float64) { storeThreshold(r, maxNonNeg(t)) }

// GetThreshold returns the recognition threshold, or 0 if unset.
func (r *FisherFaceRecognizer) GetThreshold() float64 { return loadThreshold(r) }

// SetThreshold sets the maximum acceptable prediction distance; see
// [EigenFaceRecognizer.SetThreshold].
func (r *LBPHFaceRecognizer) SetThreshold(t float64) { storeThreshold(r, maxNonNeg(t)) }

// GetThreshold returns the recognition threshold, or 0 if unset.
func (r *LBPHFaceRecognizer) GetThreshold() float64 { return loadThreshold(r) }

// maxNonNeg clamps a threshold to a non-negative value, treating anything <= 0
// as the unbounded default of 0.
func maxNonNeg(t float64) float64 {
	if t <= 0 {
		return 0
	}
	return t
}
