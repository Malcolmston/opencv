package quality

import cv "github.com/malcolmston/opencv"

// qualityRMSE is the object form of [RMSE].
type qualityRMSE struct {
	ref  *cv.Mat
	qmap *cv.Mat
}

// NewQualityRMSE returns a [QualityBase] whose Compute reports the per-channel
// [RMSE] against ref and whose QualityMap is the pooled squared-error image.
func NewQualityRMSE(ref *cv.Mat) QualityBase {
	requireImage(ref, "NewQualityRMSE")
	return &qualityRMSE{ref: ref}
}

// Compute scores cmp against the stored reference, returning the per-channel
// [RMSE] and recording the pooled squared-error map returned by QualityMap. It
// implements [QualityBase].
func (q *qualityRMSE) Compute(cmp *cv.Mat) []float64 {
	requireComparable(q.ref, cmp, "QualityRMSE.Compute")
	q.qmap = squaredErrorMap(q.ref, cmp)
	return RMSE(q.ref, cmp)
}

// QualityMap returns the pooled squared-error map from the most recent Compute,
// or nil if Compute has not been called. It implements [QualityBase].
func (q *qualityRMSE) QualityMap() *cv.Mat { return q.qmap }

// qualityFSIM is the object form of [FSIM]/[FSIMc].
type qualityFSIM struct {
	ref   *cv.Mat
	color bool
	qmap  *cv.Mat
}

// NewQualityFSIM returns a [QualityBase] whose Compute reports the [FSIM] against
// ref (as a one-element slice) and whose QualityMap is the per-pixel
// feature-similarity map.
func NewQualityFSIM(ref *cv.Mat) QualityBase {
	requireImage(ref, "NewQualityFSIM")
	return &qualityFSIM{ref: ref}
}

// NewQualityFSIMc returns a [QualityBase] whose Compute reports the colour
// [FSIMc] against ref (as a one-element slice) and whose QualityMap is the
// per-pixel feature-similarity map. For single-channel references it behaves
// exactly like [NewQualityFSIM].
func NewQualityFSIMc(ref *cv.Mat) QualityBase {
	requireImage(ref, "NewQualityFSIMc")
	return &qualityFSIM{ref: ref, color: ref.Channels == 3}
}

// Compute scores cmp against the stored reference, returning the [FSIM] (or the
// colour [FSIMc] when constructed with NewQualityFSIMc) as a one-element slice
// and recording the per-pixel feature-similarity map returned by QualityMap. It
// implements [QualityBase].
func (q *qualityFSIM) Compute(cmp *cv.Mat) []float64 {
	requireComparable(q.ref, cmp, "QualityFSIM.Compute")
	num, den, slMap := fsimAccumulate(toGray(q.ref), toGray(cmp), q.ref, cmp, q.color)
	q.qmap = similarityMapToMat(slMap)
	if den == 0 {
		return []float64{1}
	}
	return []float64{num / den}
}

// QualityMap returns the per-pixel feature-similarity map from the most recent
// Compute, or nil if Compute has not been called. It implements [QualityBase].
func (q *qualityFSIM) QualityMap() *cv.Mat { return q.qmap }

// qualityVIFP is the object form of [VIFP].
type qualityVIFP struct {
	ref  *cv.Mat
	qmap *cv.Mat
}

// NewQualityVIFP returns a [QualityBase] whose Compute reports the [VIFP] against
// ref (as a one-element slice) and whose QualityMap is the per-pixel SSIM map (a
// spatial companion to the pooled fidelity score).
func NewQualityVIFP(ref *cv.Mat) QualityBase {
	requireImage(ref, "NewQualityVIFP")
	return &qualityVIFP{ref: ref}
}

// Compute scores cmp against the stored reference, returning the [VIFP] as a
// one-element slice and recording the per-pixel SSIM map returned by QualityMap.
// It implements [QualityBase].
func (q *qualityVIFP) Compute(cmp *cv.Mat) []float64 {
	requireComparable(q.ref, cmp, "QualityVIFP.Compute")
	q.qmap = SSIMMap(q.ref, cmp)
	return []float64{VIFP(q.ref, cmp)}
}

// QualityMap returns the per-pixel SSIM map from the most recent Compute, or nil
// if Compute has not been called. It implements [QualityBase].
func (q *qualityVIFP) QualityMap() *cv.Mat { return q.qmap }
