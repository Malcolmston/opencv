package quality

import cv "github.com/malcolmston/opencv"

// QualityBase is the common interface of the object-form full-reference
// metrics, mirroring OpenCV's cv::quality::QualityBase. An implementation is
// constructed once with a reference image and then applied to any number of
// candidates:
//
//	q := quality.NewQualitySSIM(reference)
//	score := q.Compute(candidate) // score[0] is the mean SSIM
//	m := q.QualityMap()           // per-pixel similarity map
//
// Compute returns the metric as a slice — one element per channel for the
// per-channel metrics (MSE), or a single pooled value for the rest — and
// records the per-pixel map, which QualityMap then returns. QualityMap reports
// nil before the first Compute call.
type QualityBase interface {
	// Compute scores cmp against the stored reference and updates the map.
	Compute(cmp *cv.Mat) []float64
	// QualityMap returns the per-pixel map from the most recent Compute, or
	// nil if Compute has not been called.
	QualityMap() *cv.Mat
}

// qualityMSE is the object form of [MSE]/[PSNR].
type qualityMSE struct {
	ref  *cv.Mat
	psnr bool
	qmap *cv.Mat
}

// NewQualityMSE returns a [QualityBase] whose Compute reports the per-channel
// [MSE] against ref and whose QualityMap is the pooled squared-error image.
func NewQualityMSE(ref *cv.Mat) QualityBase {
	requireImage(ref, "NewQualityMSE")
	return &qualityMSE{ref: ref}
}

// NewQualityPSNR returns a [QualityBase] whose Compute reports the pooled [PSNR]
// against ref (as a one-element slice) and whose QualityMap is the pooled
// squared-error image.
func NewQualityPSNR(ref *cv.Mat) QualityBase {
	requireImage(ref, "NewQualityPSNR")
	return &qualityMSE{ref: ref, psnr: true}
}

// Compute scores cmp against the stored reference, returning the per-channel
// [MSE] (or the pooled [PSNR] as a one-element slice when the metric was
// constructed with NewQualityPSNR), and records the pooled squared-error map
// returned by QualityMap. It implements [QualityBase].
func (q *qualityMSE) Compute(cmp *cv.Mat) []float64 {
	requireComparable(q.ref, cmp, "QualityMSE.Compute")
	q.qmap = squaredErrorMap(q.ref, cmp)
	if q.psnr {
		return []float64{PSNR(q.ref, cmp)}
	}
	return MSE(q.ref, cmp)
}

// QualityMap returns the pooled squared-error map from the most recent Compute,
// or nil if Compute has not been called. It implements [QualityBase].
func (q *qualityMSE) QualityMap() *cv.Mat { return q.qmap }

// squaredErrorMap builds a single-channel map of the per-pixel squared error
// pooled over channels, clamped into the 8-bit range for visualisation.
func squaredErrorMap(a, b *cv.Mat) *cv.Mat {
	ch := a.Channels
	g := newGrid(a.Rows, a.Cols)
	for p := 0; p < a.Total(); p++ {
		base := p * ch
		var se float64
		for c := 0; c < ch; c++ {
			d := float64(a.Data[base+c]) - float64(b.Data[base+c])
			se += d * d
		}
		g.data[p] = se / float64(ch)
	}
	return grayMapToMat(g)
}

// qualitySSIM is the object form of [SSIM].
type qualitySSIM struct {
	ref  *cv.Mat
	qmap *cv.Mat
}

// NewQualitySSIM returns a [QualityBase] whose Compute reports the mean [SSIM]
// against ref (as a one-element slice) and whose QualityMap is the per-pixel
// SSIM map.
func NewQualitySSIM(ref *cv.Mat) QualityBase {
	requireImage(ref, "NewQualitySSIM")
	return &qualitySSIM{ref: ref}
}

// Compute scores cmp against the stored reference, returning the mean [SSIM] as
// a one-element slice and recording the per-pixel SSIM map returned by
// QualityMap. It implements [QualityBase].
func (q *qualitySSIM) Compute(cmp *cv.Mat) []float64 {
	requireComparable(q.ref, cmp, "QualitySSIM.Compute")
	mean, m := SSIM(q.ref, cmp)
	q.qmap = m
	return []float64{mean}
}

// QualityMap returns the per-pixel SSIM map from the most recent Compute, or nil
// if Compute has not been called. It implements [QualityBase].
func (q *qualitySSIM) QualityMap() *cv.Mat { return q.qmap }

// qualityGMSD is the object form of [GMSD].
type qualityGMSD struct {
	ref  *cv.Mat
	qmap *cv.Mat
}

// NewQualityGMSD returns a [QualityBase] whose Compute reports the [GMSD]
// against ref (as a one-element slice) and whose QualityMap is the per-pixel
// gradient-magnitude-similarity map.
func NewQualityGMSD(ref *cv.Mat) QualityBase {
	requireImage(ref, "NewQualityGMSD")
	return &qualityGMSD{ref: ref}
}

// Compute scores cmp against the stored reference, returning the [GMSD] as a
// one-element slice and recording the per-pixel gradient-magnitude-similarity
// map returned by QualityMap. It implements [QualityBase].
func (q *qualityGMSD) Compute(cmp *cv.Mat) []float64 {
	requireComparable(q.ref, cmp, "QualityGMSD.Compute")
	dev, m := GMSD(q.ref, cmp)
	q.qmap = m
	return []float64{dev}
}

// QualityMap returns the per-pixel gradient-magnitude-similarity map from the
// most recent Compute, or nil if Compute has not been called. It implements
// [QualityBase].
func (q *qualityGMSD) QualityMap() *cv.Mat { return q.qmap }
