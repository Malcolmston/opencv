package videostab

import (
	cv "github.com/malcolmston/opencv"
	"github.com/malcolmston/opencv/video"
)

// ImageMotionEstimator estimates the global 2-D motion directly from a pair of
// consecutive frames. It is the image-level counterpart of
// [MotionEstimatorBase] and mirrors cv::videostab::ImageMotionEstimatorBase.
type ImageMotionEstimator interface {
	// Estimate returns the motion mapping frame0 to frame1. The boolean result
	// reports whether a reliable estimate was found.
	Estimate(frame0, frame1 *cv.Mat) (Motion, bool)
	// MotionModel returns the model this estimator fits.
	MotionModel() MotionModel
}

// KeypointBasedMotionEstimator recovers global motion by detecting corner
// features in the first frame, tracking them into the second frame with
// pyramidal Lucas-Kanade optical flow, and feeding the resulting correspondences
// to a sparse [MotionEstimatorBase]. It mirrors
// cv::videostab::KeypointBasedMotionEstimator.
type KeypointBasedMotionEstimator struct {
	base MotionEstimatorBase

	// MaxCorners bounds the number of features detected per frame.
	MaxCorners int
	// QualityLevel is the corner-quality cutoff for feature detection.
	QualityLevel float64
	// MinDistance is the minimum spacing (pixels) between detected features.
	MinDistance float64
	// BlockSize is the neighbourhood size used by the corner detector.
	BlockSize int
	// WinSize and MaxLevel configure the Lucas-Kanade tracker.
	WinSize  int
	MaxLevel int
}

// NewKeypointBasedMotionEstimator wraps a sparse motion estimator with feature
// detection and optical-flow tracking, using defaults tuned for general video.
func NewKeypointBasedMotionEstimator(base MotionEstimatorBase) *KeypointBasedMotionEstimator {
	return &KeypointBasedMotionEstimator{
		base:         base,
		MaxCorners:   1000,
		QualityLevel: 0.01,
		MinDistance:  5,
		BlockSize:    3,
		WinSize:      21,
		MaxLevel:     3,
	}
}

// MotionModel returns the model of the wrapped estimator.
func (k *KeypointBasedMotionEstimator) MotionModel() MotionModel {
	return k.base.MotionModel()
}

// Base returns the wrapped sparse motion estimator.
func (k *KeypointBasedMotionEstimator) Base() MotionEstimatorBase { return k.base }

// Estimate detects features in frame0, tracks them into frame1 and fits the
// wrapped model to the surviving correspondences.
func (k *KeypointBasedMotionEstimator) Estimate(frame0, frame1 *cv.Mat) (Motion, bool) {
	if frame0 == nil || frame1 == nil || frame0.Empty() || frame1.Empty() {
		return IdentityMotion(), false
	}
	gray0 := grayscale(frame0)
	pts := cv.GoodFeaturesToTrack(gray0, k.MaxCorners, k.QualityLevel, k.MinDistance, k.BlockSize)
	if len(pts) < k.base.MotionModel().minPoints() {
		return IdentityMotion(), false
	}
	nextPts, status, _ := video.CalcOpticalFlowPyrLK(frame0, frame1, pts, k.WinSize, k.MaxLevel)

	from := make([]video.PointF, 0, len(pts))
	to := make([]video.PointF, 0, len(pts))
	for i := range pts {
		if !status[i] {
			continue
		}
		from = append(from, video.PointF{X: float64(pts[i].X), Y: float64(pts[i].Y)})
		to = append(to, video.PointF{X: float64(nextPts[i].X), Y: float64(nextPts[i].Y)})
	}
	if len(from) < k.base.MotionModel().minPoints() {
		return IdentityMotion(), false
	}
	return k.base.Estimate(from, to)
}

// GetMotion composes the per-consecutive-frame motions to obtain the transform
// mapping coordinates of frame from into coordinates of frame to. motions[i] is
// the motion between frame i and frame i+1 (mapping frame i onto frame i+1), so
// GetMotion(a, b) = M_{b-1}·…·M_a when b > a. When from == to the identity is
// returned; when from > to the forward composition is inverted. This mirrors
// cv::videostab::getMotion.
func GetMotion(from, to int, motions []Motion) Motion {
	m := IdentityMotion()
	switch {
	case to > from:
		// Compose M_from … M_{to-1}: maps frame `from` onto frame `to`.
		for i := from; i < to; i++ {
			m = motions[i].Mul(m)
		}
		return m
	case from > to:
		// Compose the reverse span, then invert to map `from` onto `to`.
		for i := to; i < from; i++ {
			m = motions[i].Mul(m)
		}
		if inv, ok := m.Inverse(); ok {
			return inv
		}
		return IdentityMotion()
	default:
		return m
	}
}

// grayscale returns a single-channel copy of the frame.
func grayscale(m *cv.Mat) *cv.Mat {
	if m.Channels == 1 {
		return m.Clone()
	}
	if m.Channels == 3 {
		return cv.CvtColor(m, cv.ColorRGB2Gray)
	}
	out := cv.NewMat(m.Rows, m.Cols, 1)
	for p := 0; p < m.Total(); p++ {
		out.Data[p] = m.Data[p*m.Channels]
	}
	return out
}
