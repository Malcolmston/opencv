package video

import (
	"math"
	"sort"

	cv "github.com/malcolmston/opencv"
)

// SimilarityTransform is a 2-D partial-affine (similarity) transform: a uniform
// Scale, a rotation Angle (radians, counter-clockwise in image coordinates) and
// a translation (Tx, Ty). It maps a point (x, y) to
//
//	x' = Scale*cos(Angle)*x - Scale*sin(Angle)*y + Tx
//	y' = Scale*sin(Angle)*x + Scale*cos(Angle)*y + Ty
//
// It is the model recovered by [EstimateAffinePartial2D].
type SimilarityTransform struct {
	Scale float64
	Angle float64
	Tx    float64
	Ty    float64
}

// EstimateAffinePartial2D estimates the best-fit similarity transform (uniform
// scale + rotation + translation, four degrees of freedom) mapping the points in
// from to the points in to, mirroring cv::estimateAffinePartial2D. It solves the
// linear least-squares problem in closed form via the 4x4 normal equations, so
// the fit is optimal in the total squared-error sense (no RANSAC / outlier
// rejection is performed).
//
// from and to must have equal length and contain at least two non-degenerate
// correspondences. The boolean result is false when there are too few points or
// the normal system is singular (all source points coincident).
func EstimateAffinePartial2D(from, to []PointF) (SimilarityTransform, bool) {
	if len(from) != len(to) || len(from) < 2 {
		return SimilarityTransform{}, false
	}
	// Unknowns are (a, b, c, d) with a = s*cos, b = s*sin, c = Tx, d = Ty.
	// Each point contributes two rows:
	//   [ fx -fy 1 0 ]·x = tox
	//   [ fy  fx 0 1 ]·x = toy
	var sxx, sx, sy float64
	n := float64(len(from))
	var ba, bb, bc, bd float64
	for i := range from {
		fx, fy := from[i].X, from[i].Y
		tx, ty := to[i].X, to[i].Y
		sxx += fx*fx + fy*fy
		sx += fx
		sy += fy
		ba += fx*tx + fy*ty
		bb += -fy*tx + fx*ty
		bc += tx
		bd += ty
	}
	// Normal matrix (symmetric), derived analytically from the two row templates.
	m := [][]float64{
		{sxx, 0, sx, sy},
		{0, sxx, -sy, sx},
		{sx, -sy, n, 0},
		{sy, sx, 0, n},
	}
	inv, ok := matInverse(m)
	if !ok {
		return SimilarityTransform{}, false
	}
	sol := matVec(inv, []float64{ba, bb, bc, bd})
	a, b, c, d := sol[0], sol[1], sol[2], sol[3]
	return SimilarityTransform{
		Scale: math.Hypot(a, b),
		Angle: math.Atan2(b, a),
		Tx:    c,
		Ty:    d,
	}, true
}

// VideoStabilizer is a causal digital video stabilizer. For each incoming frame
// it estimates the inter-frame motion (translation and rotation) by tracking a
// grid of features with [CalcOpticalFlowPyrLK], accumulates that motion into a
// camera trajectory, smooths the trajectory with a trailing moving average of
// radius SmoothingRadius, and warps the frame to follow the smoothed path. This
// is the classic cumulative-trajectory smoothing scheme used by OpenCV's
// videostab sample.
//
// Construct one with [NewVideoStabilizer] and feed frames one at a time to
// [VideoStabilizer.Stabilize]. Because the smoother is causal (it only sees past
// and current frames) the first frame is returned unchanged.
type VideoStabilizer struct {
	// SmoothingRadius is the trailing window radius for trajectory smoothing.
	SmoothingRadius int
	// WinSize and MaxLevel configure the Lucas-Kanade motion estimator.
	WinSize  int
	MaxLevel int

	prev  *cv.Mat
	rows  int
	cols  int
	cumX  float64
	cumY  float64
	cumA  float64
	trajX []float64
	trajY []float64
	trajA []float64
}

// NewVideoStabilizer creates a stabilizer with the given trailing smoothing
// radius (in frames). radius must be >= 0; 0 disables smoothing (frames pass
// through unchanged).
func NewVideoStabilizer(radius int) *VideoStabilizer {
	if radius < 0 {
		panic("video: NewVideoStabilizer requires radius >= 0")
	}
	return &VideoStabilizer{SmoothingRadius: radius, WinSize: 21, MaxLevel: 3}
}

// gridPoints returns an evenly spaced grid of interior feature points for motion
// estimation.
func gridPoints(rows, cols int) []cv.Point {
	var pts []cv.Point
	stepX := cols / 12
	stepY := rows / 12
	if stepX < 4 {
		stepX = 4
	}
	if stepY < 4 {
		stepY = 4
	}
	for y := stepY; y < rows-stepY; y += stepY {
		for x := stepX; x < cols-stepX; x += stepX {
			pts = append(pts, cv.Point{X: x, Y: y})
		}
	}
	return pts
}

// median returns the median of xs (which it sorts in place). It panics on an
// empty slice.
func median(xs []float64) float64 {
	sort.Float64s(xs)
	n := len(xs)
	if n%2 == 1 {
		return xs[n/2]
	}
	return 0.5 * (xs[n/2-1] + xs[n/2])
}

// estimateMotion estimates the similarity motion from prev to next by tracking a
// grid of points. It rejects diverged tracks by keeping only points whose
// displacement is close to the median displacement, then fits a similarity
// transform to the surviving inliers. It returns (dx, dy, dAngle); on failure it
// returns zeros.
func (s *VideoStabilizer) estimateMotion(prev, next *cv.Mat) (float64, float64, float64) {
	pts := gridPoints(prev.Rows, prev.Cols)
	if len(pts) < 3 {
		return 0, 0, 0
	}
	nextPts, status, _ := CalcOpticalFlowPyrLK(prev, next, pts, s.WinSize, s.MaxLevel)

	var idx []int
	var dxs, dys []float64
	for i := range pts {
		if status[i] {
			idx = append(idx, i)
			dxs = append(dxs, float64(nextPts[i].X-pts[i].X))
			dys = append(dys, float64(nextPts[i].Y-pts[i].Y))
		}
	}
	if len(idx) < 3 {
		return 0, 0, 0
	}
	// Robust translation estimate from the median displacement, used to reject
	// diverged tracks before the similarity fit.
	medX := median(append([]float64(nil), dxs...))
	medY := median(append([]float64(nil), dys...))
	const inlierTol = 2.5

	var from, to []PointF
	for j, i := range idx {
		if abs(dxs[j]-medX) <= inlierTol && abs(dys[j]-medY) <= inlierTol {
			from = append(from, PointF{X: float64(pts[i].X), Y: float64(pts[i].Y)})
			to = append(to, PointF{X: float64(nextPts[i].X), Y: float64(nextPts[i].Y)})
		}
	}
	tf, ok := EstimateAffinePartial2D(from, to)
	if !ok {
		return medX, medY, 0
	}
	return tf.Tx, tf.Ty, tf.Angle
}

// smoothedTrajectory returns the trailing moving average of the accumulated
// trajectory at the latest sample.
func (s *VideoStabilizer) smoothedTrajectory() (float64, float64, float64) {
	k := len(s.trajX) - 1
	lo := k - s.SmoothingRadius
	if lo < 0 {
		lo = 0
	}
	var sxv, syv, sav float64
	var n float64
	for i := lo; i <= k; i++ {
		sxv += s.trajX[i]
		syv += s.trajY[i]
		sav += s.trajA[i]
		n++
	}
	return sxv / n, syv / n, sav / n
}

// Stabilize consumes the next frame of the sequence and returns its stabilized
// version (same dimensions and channel count). The first frame establishes the
// reference and is returned as a clone; later frames are motion-compensated
// toward the smoothed trajectory. It panics if a frame's size differs from the
// first one or the frame is empty.
func (s *VideoStabilizer) Stabilize(frame *cv.Mat) *cv.Mat {
	if frame == nil || frame.Empty() {
		panic("video: VideoStabilizer.Stabilize requires a non-empty frame")
	}
	gray := toGray(frame)
	if s.prev == nil {
		s.prev = gray
		s.rows, s.cols = frame.Rows, frame.Cols
		s.trajX = []float64{0}
		s.trajY = []float64{0}
		s.trajA = []float64{0}
		return frame.Clone()
	}
	if frame.Rows != s.rows || frame.Cols != s.cols {
		panic("video: VideoStabilizer.Stabilize frame size changed")
	}

	dx, dy, da := s.estimateMotion(s.prev, gray)
	s.cumX += dx
	s.cumY += dy
	s.cumA += da
	s.trajX = append(s.trajX, s.cumX)
	s.trajY = append(s.trajY, s.cumY)
	s.trajA = append(s.trajA, s.cumA)

	smX, smY, smA := s.smoothedTrajectory()
	// Warp the frame so its apparent cumulative position becomes the smoothed
	// one instead of the actual one: the compensating transform is exactly the
	// difference (smoothed - actual) of the two trajectories.
	tX := smX - s.cumX
	tY := smY - s.cumY
	tA := smA - s.cumA

	cos := math.Cos(tA)
	sin := math.Sin(tA)
	warp := cv.AffineMatrix{cos, -sin, tX, sin, cos, tY}
	out := cv.WarpAffine(frame, warp, s.cols, s.rows, cv.InterLinear)

	s.prev = gray
	return out
}
