package videoproc

import (
	cv "github.com/malcolmston/opencv"
)

// WarpTranslate returns src shifted by (dx, dy): the output pixel at (x, y) is
// sampled from src at (x-dx, y-dy) with bilinear interpolation and edge
// clamping, so a positive dx moves image content to the right. It is the
// compensation warp applied by the [Stabilizer]. It panics on an empty frame.
func WarpTranslate(src *cv.Mat, dx, dy float64) *cv.Mat {
	if src == nil || src.Empty() {
		panic("videoproc: WarpTranslate requires a non-empty frame")
	}
	out := cv.NewMat(src.Rows, src.Cols, src.Channels)
	for y := 0; y < src.Rows; y++ {
		for x := 0; x < src.Cols; x++ {
			sx := float64(x) - dx
			sy := float64(y) - dy
			oi := (y*out.Cols + x) * out.Channels
			for c := 0; c < out.Channels; c++ {
				out.Data[oi+c] = videoprocClampU8(videoprocSampleBilinear(src, sx, sy, c) + 0.5)
			}
		}
	}
	return out
}

// EstimateGlobalTranslation estimates the dominant inter-frame translation from
// prev to cur by a feature-based method: it detects up to maxCorners strong
// corners in prev with cv.GoodFeaturesToTrack, tracks each into cur with
// [TrackPoints] (block matching), and returns the component-wise median of the
// surviving displacements. The median rejects outliers from independently moving
// objects, isolating global camera motion. The boolean result is false when too
// few features could be tracked. It panics on an empty frame or size mismatch.
func EstimateGlobalTranslation(prev, cur *cv.Mat, maxCorners, searchRadius int) (dx, dy float64, ok bool) {
	if prev == nil || cur == nil || prev.Empty() || cur.Empty() {
		panic("videoproc: EstimateGlobalTranslation requires two non-empty frames")
	}
	if prev.Rows != cur.Rows || prev.Cols != cur.Cols {
		panic("videoproc: EstimateGlobalTranslation frame size mismatch")
	}
	if maxCorners < 1 {
		panic("videoproc: EstimateGlobalTranslation requires maxCorners >= 1")
	}
	if searchRadius < 1 {
		searchRadius = 1
	}
	gp := videoprocToGray(prev)
	corners := cv.GoodFeaturesToTrack(gp, maxCorners, 0.01, 5, 3)
	if len(corners) < 3 {
		// fall back to a coarse grid so degenerate textures still yield motion.
		grid := SampleDenseGrid(gp.Rows, gp.Cols, maxIntStab(gp.Rows, gp.Cols)/8+1)
		corners = corners[:0]
		for _, g := range grid {
			corners = append(corners, g)
		}
	}
	pts := make([]PointF, len(corners))
	for i, c := range corners {
		pts[i] = PointF{X: float64(c.X), Y: float64(c.Y)}
	}
	tracked, valid := TrackPoints(prev, cur, pts, searchRadius, 3, 25)
	var dxs, dys []float64
	for i := range pts {
		if !valid[i] {
			continue
		}
		dxs = append(dxs, tracked[i].X-pts[i].X)
		dys = append(dys, tracked[i].Y-pts[i].Y)
	}
	if len(dxs) < 3 {
		return 0, 0, false
	}
	return videoprocMedianF(dxs), videoprocMedianF(dys), true
}

// maxIntStab returns the larger of two ints.
func maxIntStab(a, b int) int {
	if a > b {
		return a
	}
	return b
}

// SmoothTrajectory smooths a cumulative motion trajectory with a centred moving
// average of the given radius (window size 2*radius+1), clamping the window at
// the endpoints. It is applied to the accumulated per-frame translations to
// produce the intended smooth camera path; the difference between the smoothed
// and raw paths gives the per-frame stabilization correction. radius must be
// >= 0; radius 0 returns a copy unchanged. It panics on a negative radius.
func SmoothTrajectory(traj []PointF, radius int) []PointF {
	if radius < 0 {
		panic("videoproc: SmoothTrajectory requires radius >= 0")
	}
	out := make([]PointF, len(traj))
	for i := range traj {
		var sx, sy float64
		n := 0
		for k := i - radius; k <= i+radius; k++ {
			if k < 0 || k >= len(traj) {
				continue
			}
			sx += traj[k].X
			sy += traj[k].Y
			n++
		}
		if n > 0 {
			out[i] = PointF{X: sx / float64(n), Y: sy / float64(n)}
		}
	}
	return out
}

// Stabilizer is a causal (online) digital video stabilizer. For each incoming
// frame it estimates the inter-frame translation with
// [EstimateGlobalTranslation], accumulates it into a running camera path,
// low-pass filters that path with an exponential moving average to obtain the
// intended smooth path, and warps the frame by the difference so the output
// appears to follow the smooth path. Because it is causal it uses only past
// frames and introduces no latency.
type Stabilizer struct {
	// Smoothing is the EMA weight in (0,1] applied to the cumulative path;
	// smaller means a smoother (steadier) but less responsive result.
	Smoothing float64
	// MaxCorners is the number of features tracked per frame.
	MaxCorners int
	// SearchRadius is the block-matching search radius for tracking.
	SearchRadius int

	prev      *cv.Mat
	cumX      float64
	cumY      float64
	smoothX   float64
	smoothY   float64
	started   bool
	lastValid bool
}

// NewStabilizer returns a stabilizer with the given path-smoothing weight (in
// (0,1]) and OpenCV-like tracking defaults (200 corners, search radius 8). It
// panics on an out-of-range smoothing weight.
func NewStabilizer(smoothing float64) *Stabilizer {
	if smoothing <= 0 || smoothing > 1 {
		panic("videoproc: NewStabilizer requires smoothing in (0,1]")
	}
	return &Stabilizer{Smoothing: smoothing, MaxCorners: 200, SearchRadius: 8}
}

// Stabilize feeds the next frame and returns the motion-compensated frame. The
// first frame is returned unchanged and initialises the reference. Subsequent
// frames are warped by the gap between the smoothed and the raw camera path. The
// boolean result reports whether inter-frame motion was successfully estimated
// for this frame (false means the previous correction was reused). Frame
// dimensions must stay constant across calls.
func (s *Stabilizer) Stabilize(frame *cv.Mat) (*cv.Mat, bool) {
	if frame == nil || frame.Empty() {
		panic("videoproc: Stabilizer.Stabilize requires a non-empty frame")
	}
	if !s.started {
		s.started = true
		s.prev = frame.Clone()
		return frame.Clone(), true
	}
	dx, dy, ok := EstimateGlobalTranslation(s.prev, frame, s.MaxCorners, s.SearchRadius)
	if ok {
		s.cumX += dx
		s.cumY += dy
	}
	// exponential smoothing of the cumulative (raw) path.
	s.smoothX = (1-s.Smoothing)*s.smoothX + s.Smoothing*s.cumX
	s.smoothY = (1-s.Smoothing)*s.smoothY + s.Smoothing*s.cumY
	// correction moves current content from raw path toward smooth path.
	corrX := s.smoothX - s.cumX
	corrY := s.smoothY - s.cumY
	out := WarpTranslate(frame, corrX, corrY)
	s.prev = frame.Clone()
	return out, ok
}
