package videoproc

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// MotionHistory maintains a Motion History Image (MHI) as described by Bobick &
// Davis. Each pixel stores the most recent timestamp at which motion (a non-zero
// silhouette) was observed there. Recent motion appears bright and fades to zero
// over a fixed Duration, encoding both where and when movement occurred and
// giving the temporal template used for motion-based action recognition.
type MotionHistory struct {
	// Duration is the time window (in the same units as the timestamps passed to
	// Update) over which motion remains visible before decaying to zero.
	Duration float64

	rows  int
	cols  int
	mhi   *cv.FloatMat
	stamp float64
}

// NewMotionHistory allocates a rows×cols motion-history buffer with the given
// fade Duration. It panics on non-positive dimensions or duration.
func NewMotionHistory(rows, cols int, duration float64) *MotionHistory {
	if rows <= 0 || cols <= 0 {
		panic("videoproc: NewMotionHistory requires positive dimensions")
	}
	if duration <= 0 {
		panic("videoproc: NewMotionHistory requires positive duration")
	}
	return &MotionHistory{
		Duration: duration,
		rows:     rows,
		cols:     cols,
		mhi:      cv.NewFloatMat(rows, cols),
	}
}

// Update folds a new silhouette (a single-channel mask where non-zero marks
// motion) into the history at the given timestamp. Pixels that are moving are
// set to timestamp; pixels whose stored time is older than timestamp-Duration
// are cleared to zero. Timestamps must be non-decreasing across calls. It panics
// if silhouette is not single-channel or has the wrong size.
func (h *MotionHistory) Update(silhouette *cv.Mat, timestamp float64) {
	if silhouette == nil || silhouette.Empty() || silhouette.Channels != 1 {
		panic("videoproc: MotionHistory.Update requires a single-channel silhouette")
	}
	if silhouette.Rows != h.rows || silhouette.Cols != h.cols {
		panic("videoproc: MotionHistory.Update silhouette size mismatch")
	}
	h.stamp = timestamp
	UpdateMotionHistory(silhouette, h.mhi, timestamp, h.Duration)
}

// Image returns the underlying motion-history buffer (timestamps per pixel) as a
// FloatMat. The returned matrix is the live buffer; callers that need to retain
// it across further Update calls should copy it.
func (h *MotionHistory) Image() *cv.FloatMat {
	return h.mhi
}

// NormalizedImage returns the motion-history buffer scaled to 0..255, with the
// newest motion at 255 and fully faded pixels at 0, suitable for display. Pixels
// with no recorded motion are 0.
func (h *MotionHistory) NormalizedImage() *cv.Mat {
	out := cv.NewMat(h.rows, h.cols, 1)
	if h.Duration <= 0 {
		return out
	}
	for i, t := range h.mhi.Data {
		if t <= 0 {
			continue
		}
		age := h.stamp - t
		v := (h.Duration - age) / h.Duration
		if v < 0 {
			v = 0
		}
		out.Data[i] = videoprocClampU8(v*255 + 0.5)
	}
	return out
}

// EnergyImage returns the Motion Energy Image (MEI): a binary mask (255 where any
// non-faded motion is currently recorded, 0 elsewhere). It marks the spatial
// region touched by recent motion regardless of when within the window it
// occurred.
func (h *MotionHistory) EnergyImage() *cv.Mat {
	out := cv.NewMat(h.rows, h.cols, 1)
	for i, t := range h.mhi.Data {
		if t > 0 {
			out.Data[i] = 255
		}
	}
	return out
}

// UpdateMotionHistory applies one motion-history update step in place, mirroring
// cv::updateMotionHistory. For each pixel: if the silhouette is non-zero the mhi
// value becomes timestamp; otherwise, if the stored value is older than
// timestamp-duration it is reset to zero, and if not it is left unchanged. mhi
// must be single-channel-sized to match silhouette. It panics on a size or
// channel mismatch.
func UpdateMotionHistory(silhouette *cv.Mat, mhi *cv.FloatMat, timestamp, duration float64) {
	if silhouette == nil || silhouette.Empty() || silhouette.Channels != 1 {
		panic("videoproc: UpdateMotionHistory requires a single-channel silhouette")
	}
	if mhi == nil || mhi.Rows != silhouette.Rows || mhi.Cols != silhouette.Cols {
		panic("videoproc: UpdateMotionHistory mhi size mismatch")
	}
	cutoff := timestamp - duration
	for i := range mhi.Data {
		if silhouette.Data[i] != 0 {
			mhi.Data[i] = timestamp
		} else if mhi.Data[i] < cutoff {
			mhi.Data[i] = 0
		}
	}
}

// MotionGradient computes the spatial gradient orientation of a motion-history
// image, mirroring cv::calcMotionGradient. It returns a per-pixel orientation in
// degrees ([0,360)) and a validity mask (255 where the local gradient magnitude
// lies within [delta1, delta2], 0 elsewhere). Only pixels flagged in the mask
// carry a meaningful orientation. delta1 and delta2 bound the acceptable
// temporal gradient magnitude; they must satisfy 0 <= delta1 <= delta2.
func MotionGradient(mhi *cv.FloatMat, delta1, delta2 float64) (orientation *cv.FloatMat, mask *cv.Mat) {
	if mhi == nil || mhi.Rows == 0 || mhi.Cols == 0 {
		panic("videoproc: MotionGradient requires a non-empty mhi")
	}
	if delta1 < 0 || delta2 < delta1 {
		panic("videoproc: MotionGradient requires 0 <= delta1 <= delta2")
	}
	rows, cols := mhi.Rows, mhi.Cols
	orientation = cv.NewFloatMat(rows, cols)
	mask = cv.NewMat(rows, cols, 1)
	at := func(y, x int) float64 {
		if y < 0 {
			y = 0
		} else if y >= rows {
			y = rows - 1
		}
		if x < 0 {
			x = 0
		} else if x >= cols {
			x = cols - 1
		}
		return mhi.Data[y*cols+x]
	}
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			dx := (at(y, x+1) - at(y, x-1)) * 0.5
			dy := (at(y+1, x) - at(y-1, x)) * 0.5
			mag := math.Hypot(dx, dy)
			if mag >= delta1 && mag <= delta2 && mag > 0 {
				deg := math.Atan2(dy, dx) * 180 / math.Pi
				if deg < 0 {
					deg += 360
				}
				orientation.Data[y*cols+x] = deg
				mask.Data[y*cols+x] = 255
			}
		}
	}
	return orientation, mask
}

// GlobalMotionOrientation returns the dominant motion direction in degrees
// ([0,360)) as the circular (vector) mean of the valid per-pixel orientations
// produced by [MotionGradient], weighted equally. The boolean result is false
// when the mask flags no valid gradients. This is a simplified analogue of
// cv::calcGlobalOrientation without recency weighting.
func GlobalMotionOrientation(orientation *cv.FloatMat, mask *cv.Mat) (float64, bool) {
	if orientation == nil || mask == nil {
		panic("videoproc: GlobalMotionOrientation requires orientation and mask")
	}
	if orientation.Rows != mask.Rows || orientation.Cols != mask.Cols || mask.Channels != 1 {
		panic("videoproc: GlobalMotionOrientation size/channel mismatch")
	}
	var sx, sy float64
	n := 0
	for i := range orientation.Data {
		if mask.Data[i] == 0 {
			continue
		}
		rad := orientation.Data[i] * math.Pi / 180
		sx += math.Cos(rad)
		sy += math.Sin(rad)
		n++
	}
	if n == 0 {
		return 0, false
	}
	deg := math.Atan2(sy, sx) * 180 / math.Pi
	if deg < 0 {
		deg += 360
	}
	return deg, true
}
