package rgbd

import (
	"sort"

	cv "github.com/malcolmston/opencv"
)

// DepthCleaner removes the two dominant artefacts of consumer depth sensors:
// small holes (missing measurements) and speckle noise. It mirrors the intent
// of OpenCV's rgbd::DepthCleaner while using a robust, deterministic
// neighbourhood scheme.
//
// A cleaner is configured once and reused across frames. The zero value is not
// usable; build one with [NewDepthCleaner].
type DepthCleaner struct {
	// Window is the half-size of the square neighbourhood examined around each
	// pixel; the full window is (2·Window+1)².
	Window int
	// MaxDepth clamps the plausible depth range: measurements above it are
	// treated as invalid. A value ≤ 0 disables the clamp.
	MaxDepth float64
}

// NewDepthCleaner returns a [DepthCleaner] with the given neighbourhood
// half-size and no maximum-depth clamp. It panics if window is not positive.
func NewDepthCleaner(window int) *DepthCleaner {
	if window <= 0 {
		panic("rgbd: NewDepthCleaner requires a positive window")
	}
	return &DepthCleaner{Window: window}
}

// Clean returns a cleaned copy of depth. Valid pixels are replaced by the median
// of the valid depths in their window (a robust despeckling that preserves
// edges better than a mean), and invalid pixels are filled with that median
// whenever the window holds enough valid neighbours to trust it (at least a
// quarter of the window). Pixels that remain unsupported stay invalid (zero).
//
// The routine is deterministic and does not modify its input. It panics if depth
// is nil/empty.
func (c *DepthCleaner) Clean(depth *cv.FloatMat) *cv.FloatMat {
	if depth == nil || len(depth.Data) == 0 {
		panic("rgbd: DepthCleaner.Clean given an empty depth map")
	}
	out := cv.NewFloatMat(depth.Rows, depth.Cols)
	win := c.Window
	full := (2*win + 1) * (2*win + 1)
	minFill := full / 4
	if minFill < 1 {
		minFill = 1
	}
	buf := make([]float64, 0, full)
	valid := func(z float64) bool {
		if z <= 0 {
			return false
		}
		if c.MaxDepth > 0 && z > c.MaxDepth {
			return false
		}
		return true
	}
	for v := 0; v < depth.Rows; v++ {
		for u := 0; u < depth.Cols; u++ {
			buf = buf[:0]
			for dv := -win; dv <= win; dv++ {
				vv := v + dv
				if vv < 0 || vv >= depth.Rows {
					continue
				}
				for du := -win; du <= win; du++ {
					uu := u + du
					if uu < 0 || uu >= depth.Cols {
						continue
					}
					z := depth.At(vv, uu)
					if valid(z) {
						buf = append(buf, z)
					}
				}
			}
			center := depth.At(v, u)
			switch {
			case valid(center):
				out.Data[v*depth.Cols+u] = median(buf)
			case len(buf) >= minFill:
				out.Data[v*depth.Cols+u] = median(buf)
			default:
				// Leave as invalid (zero): not enough support to fill.
			}
		}
	}
	return out
}

// median returns the median of vs, which must be non-empty. It sorts a local
// copy so the caller's slice order is preserved.
func median(vs []float64) float64 {
	tmp := make([]float64, len(vs))
	copy(tmp, vs)
	sort.Float64s(tmp)
	n := len(tmp)
	if n%2 == 1 {
		return tmp[n/2]
	}
	return 0.5 * (tmp[n/2-1] + tmp[n/2])
}
