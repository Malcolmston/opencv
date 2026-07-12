package hdr

import "math"

// Response returns the linear scene radiance that channel c maps the 8-bit pixel
// value z to. It panics if c or z is out of range.
func (r *CameraResponse) Response(c, z int) float64 {
	if c < 0 || c >= r.Channels {
		panic("hdr: Response channel out of range")
	}
	if z < 0 || z > 255 {
		panic("hdr: Response value out of range")
	}
	return r.Curve[c][z]
}

// ChannelCurve returns a copy of the 256-entry response lookup table for channel
// c, so callers can inspect or plot it without risking mutation of the stored
// curve. It panics if c is out of range.
func (r *CameraResponse) ChannelCurve(c int) []float64 {
	if c < 0 || c >= r.Channels {
		panic("hdr: ChannelCurve channel out of range")
	}
	out := make([]float64, 256)
	copy(out, r.Curve[c])
	return out
}

// LogResponse returns the natural logarithm of channel c's response curve — the
// Debevec g(z) function — as a fresh 256-entry slice. It panics if c is out of
// range.
func (r *CameraResponse) LogResponse(c int) []float64 {
	if c < 0 || c >= r.Channels {
		panic("hdr: LogResponse channel out of range")
	}
	return r.logCurve(c)
}

// IsMonotonic reports whether every channel's response curve is non-decreasing
// to within tol (a small negative slope up to tol is tolerated). A well-behaved
// camera response is monotonic; this is a cheap sanity check after calibration.
func (r *CameraResponse) IsMonotonic(tol float64) bool {
	if tol < 0 {
		tol = -tol
	}
	for c := 0; c < r.Channels; c++ {
		curve := r.Curve[c]
		for z := 1; z < len(curve); z++ {
			if curve[z] < curve[z-1]-tol {
				return false
			}
		}
	}
	return true
}

// EnforceMonotonic repairs each channel's response curve in place so it becomes
// non-decreasing and strictly positive, forward-filling any non-positive or NaN
// entries. It is useful after a noisy calibration before the curve is used for
// merging.
func (r *CameraResponse) EnforceMonotonic() {
	for c := 0; c < r.Channels; c++ {
		fillMonotone(r.Curve[c])
	}
}

// Normalize rescales every channel so that its mid-range entry (value 128) is
// exactly 1, matching the convention of [CalibrateRobertson]. Because a response
// is only defined up to a global factor, this makes two responses directly
// comparable. Channels whose entry 128 is non-positive are left unchanged.
func (r *CameraResponse) Normalize() {
	for c := 0; c < r.Channels; c++ {
		mid := r.Curve[c][128]
		if mid <= 0 || math.IsNaN(mid) {
			continue
		}
		for z := 0; z < len(r.Curve[c]); z++ {
			r.Curve[c][z] /= mid
		}
	}
}
