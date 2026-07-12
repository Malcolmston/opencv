package optflow

import (
	"math"
	"sort"
)

// requireSameShape panics unless both fields are non-nil and identically sized.
func requireSameShape(a, b *FlowField, fn string) {
	if a == nil || b == nil {
		panic("optflow: " + fn + " requires non-nil fields")
	}
	if a.Rows != b.Rows || a.Cols != b.Cols {
		panic("optflow: " + fn + " requires equally sized fields")
	}
}

// EndpointError returns the per-pixel endpoint error between an estimated flow
// and a ground-truth flow: for each pixel the Euclidean distance
//
//	sqrt((ue − ug)² + (ve − vg)²)
//
// between the two displacement vectors. The result is a row-major slice of
// length Rows*Cols. Endpoint error (EE, and its average AEE below) is the
// standard Middlebury / MPI-Sintel accuracy measure for optical flow.
//
// est and gt must be non-nil and identically sized.
func EndpointError(est, gt *FlowField) []float64 {
	requireSameShape(est, gt, "EndpointError")
	n := est.Rows * est.Cols
	out := make([]float64, n)
	for i := 0; i < n; i++ {
		du := est.Data[i*2] - gt.Data[i*2]
		dv := est.Data[i*2+1] - gt.Data[i*2+1]
		out[i] = math.Hypot(du, dv)
	}
	return out
}

// AverageEndpointError returns the mean endpoint error (AEE) over the whole
// field — the single most common scalar accuracy score for optical flow. It is
// the arithmetic mean of [EndpointError].
//
// est and gt must be non-nil and identically sized.
func AverageEndpointError(est, gt *FlowField) float64 {
	ee := EndpointError(est, gt)
	if len(ee) == 0 {
		return 0
	}
	var s float64
	for _, e := range ee {
		s += e
	}
	return s / float64(len(ee))
}

// AngularError returns the per-pixel angular error (in radians) between an
// estimated and a ground-truth flow. Following the Barron/Middlebury
// convention each displacement (u, v) is embedded as the space-time direction
// (u, v, 1) and the error is the angle between the two normalised vectors:
//
//	arccos( (ue·ug + ve·vg + 1) / (√(ue²+ve²+1) · √(ug²+vg²+1)) )
//
// The +1 temporal component keeps the measure well defined even where a flow is
// zero, which a plain 2-D vector angle cannot. The result is a row-major slice
// of length Rows*Cols with values in [0, π].
//
// est and gt must be non-nil and identically sized.
func AngularError(est, gt *FlowField) []float64 {
	requireSameShape(est, gt, "AngularError")
	n := est.Rows * est.Cols
	out := make([]float64, n)
	for i := 0; i < n; i++ {
		ue := est.Data[i*2]
		ve := est.Data[i*2+1]
		ug := gt.Data[i*2]
		vg := gt.Data[i*2+1]
		dot := ue*ug + ve*vg + 1.0
		den := math.Sqrt(ue*ue+ve*ve+1.0) * math.Sqrt(ug*ug+vg*vg+1.0)
		c := dot / den
		if c > 1 {
			c = 1
		} else if c < -1 {
			c = -1
		}
		out[i] = math.Acos(c)
	}
	return out
}

// AverageAngularError returns the mean angular error (AAE, in radians) over the
// whole field. It is the arithmetic mean of [AngularError].
//
// est and gt must be non-nil and identically sized.
func AverageAngularError(est, gt *FlowField) float64 {
	ae := AngularError(est, gt)
	if len(ae) == 0 {
		return 0
	}
	var s float64
	for _, e := range ae {
		s += e
	}
	return s / float64(len(ae))
}

// ErrorStats summarises a per-pixel error distribution.
type ErrorStats struct {
	// Mean is the arithmetic mean error (AEE for endpoint errors).
	Mean float64
	// RMS is the root-mean-square error.
	RMS float64
	// Median is the 50th-percentile error.
	Median float64
	// Max is the largest error.
	Max float64
}

// EndpointErrorStats computes summary statistics (mean, RMS, median and max) of
// the endpoint error between an estimated and a ground-truth flow in a single
// pass plus one sort. It is a convenience over [EndpointError] for reporting.
//
// est and gt must be non-nil and identically sized.
func EndpointErrorStats(est, gt *FlowField) ErrorStats {
	ee := EndpointError(est, gt)
	if len(ee) == 0 {
		return ErrorStats{}
	}
	var sum, sqSum, maxv float64
	for _, e := range ee {
		sum += e
		sqSum += e * e
		if e > maxv {
			maxv = e
		}
	}
	n := float64(len(ee))
	sorted := make([]float64, len(ee))
	copy(sorted, ee)
	sort.Float64s(sorted)
	var median float64
	m := len(sorted)
	if m%2 == 1 {
		median = sorted[m/2]
	} else {
		median = 0.5 * (sorted[m/2-1] + sorted[m/2])
	}
	return ErrorStats{
		Mean:   sum / n,
		RMS:    math.Sqrt(sqSum / n),
		Median: median,
		Max:    maxv,
	}
}
