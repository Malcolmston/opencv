package ximgproc

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// DTMode selects the filtering primitive used by [DTFilter] once the guidance
// image has been mapped through the edge-aware domain transform of Gastal and
// Oliveira ("Domain Transform for Edge-Aware Image and Video Processing",
// 2011). All three converge to the same isometric smoothing but trade accuracy
// for speed differently.
type DTMode int

const (
	// DTFilterNC is normalized convolution: a moving box average taken in the
	// transformed domain. It is the most accurate mode.
	DTFilterNC DTMode = iota
	// DTFilterIC is interpolated convolution: like NC but the box integrates the
	// piecewise-linear signal, so partial cells at the window ends contribute
	// their exact area. It is accurate and free of the small NC quantisation.
	DTFilterIC
	// DTFilterRF is recursive filtering: a first-order IIR pass in each
	// direction. It is the fastest mode and has an infinite (exponential)
	// impulse response rather than a finite box.
	DTFilterRF
)

// DTFilter applies the domain-transform edge-preserving filter to src, guided by
// the edges of guide, and returns a new Mat of the same shape as src.
//
// The domain transform warps each scan-line so that the arc length between
// neighbouring pixels grows with the guidance gradient: neighbours separated by
// a strong guide edge become far apart in the transformed domain and therefore
// barely mix, while neighbours inside a flat region stay close and are averaged.
// A 2-D filter is realised by alternating horizontal and vertical 1-D passes
// over iters iterations, halving the per-pass smoothing radius each time as
// prescribed by the paper so the result is very nearly isotropic.
//
// The per-step distance between neighbouring guide samples p and q is
//
//	1 + (sigmaSpatial/sigmaColor)·Σ_c |guide_p,c − guide_q,c|,
//
// so sigmaSpatial controls the spatial extent of smoothing (in pixels) and
// sigmaColor the edge-stopping sensitivity (on the native [0,255] scale). mode
// selects the 1-D primitive ([DTFilterNC], [DTFilterIC] or [DTFilterRF]); iters
// is the number of alternating passes (three is the paper's recommendation).
//
// guide and src must share width and height; guide may be 1- or 3-channel and
// need not match src's channel count. If guide is nil, src is used as its own
// guide (self-guided mode). It panics on a size mismatch, non-positive sigmas or
// iters < 1. The filter is fully deterministic.
func DTFilter(src, guide *cv.Mat, sigmaSpatial, sigmaColor float64, mode DTMode, iters int) *cv.Mat {
	if guide == nil {
		guide = src
	}
	if src.Rows != guide.Rows || src.Cols != guide.Cols {
		panic("ximgproc: DTFilter src and guide must share dimensions")
	}
	if sigmaSpatial <= 0 || sigmaColor <= 0 {
		panic("ximgproc: DTFilter requires positive sigmas")
	}
	if iters < 1 {
		panic("ximgproc: DTFilter requires iters >= 1")
	}
	rows, cols := src.Rows, src.Cols

	// Per-neighbour domain-transform distances derived from the guide.
	// dHor[y*cols+x] is the transformed distance between (x-1,y) and (x,y);
	// dVer[y*cols+x] is the distance between (x,y-1) and (x,y). Column 0 / row 0
	// entries are 1 (no left / upper neighbour, unit spatial step).
	ratio := sigmaSpatial / sigmaColor
	gp := planesFromMat(guide)
	dHor := make([]float64, rows*cols)
	dVer := make([]float64, rows*cols)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			i := y*cols + x
			dHor[i] = 1
			dVer[i] = 1
			if x > 0 {
				var s float64
				for c := range gp {
					s += math.Abs(gp[c][i] - gp[c][i-1])
				}
				dHor[i] = 1 + ratio*s
			}
			if y > 0 {
				var s float64
				for c := range gp {
					s += math.Abs(gp[c][i] - gp[c][i-cols])
				}
				dVer[i] = 1 + ratio*s
			}
		}
	}

	planes := planesFromMat(src)
	line := make([]float64, cols)
	if rows > cols {
		line = make([]float64, rows)
	}
	dist := make([]float64, len(line))

	for it := 0; it < iters; it++ {
		// Paper's per-iteration sigma_H schedule for the 1-D filters.
		sigmaH := sigmaSpatial * math.Sqrt(3) * math.Exp2(float64(iters-it-1)) /
			math.Sqrt(math.Exp2(float64(2*iters))-1)
		boxR := sigmaH * math.Sqrt(3)
		a := math.Exp(-math.Sqrt(2) / sigmaH)

		for c := range planes {
			p := planes[c]
			// Horizontal pass: each row.
			for y := 0; y < rows; y++ {
				base := y * cols
				copy(line[:cols], p[base:base+cols])
				copy(dist[:cols], dHor[base:base+cols])
				dtFilterLine(line[:cols], dist[:cols], mode, boxR, a)
				copy(p[base:base+cols], line[:cols])
			}
			// Vertical pass: each column.
			for x := 0; x < cols; x++ {
				for y := 0; y < rows; y++ {
					line[y] = p[y*cols+x]
					dist[y] = dVer[y*cols+x]
				}
				dtFilterLine(line[:rows], dist[:rows], mode, boxR, a)
				for y := 0; y < rows; y++ {
					p[y*cols+x] = line[y]
				}
			}
		}
	}

	return matFromPlanes(planes, rows, cols)
}

// dtFilterLine filters one 1-D line in place. dist[i] is the transformed
// distance between samples i-1 and i (dist[0] is ignored). mode picks the
// primitive; boxR is the transformed-domain box radius (NC/IC) and a the
// recursive feedback base (RF).
func dtFilterLine(f, dist []float64, mode DTMode, boxR, a float64) {
	switch mode {
	case DTFilterRF:
		rfLine(f, dist, a)
	case DTFilterIC:
		icLine(f, dist, boxR)
	default:
		ncLine(f, dist, boxR)
	}
}

// rfLine runs the recursive-filter primitive: a causal then anti-causal
// first-order IIR pass whose feedback a^dist adapts to the domain transform.
func rfLine(f, dist []float64, a float64) {
	n := len(f)
	for i := 1; i < n; i++ {
		w := math.Pow(a, dist[i])
		f[i] += w * (f[i-1] - f[i])
	}
	for i := n - 2; i >= 0; i-- {
		w := math.Pow(a, dist[i+1])
		f[i] += w * (f[i+1] - f[i])
	}
}

// ncLine runs normalized convolution: a moving box average over samples whose
// transformed position lies within boxR of the current sample.
func ncLine(f, dist []float64, boxR float64) {
	n := len(f)
	ct := make([]float64, n)
	for i := 1; i < n; i++ {
		ct[i] = ct[i-1] + dist[i]
	}
	prefix := make([]float64, n+1)
	for i := 0; i < n; i++ {
		prefix[i+1] = prefix[i] + f[i]
	}
	out := make([]float64, n)
	lo, hi := 0, 0
	for i := 0; i < n; i++ {
		left := ct[i] - boxR
		right := ct[i] + boxR
		for ct[lo] < left {
			lo++
		}
		if hi < i {
			hi = i
		}
		for hi < n && ct[hi] <= right {
			hi++
		}
		out[i] = (prefix[hi] - prefix[lo]) / float64(hi-lo)
	}
	copy(f, out)
}

// icLine runs interpolated convolution: it averages the piecewise-linear signal
// over the transformed window [ct-boxR, ct+boxR], integrating exact partial
// cells at the window ends via linear interpolation.
func icLine(f, dist []float64, boxR float64) {
	n := len(f)
	if n == 1 {
		return
	}
	ct := make([]float64, n)
	for i := 1; i < n; i++ {
		ct[i] = ct[i-1] + dist[i]
	}
	// area[i] = integral of the linear interpolant from ct[0] to ct[i].
	area := make([]float64, n)
	for i := 1; i < n; i++ {
		area[i] = area[i-1] + dist[i]*(f[i-1]+f[i])/2
	}
	out := make([]float64, n)
	for i := 0; i < n; i++ {
		lo := ct[i] - boxR
		if lo < ct[0] {
			lo = ct[0]
		}
		hi := ct[i] + boxR
		if hi > ct[n-1] {
			hi = ct[n-1]
		}
		w := hi - lo
		if w <= 0 {
			out[i] = f[i]
			continue
		}
		out[i] = (areaAt(ct, f, area, hi) - areaAt(ct, f, area, lo)) / w
	}
	copy(f, out)
}

// areaAt returns the integral of the piecewise-linear signal from ct[0] up to
// transformed position p (assumed within [ct[0], ct[n-1]]), using linear
// interpolation for the partial final cell.
func areaAt(ct, f, area []float64, p float64) float64 {
	n := len(ct)
	// Binary search for the cell [ct[j-1], ct[j]] containing p.
	lo, hi := 0, n-1
	for lo < hi {
		mid := (lo + hi + 1) / 2
		if ct[mid] <= p {
			lo = mid
		} else {
			hi = mid - 1
		}
	}
	j := lo // ct[j] <= p, and p <= ct[j+1] if j < n-1
	if j >= n-1 {
		return area[n-1]
	}
	seg := ct[j+1] - ct[j]
	if seg <= 0 {
		return area[j]
	}
	frac := (p - ct[j]) / seg
	val := f[j] + frac*(f[j+1]-f[j])
	return area[j] + (p-ct[j])*(f[j]+val)/2
}
