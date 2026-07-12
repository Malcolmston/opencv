package saliency

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// StaticSaliencyIttiKochNiebur implements the classical bottom-up visual
// attention model of Itti, Koch & Niebur, "A Model of Saliency-Based Visual
// Attention for Rapid Scene Analysis" (IEEE TPAMI 1998) — the reference
// biological saliency architecture that predates and inspired OpenCV's static
// detectors.
//
// The image is decomposed into three early-vision feature families:
//
//   - intensity (I = (R+G+B)/3);
//   - colour double-opponency (red-green and blue-yellow); and
//   - local orientation energy (four oriented edge kernels).
//
// Each feature is built into a dyadic Gaussian pyramid, and center-surround
// contrast is measured as the across-scale absolute difference between a fine
// "center" level and a coarser "surround" level (each surround is upsampled to
// the center's size before subtraction). The maps are combined with Itti's
// normalisation operator N(·) — normalise to a fixed range, then multiply by
// (1-mean)² so that a map with a few strong peaks is promoted over one with many
// comparable responses — accumulated into per-feature conspicuity maps, averaged
// and resized back to the input resolution.
//
// Orientation is approximated with four fixed oriented kernels rather than a
// full bank of Gabor filters, which keeps the detector dependency-free while
// preserving the qualitative center-surround behaviour.
//
// Construct one with [NewStaticSaliencyIttiKochNiebur]. It satisfies
// [StaticSaliency].
type StaticSaliencyIttiKochNiebur struct {
	// PyramidLevels is the number of dyadic pyramid levels built for every
	// feature. The default is 7.
	PyramidLevels int
	// CenterLevels are the fine pyramid levels used as center scales.
	CenterLevels []int
	// SurroundDeltas are the level offsets (added to each center) used as
	// surround scales.
	SurroundDeltas []int
}

// NewStaticSaliencyIttiKochNiebur returns a detector configured with the
// classical center scales {2,3}, surround deltas {2,3} and a seven-level
// pyramid.
func NewStaticSaliencyIttiKochNiebur() *StaticSaliencyIttiKochNiebur {
	return &StaticSaliencyIttiKochNiebur{
		PyramidLevels:  7,
		CenterLevels:   []int{2, 3},
		SurroundDeltas: []int{2, 3},
	}
}

// ittiNormalize applies Itti's map-promotion operator: the plane is normalised
// to [0,1] and multiplied by (1-mean)², so sparse strong peaks are amplified
// relative to cluttered maps.
func ittiNormalize(p *plane) *plane {
	q := clonePlane(p)
	q.normalizeUnit()
	m := q.mean()
	f := (1 - m) * (1 - m)
	for i := range q.data {
		q.data[i] *= f
	}
	return q
}

// conspicuity builds the center-surround conspicuity of a single feature plane
// at the given target size. It sums N(center-surround) over the configured
// center/surround scale pairs.
func (s *StaticSaliencyIttiKochNiebur) conspicuity(feature *plane, targetRows, targetCols int) *plane {
	pyr := gaussPyramid(feature, s.PyramidLevels)
	acc := newPlane(targetRows, targetCols)
	for _, c := range s.CenterLevels {
		if c >= len(pyr) {
			continue
		}
		for _, d := range s.SurroundDeltas {
			sl := c + d
			if sl >= len(pyr) {
				continue
			}
			fine := pyr[c]
			coarse := resizePlane(pyr[sl], fine.rows, fine.cols)
			cs := absDiffPlanes(fine, coarse)
			cs = resizePlane(cs, targetRows, targetCols)
			acc.addScaled(ittiNormalize(cs), 1)
		}
	}
	return acc
}

// orientationFeature returns combined oriented-edge energy of the intensity
// plane, summing the absolute responses of four oriented 3×3 kernels.
func orientationFeature(intensity *plane) *plane {
	kernels := [][9]float64{
		{-1, -1, -1, 2, 2, 2, -1, -1, -1}, // horizontal (0°)
		{-1, 2, -1, -1, 2, -1, -1, 2, -1}, // vertical (90°)
		{-1, -1, 2, -1, 2, -1, 2, -1, -1}, // 45°
		{2, -1, -1, -1, 2, -1, -1, -1, 2}, // 135°
	}
	out := newPlane(intensity.rows, intensity.cols)
	for _, k := range kernels {
		r := conv3x3(intensity, k)
		for i, v := range r.data {
			out.data[i] += math.Abs(v)
		}
	}
	return out
}

// ComputeSaliency returns the Itti-Koch-Niebur saliency map of img: a
// single-channel [cv.Mat] the same size as img, normalised to [0,255]. It
// panics if img is nil or empty.
func (s *StaticSaliencyIttiKochNiebur) ComputeSaliency(img *cv.Mat) *cv.Mat {
	levels := s.PyramidLevels
	if levels < 2 {
		levels = 7
	}
	centers := s.CenterLevels
	if len(centers) == 0 {
		centers = []int{2, 3}
	}
	deltas := s.SurroundDeltas
	if len(deltas) == 0 {
		deltas = []int{2, 3}
	}
	cfg := &StaticSaliencyIttiKochNiebur{PyramidLevels: levels, CenterLevels: centers, SurroundDeltas: deltas}

	r, g, b := rgbPlanes(img)
	rows, cols := r.rows, r.cols

	// Intensity.
	intensity := newPlane(rows, cols)
	for i := range intensity.data {
		intensity.data[i] = (r.data[i] + g.data[i] + b.data[i]) / 3
	}

	// Colour double-opponency channels.
	rg := newPlane(rows, cols)
	by := newPlane(rows, cols)
	for i := range rg.data {
		rr, gg, bb := r.data[i], g.data[i], b.data[i]
		cR := rr - (gg+bb)/2
		cG := gg - (rr+bb)/2
		cB := bb - (rr+gg)/2
		cY := (rr+gg)/2 - math.Abs(rr-gg)/2 - bb
		rg.data[i] = math.Abs(cR - cG)
		by.data[i] = math.Abs(cB - cY)
	}

	// Combine at the resolution of pyramid level 2 (coarse enough to be cheap,
	// fine enough to keep object shape) and resize once at the end.
	ip := gaussPyramid(intensity, levels)
	tl := 2
	if tl >= len(ip) {
		tl = len(ip) - 1
	}
	tr, tc := ip[tl].rows, ip[tl].cols

	intConsp := cfg.conspicuity(intensity, tr, tc)
	colConsp := cfg.conspicuity(rg, tr, tc)
	colConsp.addScaled(cfg.conspicuity(by, tr, tc), 1)
	oriConsp := cfg.conspicuity(orientationFeature(intensity), tr, tc)

	final := newPlane(tr, tc)
	final.addScaled(ittiNormalize(intConsp), 1.0/3.0)
	final.addScaled(ittiNormalize(colConsp), 1.0/3.0)
	final.addScaled(ittiNormalize(oriConsp), 1.0/3.0)

	full := resizePlane(final, rows, cols)
	full = gaussianBlurPlane(full, 5, 1.5)
	return full.normalizedMat()
}
