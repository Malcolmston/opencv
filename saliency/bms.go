package saliency

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// StaticSaliencyBooleanMap implements Boolean Map based Saliency (BMS) after
// Zhang & Sclaroff, "Saliency Detection: A Boolean Map Approach" (ICCV 2013).
//
// The method rests on the Gestalt principle of surroundedness: figures tend to
// be enclosed regions. Each colour/opponency channel is thresholded at a range
// of levels to produce a stack of Boolean maps. For every Boolean map an
// attention map is formed by activating the connected regions that do NOT touch
// the image border (they are surrounded by the complementary value); this is
// done for both the map and its inverse. Each attention map is normalised by
// its own magnitude so that many small surrounded blobs cannot outvote one
// large one, and the maps are averaged and blurred into the final saliency map.
//
// A distinct object sitting away from the border is enclosed at most threshold
// levels, so it is repeatedly activated and ends up bright.
//
// Construct one with [NewStaticSaliencyBooleanMap]. It satisfies
// [StaticSaliency].
type StaticSaliencyBooleanMap struct {
	// Thresholds is the number of evenly spaced threshold levels swept across
	// each channel's [0,255] range. The default is 8.
	Thresholds int
	// BlurRadius smooths the final averaged attention map (0 disables it). The
	// default is 3.
	BlurRadius int
}

// NewStaticSaliencyBooleanMap returns a detector with eight threshold levels.
func NewStaticSaliencyBooleanMap() *StaticSaliencyBooleanMap {
	return &StaticSaliencyBooleanMap{Thresholds: 8, BlurRadius: 3}
}

// activateSurrounded returns a plane whose samples are 1 where the pixel equals
// target and its connected component does not touch the border, else 0.
func activateSurrounded(mask []bool, rows, cols int, target bool) *plane {
	touch := floodBorderMask(mask, rows, cols, target)
	out := newPlane(rows, cols)
	for i := range mask {
		if mask[i] == target && !touch[i] {
			out.data[i] = 1
		}
	}
	return out
}

// l2Normalize divides the plane by its Euclidean norm (leaves an all-zero plane
// untouched).
func l2Normalize(p *plane) {
	var ss float64
	for _, v := range p.data {
		ss += v * v
	}
	norm := math.Sqrt(ss)
	if norm <= 0 {
		return
	}
	for i := range p.data {
		p.data[i] /= norm
	}
}

// ComputeSaliency returns the Boolean-map saliency of img: a single-channel
// [cv.Mat] the same size as img, normalised to [0,255]. It panics if img is nil
// or empty.
func (s *StaticSaliencyBooleanMap) ComputeSaliency(img *cv.Mat) *cv.Mat {
	r, g, b := rgbPlanes(img)
	rows, cols := r.rows, r.cols

	// Use intensity plus two colour-opponency channels as feature maps.
	intensity := newPlane(rows, cols)
	rg := newPlane(rows, cols)
	by := newPlane(rows, cols)
	for i := range intensity.data {
		rr, gg, bb := r.data[i], g.data[i], b.data[i]
		intensity.data[i] = (rr + gg + bb) / 3
		rg.data[i] = 127.5 + (rr-gg)/2
		by.data[i] = 127.5 + (bb-(rr+gg)/2)/2
	}
	channels := []*plane{intensity, rg, by}

	levels := s.Thresholds
	if levels < 1 {
		levels = 8
	}

	acc := newPlane(rows, cols)
	mask := make([]bool, rows*cols)
	for _, ch := range channels {
		for t := 1; t <= levels; t++ {
			thr := 255 * float64(t) / float64(levels+1)
			for i, v := range ch.data {
				mask[i] = v > thr
			}
			for _, target := range []bool{true, false} {
				att := activateSurrounded(mask, rows, cols, target)
				l2Normalize(att)
				acc.addScaled(att, 1)
			}
		}
	}

	if s.BlurRadius > 0 {
		acc = meanBlur(acc, s.BlurRadius)
	}
	return acc.normalizedMat()
}
