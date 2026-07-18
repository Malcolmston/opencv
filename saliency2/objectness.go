package saliency2

import (
	"image"
	"math"
	"sort"

	cv "github.com/malcolmston/opencv"
)

// NormedGradient returns the normed-gradient magnitude map of img: the Sobel
// gradient magnitude of its luminance. Object boundaries carry strong, closed
// gradient contours, which is the cue the [ObjectnessBING] proposal generator
// scores.
func NormedGradient(img *cv.Mat) *SaliencyMap {
	gray := saliency2GrayFloat(img)
	gx, gy := saliency2Sobel(gray)
	out := NewSaliencyMap(gray.Rows, gray.Cols)
	for i := range out.Data {
		out.Data[i] = math.Hypot(gx.Data[i], gy.Data[i])
	}
	return out
}

// ObjectnessBING is a lightweight, untrained object-proposal generator in the
// spirit of Cheng, Zhang, Lin & Torr's BING, "Binarized Normed Gradients for
// Objectness Estimation at 300fps" (CVPR 2014).
//
// Full BING resizes each candidate window to 8x8 normed gradients and scores it
// with a learned linear SVM. This dependency-free variant keeps BING's core
// insight — objects are bounded by closed rings of strong gradient — but
// replaces the trained filter with a boundary-energy score: a window is rated
// by the mean normed-gradient sampled along its rectangular border, so windows
// whose edges snap onto object contours rank highest. Candidate windows are
// generated at several sizes and aspect ratios, scored, and reduced to a ranked,
// non-overlapping shortlist with greedy non-maximum suppression.
//
// Construct one with [NewObjectnessBING]; the zero value is not usable.
type ObjectnessBING struct {
	// SizeFractions lists window side lengths as fractions of the smaller image
	// dimension. Each pair of fractions also forms non-square windows.
	SizeFractions []float64
	// MaxBoxes bounds the number of proposals returned.
	MaxBoxes int
	// NMSThreshold is the intersection-over-union above which a lower-scoring
	// window is suppressed by a higher-scoring one.
	NMSThreshold float64
}

// NewObjectnessBING returns a generator with sensible defaults: window sizes at
// 1/4, 1/2 and 3/4 of the smaller image side, up to 32 proposals, and an IoU
// suppression threshold of 0.5.
func NewObjectnessBING() *ObjectnessBING {
	return &ObjectnessBING{
		SizeFractions: []float64{0.25, 0.5, 0.75},
		MaxBoxes:      32,
		NMSThreshold:  0.5,
	}
}

// ObjectnessMap returns the normed-gradient boundary-energy field used to score
// windows: for every pixel it holds the mean gradient along the border of a
// mid-sized window anchored there. Bright locations are good top-left corners
// for object windows. It is exposed for visualisation and reuse.
func (o *ObjectnessBING) ObjectnessMap(img *cv.Mat) *SaliencyMap {
	ng := NormedGradient(img)
	rows, cols := ng.Rows, ng.Cols
	side := rows
	if cols < side {
		side = cols
	}
	win := side / 2
	if win < 2 {
		win = 2
	}
	out := NewSaliencyMap(rows, cols)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			out.Data[y*cols+x] = saliency2PerimeterMean(ng, y, x, win, win)
		}
	}
	return out
}

// ObjectnessBoundingBoxes returns candidate object windows for img ranked by
// decreasing objectness score, after non-maximum suppression. It satisfies the
// [Objectness] interface. It panics if img is nil or empty.
func (o *ObjectnessBING) ObjectnessBoundingBoxes(img *cv.Mat) []Box {
	ng := NormedGradient(img)
	rows, cols := ng.Rows, ng.Cols
	side := rows
	if cols < side {
		side = cols
	}

	fracs := o.SizeFractions
	if len(fracs) == 0 {
		fracs = []float64{0.25, 0.5, 0.75}
	}
	sizes := make([]int, 0, len(fracs))
	for _, f := range fracs {
		s := int(math.Round(f * float64(side)))
		if s >= 2 && s <= side {
			sizes = append(sizes, s)
		}
	}
	if len(sizes) == 0 {
		sizes = []int{saliency2ClampInt(side/2, 2, side)}
	}

	var boxes []Box
	seen := map[[4]int]bool{}
	for _, wh := range sizes {
		for _, ww := range sizes {
			step := ww / 4
			if h4 := wh / 4; h4 < step || step < 1 {
				step = h4
			}
			if step < 1 {
				step = 1
			}
			for y := 0; y+wh <= rows; y += step {
				for x := 0; x+ww <= cols; x += step {
					key := [4]int{y, x, wh, ww}
					if seen[key] {
						continue
					}
					seen[key] = true
					score := saliency2PerimeterMean(ng, y, x, wh, ww)
					boxes = append(boxes, Box{
						Rect:  image.Rect(x, y, x+ww, y+wh),
						Score: score,
					})
				}
			}
		}
	}

	sort.SliceStable(boxes, func(i, j int) bool {
		return boxes[i].Score > boxes[j].Score
	})

	nms := o.NMSThreshold
	if nms <= 0 {
		nms = 0.5
	}
	kept := make([]Box, 0, len(boxes))
	for _, cand := range boxes {
		overlap := false
		for _, k := range kept {
			if cand.IoU(k) > nms {
				overlap = true
				break
			}
		}
		if !overlap {
			kept = append(kept, cand)
		}
		if o.MaxBoxes > 0 && len(kept) >= o.MaxBoxes {
			break
		}
	}
	return kept
}

// ComputeSaliency returns the objectness map of img as an 8-bit single-channel
// [cv.Mat], with bright pixels marking likely object-window anchors. It lets
// [ObjectnessBING] double as a static saliency source.
func (o *ObjectnessBING) ComputeSaliency(img *cv.Mat) *cv.Mat {
	return o.ObjectnessMap(img).ToMat()
}
