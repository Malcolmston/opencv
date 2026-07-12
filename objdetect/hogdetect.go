package objdetect

import (
	"fmt"
	"math"

	cv "github.com/malcolmston/opencv"
)

// ComputeGradient returns the per-pixel gradient magnitude and unsigned
// orientation of img (reduced to luma), the analogue of OpenCV's
// HOGDescriptor::computeGradient. The image is differenced with a centred
// [-1,0,1] kernel and edge replication; angle is folded into [0,180) degrees
// (unsigned gradients, as HOG uses). The mag and angle slices are row-major of
// length width*height and share the returned width/height.
func (h *HOGDescriptor) ComputeGradient(img *cv.Mat) (mag, angle []float64, width, height int) {
	g := matToGray(img)
	mag, angle = g.gradients()
	return mag, angle, g.w, g.h
}

// DetectMultiScaleWeights is [HOGDescriptor.DetectMultiScale] that also reports
// the linear-SVM score of every returned window. Unlike DetectMultiScale it does
// not group or suppress overlapping hits: it returns each raw above-threshold
// window together with its score, so callers can post-process with
// [NMSBoxes], [SoftNMSBoxes] or [GroupRectanglesWeights].
//
// The svmWeights, hitThreshold and scale arguments behave exactly as in
// [HOGDescriptor.DetectMultiScale]. The two returned slices are parallel and
// ordered by pyramid level (coarsest window first) then raster position. It
// panics on an invalid geometry, a wrong-length svmWeights, or scale <= 1.
func (h *HOGDescriptor) DetectMultiScaleWeights(img *cv.Mat, svmWeights []float64, hitThreshold, scale float64) ([]cv.Rect, []float64) {
	h.validate()
	if scale <= 1 {
		panic("objdetect: DetectMultiScaleWeights requires scale > 1")
	}
	descLen := h.DescriptorSize()
	var bias float64
	weights := svmWeights
	if len(weights) == descLen+1 {
		bias = weights[descLen]
		weights = weights[:descLen]
	}
	if len(weights) != descLen {
		panic(fmt.Sprintf("objdetect: DetectMultiScaleWeights svmWeights length %d, want %d or %d",
			len(svmWeights), descLen, descLen+1))
	}

	base := matToGray(img)
	var hits []cv.Rect
	var scores []float64
	s := 1.0
	for {
		lw := int(float64(base.w)/s + 0.5)
		lh := int(float64(base.h)/s + 0.5)
		if lw < h.WinSize.Width || lh < h.WinSize.Height {
			break
		}
		level := base
		if s != 1.0 {
			level = base.resize(lw, lh)
		}
		mag, ori := level.gradients()
		for y0 := 0; y0+h.WinSize.Height <= lh; y0 += h.BlockStride.Height {
			for x0 := 0; x0+h.WinSize.Width <= lw; x0 += h.BlockStride.Width {
				desc := h.window(mag, ori, lw, x0, y0)
				score := bias
				for i, w := range weights {
					score += w * desc[i]
				}
				if score >= hitThreshold {
					hits = append(hits, cv.Rect{
						X:      int(float64(x0)*s + 0.5),
						Y:      int(float64(y0)*s + 0.5),
						Width:  int(float64(h.WinSize.Width)*s + 0.5),
						Height: int(float64(h.WinSize.Height)*s + 0.5),
					})
					scores = append(scores, score)
				}
			}
		}
		s *= scale
	}
	return hits, scores
}

// DefaultPeopleDetector returns an approximate linear-SVM weight vector for
// upright-person detection, the standard-library analogue of OpenCV's
// HOGDescriptor::getDefaultPeopleDetector. OpenCV ships a weight vector trained
// on the INRIA person set; reproducing those exact 3781 coefficients without the
// training data is out of scope, so this method instead synthesises a usable
// matched-filter classifier from a prototype pedestrian silhouette.
//
// The prototype is a dark head-and-torso figure on a light ground rendered at
// the descriptor's WinSize; its HOG descriptor, mean-centred and unit-scaled,
// becomes the weight vector, and a small negative bias is appended so that a
// window scores positively only when its gradient structure genuinely resembles
// a standing person. The returned slice has length
// [HOGDescriptor.DescriptorSize]+1 (weights followed by bias) and is directly
// usable as the svmWeights argument to [HOGDescriptor.DetectMultiScale] and
// [HOGDescriptor.DetectMultiScaleWeights].
//
// It is an approximation: it discriminates person-like silhouettes from flat or
// randomly textured windows, but does not match OpenCV's detection accuracy.
func (h *HOGDescriptor) DefaultPeopleDetector() []float64 {
	h.validate()
	descLen := h.DescriptorSize()

	// Render a prototype upright figure at the detection window size.
	w, ht := h.WinSize.Width, h.WinSize.Height
	proto := &grayImage{w: w, h: ht, pix: make([]float64, w*ht)}
	const bg, fg = 210.0, 60.0
	for i := range proto.pix {
		proto.pix[i] = bg
	}
	cx := float64(w) / 2
	headR := float64(w) * 0.16
	headCy := float64(ht) * 0.14
	// Torso/legs: a vertical bar tapering slightly, centred horizontally.
	bodyTop := float64(ht) * 0.24
	bodyHalf := float64(w) * 0.26
	for y := 0; y < ht; y++ {
		fy := float64(y)
		for x := 0; x < w; x++ {
			fx := float64(x)
			inHead := math.Hypot(fx-cx, fy-headCy) <= headR
			inBody := fy >= bodyTop && math.Abs(fx-cx) <= bodyHalf
			if inHead || inBody {
				proto.pix[y*w+x] = fg
			}
		}
	}

	mag, ori := proto.gradients()
	desc := h.window(mag, ori, w, 0, 0)

	// Mean-centre so a flat (all-equal) window scores ~0, then scale to unit
	// L2 norm for a well-conditioned classifier.
	var mean float64
	for _, v := range desc {
		mean += v
	}
	mean /= float64(len(desc))
	var norm float64
	for i := range desc {
		desc[i] -= mean
		norm += desc[i] * desc[i]
	}
	norm = math.Sqrt(norm)
	if norm == 0 {
		norm = 1
	}
	out := make([]float64, descLen+1)
	for i := 0; i < descLen; i++ {
		out[i] = desc[i] / norm
	}
	// Bias: require the correlation to clear a modest positive margin.
	out[descLen] = -0.15
	return out
}
