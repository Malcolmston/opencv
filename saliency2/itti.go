package saliency2

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// StaticSaliencyIttiKoch implements a center-surround attention model in the
// spirit of Itti, Koch & Niebur, "A Model of Saliency-Based Visual Attention
// for Rapid Scene Analysis" (PAMI 1998).
//
// Three biologically motivated feature families are extracted: an intensity
// channel, two colour-opponency channels (red-green and blue-yellow) and four
// orientation channels (0, 45, 90 and 135 degrees, from oriented gradients).
// Each feature is built into a Gaussian pyramid and its saliency comes from
// across-scale center-surround differences — a fine "centre" level minus a
// coarse "surround" level — which respond wherever a location differs from its
// neighbourhood. The per-feature maps are combined with the [IttiNormalize]
// operator N(.) into intensity, colour and orientation conspicuity maps, whose
// normalised average is the final saliency map. This is the heaviest detector
// in the package.
//
// Construct one with [NewStaticSaliencyIttiKoch]; the zero value is not usable.
type StaticSaliencyIttiKoch struct {
	// MaxLevels caps the depth of the Gaussian pyramids.
	MaxLevels int
	// CenterLevels and SurroundDeltas select the pyramid levels differenced:
	// for each centre level c and delta d, level c is compared against level
	// c+d. Missing levels are skipped.
	CenterLevels, SurroundDeltas []int
}

// NewStaticSaliencyIttiKoch returns a detector with the classical pyramid
// configuration (up to 9 levels, centres {2,3}, surround deltas {2,3}).
func NewStaticSaliencyIttiKoch() *StaticSaliencyIttiKoch {
	return &StaticSaliencyIttiKoch{
		MaxLevels:      9,
		CenterLevels:   []int{2, 3},
		SurroundDeltas: []int{2, 3},
	}
}

// saliency2CSAcross accumulates across-scale center-surround differences of a
// feature pyramid onto a reference-sized map. Each contribution is the absolute
// difference between a centre level and an up-sampled surround level, passed
// through the N(.) operator and resized to (refRows, refCols).
func (s *StaticSaliencyIttiKoch) saliency2CSAcross(pyr []*SaliencyMap, refRows, refCols int) *SaliencyMap {
	acc := NewSaliencyMap(refRows, refCols)
	any := false
	for _, c := range s.CenterLevels {
		for _, d := range s.SurroundDeltas {
			surIdx := c + d
			if c < 0 || c >= len(pyr) || surIdx >= len(pyr) {
				continue
			}
			center := pyr[c]
			surround := saliency2ResizeMap(pyr[surIdx], center.Rows, center.Cols)
			diff := NewSaliencyMap(center.Rows, center.Cols)
			for i := range diff.Data {
				v := center.Data[i] - surround.Data[i]
				if v < 0 {
					v = -v
				}
				diff.Data[i] = v
			}
			norm := IttiNormalize(diff)
			resized := saliency2ResizeMap(norm, refRows, refCols)
			for i := range acc.Data {
				acc.Data[i] += resized.Data[i]
			}
			any = true
		}
	}
	if !any {
		// Fall back to a single-scale center-surround so the model still
		// produces a meaningful map on very small images.
		base := pyr[0]
		blur := saliency2BoxBlurMap(base, 3)
		diff := NewSaliencyMap(base.Rows, base.Cols)
		for i := range diff.Data {
			v := base.Data[i] - blur.Data[i]
			if v < 0 {
				v = -v
			}
			diff.Data[i] = v
		}
		return saliency2ResizeMap(IttiNormalize(diff), refRows, refCols)
	}
	return acc
}

// ComputeSaliencyMap returns the Itti-Koch saliency of img as a [SaliencyMap]
// the same size as img. It panics if img is nil or empty.
func (s *StaticSaliencyIttiKoch) ComputeSaliencyMap(img *cv.Mat) *SaliencyMap {
	r, g, b := saliency2RGBFloat(img)
	rows, cols := r.Rows, r.Cols

	// Intensity channel.
	intensity := NewSaliencyMap(rows, cols)
	for i := range intensity.Data {
		intensity.Data[i] = (r.Data[i] + g.Data[i] + b.Data[i]) / 3
	}

	// Colour-opponency channels (Itti's broadly-tuned R,G,B,Y then RG, BY).
	rg := NewSaliencyMap(rows, cols)
	by := NewSaliencyMap(rows, cols)
	for i := range rg.Data {
		rr, gg, bb := r.Data[i], g.Data[i], b.Data[i]
		maxc := math.Max(rr, math.Max(gg, bb))
		if maxc < 1e-6 {
			continue
		}
		// Normalise by intensity to decouple hue from brightness.
		R := rr - (gg+bb)/2
		G := gg - (rr+bb)/2
		B := bb - (rr+gg)/2
		Y := (rr+gg)/2 - math.Abs(rr-gg)/2 - bb
		rg.Data[i] = (R - G) / maxc
		by.Data[i] = (B - Y) / maxc
	}

	maxLevels := s.MaxLevels
	if maxLevels < 2 {
		maxLevels = 9
	}
	const minSize = 4
	refLevel := 2

	intPyr := saliency2GaussPyramid(intensity, maxLevels, minSize)
	rgPyr := saliency2GaussPyramid(rg, maxLevels, minSize)
	byPyr := saliency2GaussPyramid(by, maxLevels, minSize)

	if refLevel >= len(intPyr) {
		refLevel = len(intPyr) - 1
	}
	refRows := intPyr[refLevel].Rows
	refCols := intPyr[refLevel].Cols

	// Intensity conspicuity.
	consI := s.saliency2CSAcross(intPyr, refRows, refCols)

	// Colour conspicuity: sum of the two opponency contributions.
	consC := s.saliency2CSAcross(rgPyr, refRows, refCols).AddMap(
		s.saliency2CSAcross(byPyr, refRows, refCols))

	// Orientation conspicuity across four angles.
	consO := NewSaliencyMap(refRows, refCols)
	gx, gy := saliency2Sobel(intensity)
	for _, theta := range []float64{0, math.Pi / 4, math.Pi / 2, 3 * math.Pi / 4} {
		ct := math.Cos(theta)
		st := math.Sin(theta)
		orient := NewSaliencyMap(rows, cols)
		for i := range orient.Data {
			v := gx.Data[i]*ct + gy.Data[i]*st
			if v < 0 {
				v = -v
			}
			orient.Data[i] = v
		}
		oPyr := saliency2GaussPyramid(orient, maxLevels, minSize)
		contrib := s.saliency2CSAcross(oPyr, refRows, refCols)
		consO = consO.AddMap(IttiNormalize(contrib))
	}

	// Combine the three conspicuity maps with N(.) and average.
	combined := NewSaliencyMap(refRows, refCols)
	for _, cm := range []*SaliencyMap{IttiNormalize(consI), IttiNormalize(consC), IttiNormalize(consO)} {
		combined = combined.AddMap(cm)
	}
	combined = combined.Scale(1.0 / 3.0)

	return saliency2ResizeMap(combined, rows, cols)
}

// ComputeSaliency returns the Itti-Koch saliency map of img as an 8-bit
// single-channel [cv.Mat]. It satisfies the [StaticSaliency] interface.
func (s *StaticSaliencyIttiKoch) ComputeSaliency(img *cv.Mat) *cv.Mat {
	return s.ComputeSaliencyMap(img).ToMat()
}

// IttiKochSaliency is a convenience wrapper that computes the Itti-Koch
// saliency map of img with the default detector settings.
func IttiKochSaliency(img *cv.Mat) *cv.Mat {
	return NewStaticSaliencyIttiKoch().ComputeSaliency(img)
}

// CenterSurround returns the center-surround response of a single feature map:
// the absolute difference between center and an up-sampled copy of surround,
// passed through the [IttiNormalize] operator. It is the elementary operation
// the Itti-Koch model repeats across scales, exposed for reuse. The surround
// map is resized to the center map's dimensions.
func CenterSurround(center, surround *SaliencyMap) *SaliencyMap {
	su := saliency2ResizeMap(surround, center.Rows, center.Cols)
	diff := NewSaliencyMap(center.Rows, center.Cols)
	for i := range diff.Data {
		v := center.Data[i] - su.Data[i]
		if v < 0 {
			v = -v
		}
		diff.Data[i] = v
	}
	return IttiNormalize(diff)
}
