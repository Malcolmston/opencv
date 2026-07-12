package saliency

import (
	"math"
	"sort"

	cv "github.com/malcolmston/opencv"
)

// This file provides the standard saliency-benchmark evaluation metrics used to
// compare a predicted saliency map against a ground-truth human-fixation record
// or another map, as collected in Bylinskii et al., "What do different
// evaluation metrics tell us about saliency models?" (IEEE TPAMI 2019).

// requireSameShape panics unless both maps are non-nil, single-channel and the
// same size.
func requireSameShape(a, b *cv.Mat, who string) {
	if a == nil || a.Empty() || b == nil || b.Empty() {
		panic("saliency: " + who + " given an empty map")
	}
	if a.Channels != 1 || b.Channels != 1 {
		panic("saliency: " + who + " requires single-channel maps")
	}
	if a.Rows != b.Rows || a.Cols != b.Cols {
		panic("saliency: " + who + " map sizes differ")
	}
}

// normFloats returns the saliency map's samples as float64 scaled to [0,1].
func normFloats(m *cv.Mat) []float64 {
	out := make([]float64, len(m.Data))
	for i, v := range m.Data {
		out[i] = float64(v) / 255
	}
	return out
}

// AUCJudd computes the AUC-Judd area-under-ROC score of a saliency map against a
// binary fixation map (any non-zero sample marks a fixated pixel). The saliency
// map is treated as a binary classifier of fixated vs. non-fixated pixels swept
// over every fixation-valued threshold; the returned value is the area under
// the resulting ROC curve (0.5 is chance, 1.0 is perfect). It panics if the
// maps differ in size or are not single-channel, and returns NaN if there are no
// fixations. Both maps must be the same size.
func AUCJudd(salMap, fixation *cv.Mat) float64 {
	requireSameShape(salMap, fixation, "AUCJudd")
	sal := normFloats(salMap)
	nPix := len(sal)

	var fixSal []float64
	for i, v := range fixation.Data {
		if v != 0 {
			fixSal = append(fixSal, sal[i])
		}
	}
	nFix := len(fixSal)
	if nFix == 0 {
		return math.NaN()
	}
	nOther := nPix - nFix
	if nOther <= 0 {
		return math.NaN()
	}
	// Thresholds are the fixation saliency values in descending order.
	sort.Sort(sort.Reverse(sort.Float64Slice(fixSal)))
	sortedAll := make([]float64, nPix)
	copy(sortedAll, sal)
	sort.Float64Slice(sortedAll).Sort() // ascending

	tp := make([]float64, nFix+2)
	fp := make([]float64, nFix+2)
	tp[nFix+1] = 1
	fp[nFix+1] = 1
	for i, thr := range fixSal {
		// pixels with saliency >= thr, via binary search on ascending array.
		aboveEqual := nPix - sort.Search(nPix, func(k int) bool { return sortedAll[k] >= thr })
		fixAbove := i + 1 // fixSal is sorted desc, so i+1 fixations are >= thr
		tp[i+1] = float64(fixAbove) / float64(nFix)
		fp[i+1] = float64(aboveEqual-fixAbove) / float64(nOther)
	}
	// Trapezoidal integration of the ROC (fp ascending, tp ascending).
	var auc float64
	for i := 1; i < len(tp); i++ {
		auc += (fp[i] - fp[i-1]) * (tp[i] + tp[i-1]) / 2
	}
	return auc
}

// NSS computes the Normalized Scanpath Saliency: the saliency map is normalised
// to zero mean and unit standard deviation, and NSS is the average of the
// normalised saliency at the fixated pixels (any non-zero sample of fixation).
// Positive values mean the map predicts fixations above chance. It panics if the
// maps differ in size or are not single-channel, and returns NaN when the map is
// flat or there are no fixations. Both maps must be the same size.
func NSS(salMap, fixation *cv.Mat) float64 {
	requireSameShape(salMap, fixation, "NSS")
	sal := normFloats(salMap)
	n := float64(len(sal))
	var mean float64
	for _, v := range sal {
		mean += v
	}
	mean /= n
	var varSum float64
	for _, v := range sal {
		d := v - mean
		varSum += d * d
	}
	std := math.Sqrt(varSum / n)
	if std == 0 {
		return math.NaN()
	}
	var sum float64
	var nFix int
	for i, v := range fixation.Data {
		if v != 0 {
			sum += (sal[i] - mean) / std
			nFix++
		}
	}
	if nFix == 0 {
		return math.NaN()
	}
	return sum / float64(nFix)
}

// CC computes the linear (Pearson) correlation coefficient between two saliency
// maps, in [-1,1]. It is symmetric and invariant to affine rescaling of either
// map. It panics if the maps differ in size or are not single-channel, and
// returns NaN if either map is flat. Both maps must be the same size.
func CC(a, b *cv.Mat) float64 {
	requireSameShape(a, b, "CC")
	fa := normFloats(a)
	fb := normFloats(b)
	n := float64(len(fa))
	var ma, mb float64
	for i := range fa {
		ma += fa[i]
		mb += fb[i]
	}
	ma /= n
	mb /= n
	var cov, va, vb float64
	for i := range fa {
		da := fa[i] - ma
		db := fb[i] - mb
		cov += da * db
		va += da * da
		vb += db * db
	}
	if va == 0 || vb == 0 {
		return math.NaN()
	}
	return cov / math.Sqrt(va*vb)
}

// SIM computes the similarity (histogram-intersection) metric between two
// saliency maps: each map is normalised to sum to one and SIM is the sum of the
// per-pixel minima, in [0,1] (1 means identical distributions). It panics if the
// maps differ in size or are not single-channel, and returns 0 if either map is
// all zero. Both maps must be the same size.
func SIM(a, b *cv.Mat) float64 {
	requireSameShape(a, b, "SIM")
	pa := probFloats(a)
	pb := probFloats(b)
	if pa == nil || pb == nil {
		return 0
	}
	var sim float64
	for i := range pa {
		sim += math.Min(pa[i], pb[i])
	}
	return sim
}

// KLDiv computes the Kullback-Leibler divergence of a predicted saliency map
// from a ground-truth fixation-density map, KL(groundTruth ‖ prediction). Both
// are normalised to probability distributions; lower is better (0 means the
// prediction matches the ground truth). It panics if the maps differ in size or
// are not single-channel, and returns NaN if the ground truth is all zero. Both
// maps must be the same size.
func KLDiv(prediction, groundTruth *cv.Mat) float64 {
	requireSameShape(prediction, groundTruth, "KLDiv")
	q := probFloats(prediction)
	p := probFloats(groundTruth)
	if p == nil {
		return math.NaN()
	}
	if q == nil {
		return math.Inf(1)
	}
	const eps = 1e-12
	var kl float64
	for i := range p {
		if p[i] <= 0 {
			continue
		}
		kl += p[i] * math.Log(p[i]/(q[i]+eps)+eps)
	}
	return kl
}

// probFloats returns the map's samples normalised to sum to one, or nil if the
// map sums to zero.
func probFloats(m *cv.Mat) []float64 {
	out := make([]float64, len(m.Data))
	var sum float64
	for i, v := range m.Data {
		out[i] = float64(v)
		sum += out[i]
	}
	if sum <= 0 {
		return nil
	}
	for i := range out {
		out[i] /= sum
	}
	return out
}
