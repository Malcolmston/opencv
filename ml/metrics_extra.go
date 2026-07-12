package ml

import (
	"math"
	"sort"
)

// binaryCounts tallies the true positives, false positives and false negatives
// for the class positive under a one-vs-rest view.
func binaryCounts(predicted, actual []int, positive int) (tp, fp, fn int) {
	if len(predicted) != len(actual) {
		panic("ml: predicted and actual must have equal length")
	}
	for i := range actual {
		switch {
		case predicted[i] == positive && actual[i] == positive:
			tp++
		case predicted[i] == positive && actual[i] != positive:
			fp++
		case predicted[i] != positive && actual[i] == positive:
			fn++
		}
	}
	return tp, fp, fn
}

// Precision returns the precision TP/(TP+FP) for the class positive, treating
// every other label as negative. With no positive predictions it returns 0. It
// panics if the two slices differ in length.
func Precision(predicted, actual []int, positive int) float64 {
	tp, fp, _ := binaryCounts(predicted, actual, positive)
	if tp+fp == 0 {
		return 0
	}
	return float64(tp) / float64(tp+fp)
}

// Recall returns the recall TP/(TP+FN) for the class positive, treating every
// other label as negative. With no actual positives it returns 0. It panics if
// the two slices differ in length.
func Recall(predicted, actual []int, positive int) float64 {
	tp, _, fn := binaryCounts(predicted, actual, positive)
	if tp+fn == 0 {
		return 0
	}
	return float64(tp) / float64(tp+fn)
}

// F1Score returns the harmonic mean of [Precision] and [Recall] for the class
// positive. It is 0 when both are 0. It panics if the two slices differ in
// length.
func F1Score(predicted, actual []int, positive int) float64 {
	p := Precision(predicted, actual, positive)
	r := Recall(predicted, actual, positive)
	if p+r == 0 {
		return 0
	}
	return 2 * p * r / (p + r)
}

// distinctLabels returns the sorted set of labels appearing in either slice.
func distinctLabels(predicted, actual []int) []int {
	set := make(map[int]struct{})
	for _, l := range actual {
		set[l] = struct{}{}
	}
	for _, l := range predicted {
		set[l] = struct{}{}
	}
	out := make([]int, 0, len(set))
	for l := range set {
		out = append(out, l)
	}
	sort.Ints(out)
	return out
}

// MacroPrecision returns the unweighted mean of the per-class [Precision] over
// every label present in the data. It panics if the two slices differ in length.
func MacroPrecision(predicted, actual []int) float64 {
	labels := distinctLabels(predicted, actual)
	if len(labels) == 0 {
		return 0
	}
	var sum float64
	for _, l := range labels {
		sum += Precision(predicted, actual, l)
	}
	return sum / float64(len(labels))
}

// MacroRecall returns the unweighted mean of the per-class [Recall] over every
// label present in the data. It panics if the two slices differ in length.
func MacroRecall(predicted, actual []int) float64 {
	labels := distinctLabels(predicted, actual)
	if len(labels) == 0 {
		return 0
	}
	var sum float64
	for _, l := range labels {
		sum += Recall(predicted, actual, l)
	}
	return sum / float64(len(labels))
}

// MacroF1 returns the unweighted mean of the per-class [F1Score] over every
// label present in the data. It panics if the two slices differ in length.
func MacroF1(predicted, actual []int) float64 {
	labels := distinctLabels(predicted, actual)
	if len(labels) == 0 {
		return 0
	}
	var sum float64
	for _, l := range labels {
		sum += F1Score(predicted, actual, l)
	}
	return sum / float64(len(labels))
}

// ROCCurve computes the receiver-operating-characteristic curve for a binary
// scorer. scores[i] is the model's confidence that sample i is the positive
// class, and actual[i] is its true label; positive names the positive class.
// It returns the false-positive and true-positive rates at every distinct score
// threshold, ordered from the strictest threshold (0,0) to the loosest (1,1),
// suitable for plotting or passing to [AUCFromCurve]. It panics if the slices
// differ in length.
func ROCCurve(scores []float64, actual []int, positive int) (fpr, tpr []float64) {
	if len(scores) != len(actual) {
		panic("ml: ROCCurve requires scores and actual to have equal length")
	}
	type pair struct {
		score float64
		pos   bool
	}
	pairs := make([]pair, len(scores))
	var totalPos, totalNeg int
	for i := range scores {
		p := actual[i] == positive
		pairs[i] = pair{scores[i], p}
		if p {
			totalPos++
		} else {
			totalNeg++
		}
	}
	sort.SliceStable(pairs, func(a, b int) bool { return pairs[a].score > pairs[b].score })

	fpr = append(fpr, 0)
	tpr = append(tpr, 0)
	var tp, fp int
	i := 0
	for i < len(pairs) {
		thr := pairs[i].score
		for i < len(pairs) && pairs[i].score == thr {
			if pairs[i].pos {
				tp++
			} else {
				fp++
			}
			i++
		}
		if totalPos > 0 {
			tpr = append(tpr, float64(tp)/float64(totalPos))
		} else {
			tpr = append(tpr, 0)
		}
		if totalNeg > 0 {
			fpr = append(fpr, float64(fp)/float64(totalNeg))
		} else {
			fpr = append(fpr, 0)
		}
	}
	return fpr, tpr
}

// AUCFromCurve returns the area under an ROC curve given as parallel fpr/tpr
// slices (as produced by [ROCCurve]), integrated with the trapezoidal rule.
func AUCFromCurve(fpr, tpr []float64) float64 {
	if len(fpr) != len(tpr) {
		panic("ml: AUCFromCurve requires fpr and tpr to have equal length")
	}
	var area float64
	for i := 1; i < len(fpr); i++ {
		area += (fpr[i] - fpr[i-1]) * (tpr[i] + tpr[i-1]) / 2
	}
	return area
}

// AUC returns the area under the ROC curve for a binary scorer, computed
// directly from the Mann-Whitney U statistic (the probability that a random
// positive sample scores above a random negative one, with ties counted as a
// half). It equals the trapezoidal area of [ROCCurve] but avoids materialising
// the curve. It panics if the slices differ in length.
func AUC(scores []float64, actual []int, positive int) float64 {
	if len(scores) != len(actual) {
		panic("ml: AUC requires scores and actual to have equal length")
	}
	type pair struct {
		score float64
		pos   bool
	}
	pairs := make([]pair, len(scores))
	var nPos, nNeg int
	for i := range scores {
		p := actual[i] == positive
		pairs[i] = pair{scores[i], p}
		if p {
			nPos++
		} else {
			nNeg++
		}
	}
	if nPos == 0 || nNeg == 0 {
		return 0
	}
	sort.SliceStable(pairs, func(a, b int) bool { return pairs[a].score < pairs[b].score })
	// Assign average ranks to break score ties.
	ranks := make([]float64, len(pairs))
	i := 0
	for i < len(pairs) {
		j := i
		for j < len(pairs) && pairs[j].score == pairs[i].score {
			j++
		}
		avg := float64(i+j-1)/2 + 1 // 1-based average rank
		for k := i; k < j; k++ {
			ranks[k] = avg
		}
		i = j
	}
	var sumRankPos float64
	for k := range pairs {
		if pairs[k].pos {
			sumRankPos += ranks[k]
		}
	}
	u := sumRankPos - float64(nPos)*float64(nPos+1)/2
	return u / (float64(nPos) * float64(nNeg))
}

// MSE returns the mean squared error between predicted and actual regression
// targets. It panics if the slices differ in length; an empty input yields 0.
func MSE(predicted, actual []float64) float64 {
	if len(predicted) != len(actual) {
		panic("ml: MSE requires predicted and actual to have equal length")
	}
	if len(actual) == 0 {
		return 0
	}
	var sum float64
	for i := range actual {
		d := predicted[i] - actual[i]
		sum += d * d
	}
	return sum / float64(len(actual))
}

// RMSE returns the square root of [MSE].
func RMSE(predicted, actual []float64) float64 {
	return math.Sqrt(MSE(predicted, actual))
}

// MAE returns the mean absolute error between predicted and actual regression
// targets. It panics if the slices differ in length; an empty input yields 0.
func MAE(predicted, actual []float64) float64 {
	if len(predicted) != len(actual) {
		panic("ml: MAE requires predicted and actual to have equal length")
	}
	if len(actual) == 0 {
		return 0
	}
	var sum float64
	for i := range actual {
		sum += math.Abs(predicted[i] - actual[i])
	}
	return sum / float64(len(actual))
}

// R2Score returns the coefficient of determination R², the fraction of variance
// in actual explained by predicted. A perfect fit is 1; predicting the mean
// gives 0; worse-than-mean fits are negative. It panics if the slices differ in
// length or are empty.
func R2Score(predicted, actual []float64) float64 {
	if len(predicted) != len(actual) {
		panic("ml: R2Score requires predicted and actual to have equal length")
	}
	if len(actual) == 0 {
		panic("ml: R2Score requires at least one sample")
	}
	var mean float64
	for _, v := range actual {
		mean += v
	}
	mean /= float64(len(actual))
	var ssRes, ssTot float64
	for i := range actual {
		dr := actual[i] - predicted[i]
		ssRes += dr * dr
		dt := actual[i] - mean
		ssTot += dt * dt
	}
	if ssTot == 0 {
		return 0
	}
	return 1 - ssRes/ssTot
}
