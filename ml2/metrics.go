package ml2

// Accuracy returns the fraction of predictions that match the true labels. It
// panics if the slices differ in length; an empty input yields zero.
func Accuracy(yTrue, yPred []int) float64 {
	if len(yTrue) != len(yPred) {
		panic("ml2: Accuracy requires equal-length inputs")
	}
	if len(yTrue) == 0 {
		return 0
	}
	correct := 0
	for i := range yTrue {
		if yTrue[i] == yPred[i] {
			correct++
		}
	}
	return float64(correct) / float64(len(yTrue))
}

// ConfusionMatrix returns a numClasses-by-numClasses matrix whose entry [t][p]
// counts samples with true class t predicted as class p. It panics if the
// slices differ in length.
func ConfusionMatrix(yTrue, yPred []int, numClasses int) [][]int {
	if len(yTrue) != len(yPred) {
		panic("ml2: ConfusionMatrix requires equal-length inputs")
	}
	m := make([][]int, numClasses)
	for i := range m {
		m[i] = make([]int, numClasses)
	}
	for i := range yTrue {
		m[yTrue[i]][yPred[i]]++
	}
	return m
}

// Precision returns the precision for a single class: true positives divided by
// all samples predicted to be that class. It returns zero when the class is
// never predicted.
func Precision(yTrue, yPred []int, class int) float64 {
	tp, fp := 0, 0
	for i := range yTrue {
		if yPred[i] == class {
			if yTrue[i] == class {
				tp++
			} else {
				fp++
			}
		}
	}
	if tp+fp == 0 {
		return 0
	}
	return float64(tp) / float64(tp+fp)
}

// Recall returns the recall for a single class: true positives divided by all
// samples that truly belong to that class. It returns zero when the class never
// occurs.
func Recall(yTrue, yPred []int, class int) float64 {
	tp, fn := 0, 0
	for i := range yTrue {
		if yTrue[i] == class {
			if yPred[i] == class {
				tp++
			} else {
				fn++
			}
		}
	}
	if tp+fn == 0 {
		return 0
	}
	return float64(tp) / float64(tp+fn)
}

// F1Score returns the harmonic mean of [Precision] and [Recall] for a single
// class, or zero when both are zero.
func F1Score(yTrue, yPred []int, class int) float64 {
	p := Precision(yTrue, yPred, class)
	r := Recall(yTrue, yPred, class)
	if p+r == 0 {
		return 0
	}
	return 2 * p * r / (p + r)
}

// MacroF1 returns the unweighted mean of the per-class [F1Score] over all
// numClasses classes.
func MacroF1(yTrue, yPred []int, numClasses int) float64 {
	if numClasses == 0 {
		return 0
	}
	var sum float64
	for c := 0; c < numClasses; c++ {
		sum += F1Score(yTrue, yPred, c)
	}
	return sum / float64(numClasses)
}
