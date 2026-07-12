package ml

// Accuracy returns the fraction of predictions that match the ground truth, a
// value in [0, 1]. It panics if the two slices differ in length. An empty input
// yields 0.
func Accuracy(predicted, actual []int) float64 {
	if len(predicted) != len(actual) {
		panic("ml: Accuracy requires predicted and actual to have equal length")
	}
	if len(actual) == 0 {
		return 0
	}
	var correct int
	for i := range actual {
		if predicted[i] == actual[i] {
			correct++
		}
	}
	return float64(correct) / float64(len(actual))
}

// ConfusionMatrix returns a numClasses×numClasses table whose entry [a][p]
// counts the samples whose actual label is a and whose predicted label is p.
// Labels are used directly as indices, so they must lie in [0, numClasses). It
// panics if the two slices differ in length or a label is out of range.
func ConfusionMatrix(predicted, actual []int, numClasses int) [][]int {
	if len(predicted) != len(actual) {
		panic("ml: ConfusionMatrix requires predicted and actual to have equal length")
	}
	if numClasses <= 0 {
		panic("ml: ConfusionMatrix requires numClasses > 0")
	}
	m := make([][]int, numClasses)
	for i := range m {
		m[i] = make([]int, numClasses)
	}
	for i := range actual {
		a, p := actual[i], predicted[i]
		if a < 0 || a >= numClasses || p < 0 || p >= numClasses {
			panic("ml: ConfusionMatrix label out of range [0, numClasses)")
		}
		m[a][p]++
	}
	return m
}
