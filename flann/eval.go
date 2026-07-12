package flann

// Recall measures how many true nearest neighbours an approximate index
// recovers, averaged over queries. For each query it takes the k neighbours the
// exact index returns as ground truth and reports the fraction of them that also
// appear in the approximate index's k results; the returned value is the mean
// over all queries, in [0, 1]. exact is normally a [LinearIndex] (or any index
// with unbounded checks). Queries for which the ground truth is empty are
// ignored; if none remain the result is 1.
func Recall[T any](approx, exact Index[T], queries []T, k int) float64 {
	if k <= 0 || len(queries) == 0 {
		return 1
	}
	var sum float64
	var counted int
	for _, q := range queries {
		truth := exact.KnnSearch(q, k)
		if len(truth) == 0 {
			continue
		}
		got := approx.KnnSearch(q, k)
		hits := overlap(truth, got)
		sum += float64(hits) / float64(len(truth))
		counted++
	}
	if counted == 0 {
		return 1
	}
	return sum / float64(counted)
}

// Precision measures how many of an approximate index's returned neighbours are
// genuine, averaged over queries. For each query it reports the fraction of the
// approximate result that also appears in the exact index's k results, then
// averages over all queries, in [0, 1]. Queries that return nothing are ignored;
// if none remain the result is 1. For k-nearest-neighbour search where both
// indices return k items precision and [Recall] coincide, but they differ once
// the approximate index returns fewer than k candidates.
func Precision[T any](approx, exact Index[T], queries []T, k int) float64 {
	if k <= 0 || len(queries) == 0 {
		return 1
	}
	var sum float64
	var counted int
	for _, q := range queries {
		got := approx.KnnSearch(q, k)
		if len(got) == 0 {
			continue
		}
		truth := exact.KnnSearch(q, k)
		hits := overlap(truth, got)
		sum += float64(hits) / float64(len(got))
		counted++
	}
	if counted == 0 {
		return 1
	}
	return sum / float64(counted)
}

// overlap counts how many indices appear in both result lists.
func overlap(truth, got []Neighbor) int {
	set := make(map[int]struct{}, len(truth))
	for _, nb := range truth {
		set[nb.Index] = struct{}{}
	}
	hits := 0
	for _, nb := range got {
		if _, ok := set[nb.Index]; ok {
			hits++
		}
	}
	return hits
}
