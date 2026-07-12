package flann

// KnnSearchBatch answers many k-nearest-neighbour queries in one call, the
// analogue of OpenCV's matrix-valued flann::Index::knnSearch. It returns one
// result slice per query, result[i] being exactly what idx.KnnSearch(queries[i],
// k) returns, in the same order as queries. This keeps the per-query results
// aligned with the input so callers need not thread indices through themselves.
// Queries are processed sequentially, so the call is safe for any [Index]
// implementation and fully deterministic.
func KnnSearchBatch[T any](idx Index[T], queries []T, k int) [][]Neighbor {
	out := make([][]Neighbor, len(queries))
	for i, q := range queries {
		out[i] = idx.KnnSearch(q, k)
	}
	return out
}

// RadiusSearchBatch answers many radius queries in one call. It returns one
// result slice per query, result[i] being exactly what
// idx.RadiusSearch(queries[i], radius) returns, aligned with queries. Queries are
// processed sequentially, so the call is safe for any [Index] implementation and
// fully deterministic.
func RadiusSearchBatch[T any](idx Index[T], queries []T, radius float64) [][]Neighbor {
	out := make([][]Neighbor, len(queries))
	for i, q := range queries {
		out[i] = idx.RadiusSearch(q, radius)
	}
	return out
}
