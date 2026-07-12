package flann

// Compile-time checks that every additional index satisfies the common
// [Index] interface, alongside the assertions in index.go.
var (
	_ Index[[]float64] = (*KDForestIndex)(nil)
	_ Index[[]float64] = (*HierarchicalClusteringIndex)(nil)
	_ Index[[]float64] = (*CompositeIndex)(nil)
	_ Index[[]float64] = (*AutotunedIndex)(nil)
)
