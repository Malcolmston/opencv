package flann

// CompositeIndex searches a real-valued dataset with two complementary
// structures at once — a randomized k-d forest ([KDForestIndex]) and a
// hierarchical k-means tree ([KMeansIndex]) — and merges their results. The two
// families make different approximation errors, so a point missed by one is
// often found by the other; querying both and keeping the union of candidates
// yields higher recall at a given total budget than either alone. This mirrors
// OpenCV's flann CompositeIndex.
//
// With both sub-indices exact (MaxChecks/Checks == 0, the default) the merged
// result is exact. Set [CompositeIndex.MaxChecks] and [CompositeIndex.Checks] to
// bound the forest and the k-means tree respectively.
type CompositeIndex struct {
	// MaxChecks bounds the k-d forest search; 0 means exact.
	MaxChecks int
	// Checks bounds the k-means search; 0 means exact.
	Checks int

	forest *KDForestIndex
	kmeans *KMeansIndex
}

// NewCompositeIndex builds a composite index over the real-valued dataset. trees
// is the number of randomized k-d trees, branching and leafSize configure the
// k-means tree (all accept 0 for their defaults) and seed makes both structures
// reproducible. It panics if the dataset is ragged. An empty dataset is allowed
// and yields empty searches.
func NewCompositeIndex(data [][]float64, trees, branching, leafSize int, seed int64) *CompositeIndex {
	return &CompositeIndex{
		forest: NewKDForestIndex(data, trees, seed),
		kmeans: NewKMeansIndex(data, branching, leafSize, seed),
	}
}

// Size returns the number of points in the index.
func (c *CompositeIndex) Size() int { return c.forest.Size() }

// KnnSearch returns the k nearest neighbours of query by querying both
// sub-indices and re-ranking the union of their candidates. With both budgets
// unbounded the result is exact.
func (c *CompositeIndex) KnnSearch(query []float64, k int) []Neighbor {
	if k <= 0 {
		return nil
	}
	c.forest.MaxChecks = c.MaxChecks
	c.kmeans.Checks = c.Checks
	a := c.forest.KnnSearch(query, k)
	b := c.kmeans.KnnSearch(query, k)
	return mergeKnn(a, b, k)
}

// RadiusSearch returns every point within radius of query. Both sub-searches are
// exact, so their union is exact; duplicates are removed.
func (c *CompositeIndex) RadiusSearch(query []float64, radius float64) []Neighbor {
	a := c.forest.RadiusSearch(query, radius)
	b := c.kmeans.RadiusSearch(query, radius)
	return mergeRadius(a, b)
}

// mergeKnn unions two ranked result lists, discards duplicate indices and keeps
// the k nearest, in canonical order.
func mergeKnn(a, b []Neighbor, k int) []Neighbor {
	res := &knnSet{k: k}
	seen := make(map[int]struct{}, len(a)+len(b))
	for _, src := range [][]Neighbor{a, b} {
		for _, nb := range src {
			if _, ok := seen[nb.Index]; ok {
				continue
			}
			seen[nb.Index] = struct{}{}
			res.add(nb)
		}
	}
	return res.n
}

// mergeRadius unions two radius result lists, discards duplicates and returns
// them in canonical order.
func mergeRadius(a, b []Neighbor) []Neighbor {
	seen := make(map[int]struct{}, len(a)+len(b))
	var out []Neighbor
	for _, src := range [][]Neighbor{a, b} {
		for _, nb := range src {
			if _, ok := seen[nb.Index]; ok {
				continue
			}
			seen[nb.Index] = struct{}{}
			out = append(out, nb)
		}
	}
	sortNeighbors(out)
	return out
}
