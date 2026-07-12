package flann

// Centroids returns the centres of the top-level clusters of the k-means tree —
// the means of the root node's immediate children — which are the coarsest
// partition the index computed. Each centre is a freshly allocated copy the
// caller may modify. If the whole dataset fit in a single leaf (or the dataset
// is empty) the root itself is returned as the sole centre (or no centres for an
// empty index). These are the vectors OpenCV exposes via
// flann::KMeansIndex::getClusterCenters for a one-level clustering.
func (idx *KMeansIndex) Centroids() [][]float64 {
	if idx.root == nil {
		return nil
	}
	if idx.root.leaf || len(idx.root.children) == 0 {
		return [][]float64{cloneVec(idx.root.center)}
	}
	out := make([][]float64, len(idx.root.children))
	for i, c := range idx.root.children {
		out[i] = cloneVec(c.center)
	}
	return out
}

// LeafCentroids returns the centre of every leaf of the k-means tree, i.e. the
// mean of each finest-grained cluster of points. Each centre is a freshly
// allocated copy. The order is a stable depth-first walk of the tree, so it is
// deterministic for a given build.
func (idx *KMeansIndex) LeafCentroids() [][]float64 {
	if idx.root == nil {
		return nil
	}
	var out [][]float64
	var walk func(n *kmeansNode)
	walk = func(n *kmeansNode) {
		if n.leaf {
			out = append(out, cloneVec(n.center))
			return
		}
		for _, c := range n.children {
			walk(c)
		}
	}
	walk(idx.root)
	return out
}

// ClusterCount returns the number of leaf clusters in the k-means tree, the
// length of the slice [KMeansIndex.LeafCentroids] returns.
func (idx *KMeansIndex) ClusterCount() int {
	if idx.root == nil {
		return 0
	}
	count := 0
	var walk func(n *kmeansNode)
	walk = func(n *kmeansNode) {
		if n.leaf {
			count++
			return
		}
		for _, c := range n.children {
			walk(c)
		}
	}
	walk(idx.root)
	return count
}
