package segmentation

import (
	"math"
	"sort"

	cv "github.com/malcolmston/opencv"
)

// RAG is a Region Adjacency Graph: the nodes are the regions of a [LabelMap] and
// the edges connect regions that touch, weighted by the Euclidean distance
// between their mean colours. It mirrors skimage's graph.RAG and underpins the
// region-merging routines [RAG.MergeByColor] and [RAG.MergeBySize].
type RAG struct {
	rows, cols int
	channels   int
	count      int
	size       []int         // pixels per region
	meanColor  [][]float64   // channels-length mean colour per region
	adj        map[pair]bool // unordered adjacency between regions
	srcLabels  []int         // per-pixel source labelling, retained for merging
}

// BuildRAG constructs the region adjacency graph of lm over the colours of img.
// img must have the same dimensions as lm. Each region's mean colour is measured
// from img and two regions are adjacent when they share a 4-connected boundary.
//
// It panics if the dimensions disagree or img is empty.
func BuildRAG(lm *LabelMap, img *cv.Mat) *RAG {
	if img.Empty() {
		panic("segmentation: BuildRAG on empty image")
	}
	if img.Rows != lm.Rows || img.Cols != lm.Cols {
		panic("segmentation: BuildRAG image and label map dimensions differ")
	}
	ch := img.Channels
	r := &RAG{
		rows: lm.Rows, cols: lm.Cols, channels: ch, count: lm.Count,
		size:      make([]int, lm.Count),
		meanColor: make([][]float64, lm.Count),
		adj:       make(map[pair]bool),
		srcLabels: append([]int(nil), lm.Labels...),
	}
	sum := make([][]float64, lm.Count)
	for i := range sum {
		sum[i] = make([]float64, ch)
		r.meanColor[i] = make([]float64, ch)
	}
	for y := 0; y < lm.Rows; y++ {
		for x := 0; x < lm.Cols; x++ {
			lbl := lm.Labels[y*lm.Cols+x]
			r.size[lbl]++
			b := (y*lm.Cols + x) * ch
			for c := 0; c < ch; c++ {
				sum[lbl][c] += float64(img.Data[b+c])
			}
			if x+1 < lm.Cols {
				if nb := lm.Labels[y*lm.Cols+x+1]; nb != lbl {
					r.adj[orderedPair(lbl, nb)] = true
				}
			}
			if y+1 < lm.Rows {
				if nb := lm.Labels[(y+1)*lm.Cols+x]; nb != lbl {
					r.adj[orderedPair(lbl, nb)] = true
				}
			}
		}
	}
	for i := 0; i < lm.Count; i++ {
		if r.size[i] == 0 {
			continue
		}
		for c := 0; c < ch; c++ {
			r.meanColor[i][c] = sum[i][c] / float64(r.size[i])
		}
	}
	return r
}

// Regions returns the number of regions (nodes) in the graph.
func (r *RAG) Regions() int { return r.count }

// MeanColor returns a copy of the mean colour of region i.
func (r *RAG) MeanColor(i int) []float64 {
	out := make([]float64, r.channels)
	copy(out, r.meanColor[i])
	return out
}

// Neighbors returns the labels of the regions adjacent to region i, in
// ascending order.
func (r *RAG) Neighbors(i int) []int {
	var out []int
	for p := range r.adj {
		if p.a == i {
			out = append(out, p.b)
		} else if p.b == i {
			out = append(out, p.a)
		}
	}
	sort.Ints(out)
	return out
}

// EdgeWeight returns the Euclidean distance between the mean colours of regions
// a and b, the weight used for merging. Non-adjacent regions still return a
// well-defined value.
func (r *RAG) EdgeWeight(a, b int) float64 {
	return colorDist(r.meanColor[a], r.meanColor[b])
}

// MergeByColor greedily merges adjacent regions whose mean-colour distance is at
// most threshold, the classic RAG threshold-merging step. Merging proceeds from
// the closest colour pair upward, recomputing size-weighted mean colours after
// each merge, until no adjacent pair is within the threshold. It returns a new
// [LabelMap] with the merged, consecutively renumbered regions and does not
// modify the receiver.
func (r *RAG) MergeByColor(threshold float64) *LabelMap {
	uf := newUnionFind(r.count)
	// Working copies of the per-region statistics, keyed by union-find root.
	size := make([]int, r.count)
	copy(size, r.size)
	mean := make([][]float64, r.count)
	for i := range mean {
		mean[i] = make([]float64, r.channels)
		copy(mean[i], r.meanColor[i])
	}
	edges := r.edgeList()

	for {
		bestW := math.MaxFloat64
		bestA, bestB := -1, -1
		for _, e := range edges {
			ra, rb := uf.find(e.a), uf.find(e.b)
			if ra == rb {
				continue
			}
			w := colorDist(mean[ra], mean[rb])
			if w < bestW {
				bestW = w
				bestA, bestB = ra, rb
			}
		}
		if bestA < 0 || bestW > threshold {
			break
		}
		root := uf.union(bestA, bestB)
		other := bestA
		if root == bestA {
			other = bestB
		}
		mergeMean(mean[root], size[root], mean[other], size[other])
		size[root] += size[other]
	}
	return r.applyMerge(uf)
}

// MergeBySize merges every region smaller than minSize pixels into the adjacent
// region whose mean colour is closest, repeating until no region below the
// threshold has an eligible neighbour. This removes speckle from an
// over-segmentation. It returns a new [LabelMap] and does not modify the
// receiver.
func (r *RAG) MergeBySize(minSize int) *LabelMap {
	uf := newUnionFind(r.count)
	size := make([]int, r.count)
	copy(size, r.size)
	mean := make([][]float64, r.count)
	for i := range mean {
		mean[i] = make([]float64, r.channels)
		copy(mean[i], r.meanColor[i])
	}
	edges := r.edgeList()

	for {
		// Find the smallest under-sized region that has at least one neighbour.
		merged := false
		// Deterministic order: ascending root index.
		for root := 0; root < r.count; root++ {
			if uf.find(root) != root || size[root] >= minSize {
				continue
			}
			// Closest-colour neighbour.
			bestW := math.MaxFloat64
			bestN := -1
			for _, e := range edges {
				var nb int
				ra, rb := uf.find(e.a), uf.find(e.b)
				switch {
				case ra == root && rb != root:
					nb = rb
				case rb == root && ra != root:
					nb = ra
				default:
					continue
				}
				w := colorDist(mean[root], mean[nb])
				if w < bestW {
					bestW = w
					bestN = nb
				}
			}
			if bestN < 0 {
				continue
			}
			newRoot := uf.union(root, bestN)
			absorbed := root
			if newRoot == root {
				absorbed = bestN
			}
			mergeMean(mean[newRoot], size[newRoot], mean[absorbed], size[absorbed])
			size[newRoot] += size[absorbed]
			merged = true
			break
		}
		if !merged {
			break
		}
	}
	return r.applyMerge(uf)
}

// edge is an adjacency edge between two region labels.
type ragEdge struct{ a, b int }

// edgeList returns the adjacency edges in a deterministic order.
func (r *RAG) edgeList() []ragEdge {
	edges := make([]ragEdge, 0, len(r.adj))
	for p := range r.adj {
		edges = append(edges, ragEdge(p))
	}
	sort.Slice(edges, func(i, j int) bool {
		if edges[i].a != edges[j].a {
			return edges[i].a < edges[j].a
		}
		return edges[i].b < edges[j].b
	})
	return edges
}

// applyMerge builds a fresh LabelMap whose pixels carry the union-find root of
// their original region, renumbered consecutively.
func (r *RAG) applyMerge(uf *unionFind) *LabelMap {
	// Map every pixel's original region to its union-find root.
	raw := make([]int, len(r.srcLabels))
	for i, lbl := range r.srcLabels {
		raw[i] = uf.find(lbl)
	}
	labels, count := relabelConsecutive(raw)
	return &LabelMap{Rows: r.rows, Cols: r.cols, Count: count, Labels: labels}
}

// mergeMean updates dst (the size-dstSize mean) in place to the size-weighted
// average of itself and src.
func mergeMean(dst []float64, dstSize int, src []float64, srcSize int) {
	total := float64(dstSize + srcSize)
	if total == 0 {
		return
	}
	for c := range dst {
		dst[c] = (dst[c]*float64(dstSize) + src[c]*float64(srcSize)) / total
	}
}

// colorDist is the Euclidean distance between two equal-length colour vectors.
func colorDist(a, b []float64) float64 {
	var s float64
	for i := range a {
		d := a[i] - b[i]
		s += d * d
	}
	return math.Sqrt(s)
}
