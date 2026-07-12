package segmentation

import (
	"math"
	"sort"

	cv "github.com/malcolmston/opencv"
)

// EfficientGraphSegmentation segments img with the Felzenszwalb-Huttenlocher
// graph-based algorithm ("Efficient Graph-Based Image Segmentation", 2004), the
// method behind OpenCV's ximgproc GraphSegmentation. It returns a [LabelMap]
// whose regions are the components of a minimum-spanning-forest built over the
// pixel grid.
//
// The image is treated as a graph: every pixel is a node connected to its 8
// neighbours by an edge whose weight is the Euclidean colour distance between
// the two pixels. Edges are processed in nondecreasing weight order and the
// endpoints are merged when the edge weight does not exceed the internal
// variation of either component, defined as the largest edge inside that
// component plus a threshold k/size. A larger k yields larger regions.
//
// sigma controls an optional Gaussian pre-smoothing ([cv.GaussianBlur]); pass 0
// to skip it. After the main pass, components smaller than minSize pixels are
// merged into the adjacent component joined by the lightest edge, matching the
// reference post-processing. img may be single- or three-channel.
//
// It panics if img is empty.
func EfficientGraphSegmentation(img *cv.Mat, sigma, k float64, minSize int) *LabelMap {
	if img.Empty() {
		panic("segmentation: EfficientGraphSegmentation on empty image")
	}
	src := img
	if sigma > 0 {
		src = cv.GaussianBlur(img, 5, sigma)
	}
	rows, cols, ch := src.Rows, src.Cols, src.Channels
	n := rows * cols

	// Collect grid edges (right, down, and the two diagonals) with colour-distance
	// weights so every pair is visited once.
	type edge struct {
		a, b int
		w    float64
	}
	edges := make([]edge, 0, n*4)
	pixDist := func(ia, ib int) float64 {
		ba, bb := ia*ch, ib*ch
		var s float64
		for c := 0; c < ch; c++ {
			d := float64(src.Data[ba+c]) - float64(src.Data[bb+c])
			s += d * d
		}
		return math.Sqrt(s)
	}
	add := func(x, y, nx, ny int) {
		a := y*cols + x
		b := ny*cols + nx
		edges = append(edges, edge{a: a, b: b, w: pixDist(a, b)})
	}
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			if x+1 < cols {
				add(x, y, x+1, y)
			}
			if y+1 < rows {
				add(x, y, x, y+1)
			}
			if x+1 < cols && y+1 < rows {
				add(x, y, x+1, y+1)
			}
			if x-1 >= 0 && y+1 < rows {
				add(x, y, x-1, y+1)
			}
		}
	}
	sort.SliceStable(edges, func(i, j int) bool { return edges[i].w < edges[j].w })

	uf := newUnionFind(n)
	internal := make([]float64, n) // largest internal edge weight per component root
	threshold := func(root int) float64 { return k / float64(uf.setSize(root)) }

	for _, e := range edges {
		ra, rb := uf.find(e.a), uf.find(e.b)
		if ra == rb {
			continue
		}
		if e.w <= internal[ra]+threshold(ra) && e.w <= internal[rb]+threshold(rb) {
			root := uf.union(ra, rb)
			if e.w > internal[root] {
				internal[root] = e.w
			}
		}
	}

	// Merge tiny components across the lightest connecting edge.
	if minSize > 1 {
		for _, e := range edges {
			ra, rb := uf.find(e.a), uf.find(e.b)
			if ra == rb {
				continue
			}
			if uf.size[ra] < minSize || uf.size[rb] < minSize {
				uf.union(ra, rb)
			}
		}
	}

	raw := make([]int, n)
	for i := 0; i < n; i++ {
		raw[i] = uf.find(i)
	}
	labels, count := relabelConsecutive(raw)
	return &LabelMap{Rows: rows, Cols: cols, Count: count, Labels: labels}
}
