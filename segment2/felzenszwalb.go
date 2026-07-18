package segment2

import (
	"math"
	"sort"

	cv "github.com/malcolmston/opencv"
)

// segment2edge is a weighted edge of the pixel grid graph.
type segment2edge struct {
	a, b int
	w    float64
}

// segment2gaussianBlur applies a small separable Gaussian to each channel of img
// and returns a new [cv.Mat]. sigma <= 0 returns a clone.
func segment2gaussianBlur(img *cv.Mat, sigma float64) *cv.Mat {
	if sigma <= 0 {
		return img.Clone()
	}
	radius := int(math.Ceil(sigma * 3))
	if radius < 1 {
		radius = 1
	}
	kernel := make([]float64, 2*radius+1)
	var sum float64
	for i := -radius; i <= radius; i++ {
		v := math.Exp(-float64(i*i) / (2 * sigma * sigma))
		kernel[i+radius] = v
		sum += v
	}
	for i := range kernel {
		kernel[i] /= sum
	}
	rows, cols, ch := img.Rows, img.Cols, img.Channels
	clampY := func(y int) int {
		if y < 0 {
			return 0
		}
		if y >= rows {
			return rows - 1
		}
		return y
	}
	clampX := func(x int) int {
		if x < 0 {
			return 0
		}
		if x >= cols {
			return cols - 1
		}
		return x
	}
	tmp := make([]float64, rows*cols*ch)
	// Horizontal pass.
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			for c := 0; c < ch; c++ {
				var acc float64
				for k := -radius; k <= radius; k++ {
					sx := clampX(x + k)
					acc += kernel[k+radius] * float64(img.Data[(y*cols+sx)*ch+c])
				}
				tmp[(y*cols+x)*ch+c] = acc
			}
		}
	}
	// Vertical pass.
	out := cv.NewMat(rows, cols, ch)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			for c := 0; c < ch; c++ {
				var acc float64
				for k := -radius; k <= radius; k++ {
					sy := clampY(y + k)
					acc += kernel[k+radius] * tmp[(sy*cols+x)*ch+c]
				}
				out.Data[(y*cols+x)*ch+c] = segment2clampU8(acc)
			}
		}
	}
	return out
}

// Felzenszwalb segments img with the Felzenszwalb-Huttenlocher graph-based
// method — a minimum-spanning-forest over the 8-connected pixel grid whose edge
// weights are colour differences. img is first smoothed by a Gaussian of the
// given sigma. Two components are merged while the edge joining them is no
// larger than the internal difference of both plus k/size, so k sets the
// preferred region scale. Components smaller than minSize pixels are then merged
// into their nearest neighbour. The result is a [LabelMap] with contiguous
// labels.
//
// It panics if img is empty, k <= 0 or minSize < 1.
func Felzenszwalb(img *cv.Mat, sigma, k float64, minSize int) *LabelMap {
	segment2requireNonEmpty(img, "Felzenszwalb")
	if k <= 0 {
		panic("segment2: Felzenszwalb requires k > 0")
	}
	if minSize < 1 {
		panic("segment2: Felzenszwalb requires minSize >= 1")
	}
	sm := segment2gaussianBlur(img, sigma)
	rows, cols := sm.Rows, sm.Cols
	n := rows * cols

	colAt := make([][]float64, n)
	for i := 0; i < n; i++ {
		colAt[i] = segment2colorAt(sm, i%cols, i/cols)
	}

	var edges []segment2edge
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			i := y*cols + x
			if x+1 < cols {
				j := i + 1
				edges = append(edges, segment2edge{i, j, segment2colorDist(colAt[i], colAt[j])})
			}
			if y+1 < rows {
				j := i + cols
				edges = append(edges, segment2edge{i, j, segment2colorDist(colAt[i], colAt[j])})
			}
			if x+1 < cols && y+1 < rows {
				j := i + cols + 1
				edges = append(edges, segment2edge{i, j, segment2colorDist(colAt[i], colAt[j])})
			}
			if x > 0 && y+1 < rows {
				j := i + cols - 1
				edges = append(edges, segment2edge{i, j, segment2colorDist(colAt[i], colAt[j])})
			}
		}
	}
	sort.SliceStable(edges, func(a, b int) bool { return edges[a].w < edges[b].w })

	uf := segment2newUF(n)
	size := make([]int, n)
	intDiff := make([]float64, n)
	for i := range size {
		size[i] = 1
	}
	thr := func(comp int) float64 { return k / float64(size[comp]) }

	for _, e := range edges {
		a := uf.find(e.a)
		b := uf.find(e.b)
		if a == b {
			continue
		}
		if e.w <= intDiff[a]+thr(a) && e.w <= intDiff[b]+thr(b) {
			uf.union(a, b)
			root := uf.find(a)
			size[root] = size[a] + size[b]
			if e.w > intDiff[root] {
				intDiff[root] = e.w
			}
		}
	}

	// Enforce minimum component size.
	for _, e := range edges {
		a := uf.find(e.a)
		b := uf.find(e.b)
		if a != b && (size[a] < minSize || size[b] < minSize) {
			uf.union(a, b)
			root := uf.find(a)
			size[root] = size[a] + size[b]
		}
	}

	lm := NewLabelMap(rows, cols)
	for i := range lm.Labels {
		lm.Labels[i] = uf.find(i)
	}
	lm.NumLabels = n
	lm.Compact()
	return lm
}
