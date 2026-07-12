package segmentation

import (
	"sort"

	cv "github.com/malcolmston/opencv"
)

// ssColorBins is the number of histogram bins per channel used by the colour
// similarity in selective search.
const ssColorBins = 8

// SelectiveSearchSegmentation generates object-region proposals from img using
// the selective-search strategy of Uijlings et al. (2013), the algorithm behind
// OpenCV's ximgproc SelectiveSearchSegmentation. It returns candidate bounding
// boxes ordered from largest to smallest area, with duplicates removed.
//
// An initial over-segmentation is produced with [EfficientGraphSegmentation]
// using the given sigma, k and minSize. Adjacent regions are then greedily
// merged in order of a hierarchical similarity that combines colour-histogram
// intersection, a size term favouring the growth of small regions, and a fill
// term favouring merges that fit tightly inside their joint bounding box. The
// bounding box of every region that ever exists — the initial regions plus each
// region formed by a merge — becomes a proposal.
//
// img must be three-channel. It panics if img is empty or not three-channel.
func SelectiveSearchSegmentation(img *cv.Mat, sigma, k float64, minSize int) []cv.Rect {
	if img.Empty() {
		panic("segmentation: SelectiveSearchSegmentation on empty image")
	}
	if img.Channels != 3 {
		panic("segmentation: SelectiveSearchSegmentation requires a 3-channel image")
	}
	lm := EfficientGraphSegmentation(img, sigma, k, minSize)
	rows, cols := lm.Rows, lm.Cols
	nreg := lm.Count

	// Per-region statistics: pixel count, bounding box and a colour histogram.
	type region struct {
		size       int
		minX, minY int
		maxX, maxY int
		hist       []float64 // ssColorBins*3, L1-normalised
		alive      bool
	}
	regs := make([]region, nreg)
	for i := range regs {
		regs[i].alive = true
		regs[i].hist = make([]float64, ssColorBins*3)
		regs[i].minX, regs[i].minY = cols, rows
	}
	binOf := func(v uint8) int {
		b := int(v) * ssColorBins / 256
		if b >= ssColorBins {
			b = ssColorBins - 1
		}
		return b
	}
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			r := lm.Labels[y*cols+x]
			reg := &regs[r]
			reg.size++
			if x < reg.minX {
				reg.minX = x
			}
			if x > reg.maxX {
				reg.maxX = x
			}
			if y < reg.minY {
				reg.minY = y
			}
			if y > reg.maxY {
				reg.maxY = y
			}
			b := (y*cols + x) * 3
			for c := 0; c < 3; c++ {
				reg.hist[c*ssColorBins+binOf(img.Data[b+c])]++
			}
		}
	}
	for i := range regs {
		l1 := 0.0
		for _, v := range regs[i].hist {
			l1 += v
		}
		if l1 > 0 {
			for j := range regs[i].hist {
				regs[i].hist[j] /= l1
			}
		}
	}

	// Region adjacency from the label boundaries.
	adj := adjacentRegionPairs(lm)

	imSize := float64(rows * cols)
	histIntersect := func(a, b *region) float64 {
		s := 0.0
		for i := range a.hist {
			if a.hist[i] < b.hist[i] {
				s += a.hist[i]
			} else {
				s += b.hist[i]
			}
		}
		return s
	}
	similarity := func(a, b *region) float64 {
		sColor := histIntersect(a, b)
		sSize := 1.0 - float64(a.size+b.size)/imSize
		// Fill: penalise merges whose joint bounding box is much larger than the
		// two regions combined.
		bx0 := min2(a.minX, b.minX)
		by0 := min2(a.minY, b.minY)
		bx1 := max2(a.maxX, b.maxX)
		by1 := max2(a.maxY, b.maxY)
		bbox := float64((bx1 - bx0 + 1) * (by1 - by0 + 1))
		sFill := 1.0 - (bbox-float64(a.size)-float64(b.size))/imSize
		return sColor + sSize + sFill
	}

	uf := newUnionFind(nreg)
	// Track proposals as we go: start with every initial region.
	var proposals []cv.Rect
	pushRect := func(r *region) {
		proposals = append(proposals, cv.Rect{X: r.minX, Y: r.minY, Width: r.maxX - r.minX + 1, Height: r.maxY - r.minY + 1})
	}
	for i := range regs {
		pushRect(&regs[i])
	}

	// Greedy hierarchical merging. Each round recomputes the best adjacent pair;
	// this is O(rounds * edges) which is ample for test-scale images and keeps the
	// ordering deterministic.
	edgeList := make([][2]int, 0, len(adj))
	for pair := range adj {
		edgeList = append(edgeList, [2]int{pair.a, pair.b})
	}
	sort.Slice(edgeList, func(i, j int) bool {
		if edgeList[i][0] != edgeList[j][0] {
			return edgeList[i][0] < edgeList[j][0]
		}
		return edgeList[i][1] < edgeList[j][1]
	})

	for {
		bestSim := -1.0
		bestA, bestB := -1, -1
		for _, e := range edgeList {
			ra, rb := uf.find(e[0]), uf.find(e[1])
			if ra == rb || !regs[ra].alive || !regs[rb].alive {
				continue
			}
			s := similarity(&regs[ra], &regs[rb])
			if s > bestSim {
				bestSim = s
				bestA, bestB = ra, rb
			}
		}
		if bestA < 0 {
			break
		}
		root := uf.union(bestA, bestB)
		other := bestA
		if root == bestA {
			other = bestB
		}
		// Fold the absorbed region's statistics into the root.
		dst, srcReg := &regs[root], &regs[other]
		total := float64(dst.size + srcReg.size)
		for i := range dst.hist {
			dst.hist[i] = (dst.hist[i]*float64(dst.size) + srcReg.hist[i]*float64(srcReg.size)) / total
		}
		dst.size += srcReg.size
		dst.minX = min2(dst.minX, srcReg.minX)
		dst.minY = min2(dst.minY, srcReg.minY)
		dst.maxX = max2(dst.maxX, srcReg.maxX)
		dst.maxY = max2(dst.maxY, srcReg.maxY)
		srcReg.alive = false
		dst.alive = true
		pushRect(dst)
	}

	return dedupRects(proposals)
}

// adjacentRegionPairs returns the set of unordered region pairs that share a
// 4-connected boundary in lm.
func adjacentRegionPairs(lm *LabelMap) map[pair]struct{} {
	adj := make(map[pair]struct{})
	for y := 0; y < lm.Rows; y++ {
		for x := 0; x < lm.Cols; x++ {
			a := lm.Labels[y*lm.Cols+x]
			if x+1 < lm.Cols {
				b := lm.Labels[y*lm.Cols+x+1]
				if a != b {
					adj[orderedPair(a, b)] = struct{}{}
				}
			}
			if y+1 < lm.Rows {
				b := lm.Labels[(y+1)*lm.Cols+x]
				if a != b {
					adj[orderedPair(a, b)] = struct{}{}
				}
			}
		}
	}
	return adj
}

// pair is an unordered region-index pair with a < b.
type pair struct{ a, b int }

func orderedPair(a, b int) pair {
	if a < b {
		return pair{a, b}
	}
	return pair{b, a}
}

func dedupRects(in []cv.Rect) []cv.Rect {
	seen := make(map[cv.Rect]struct{}, len(in))
	out := make([]cv.Rect, 0, len(in))
	for _, r := range in {
		if _, ok := seen[r]; ok {
			continue
		}
		seen[r] = struct{}{}
		out = append(out, r)
	}
	sort.SliceStable(out, func(i, j int) bool {
		ai := out[i].Width * out[i].Height
		aj := out[j].Width * out[j].Height
		if ai != aj {
			return ai > aj
		}
		if out[i].Y != out[j].Y {
			return out[i].Y < out[j].Y
		}
		return out[i].X < out[j].X
	})
	return out
}

func min2(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func max2(a, b int) int {
	if a > b {
		return a
	}
	return b
}
