package segment2

import (
	cv "github.com/malcolmston/opencv"
)

// segment2uf is a disjoint-set forest with size-weighted union by index.
type segment2uf struct {
	parent []int
}

func segment2newUF(n int) *segment2uf {
	u := &segment2uf{parent: make([]int, n)}
	for i := range u.parent {
		u.parent[i] = i
	}
	return u
}

func (u *segment2uf) find(x int) int {
	for u.parent[x] != x {
		u.parent[x] = u.parent[u.parent[x]]
		x = u.parent[x]
	}
	return x
}

func (u *segment2uf) union(a, b int) {
	ra, rb := u.find(a), u.find(b)
	if ra == rb {
		return
	}
	if ra < rb {
		u.parent[rb] = ra
	} else {
		u.parent[ra] = rb
	}
}

// segment2region is a rectangular block produced by the split phase.
type segment2region struct {
	x, y, w, h int
}

// segment2homogeneous reports whether the block's per-channel intensity range is
// within threshold everywhere.
func segment2homogeneous(img *cv.Mat, r segment2region, threshold float64) bool {
	ch := img.Channels
	var mn, mx [16]float64
	for c := 0; c < ch; c++ {
		mn[c] = 1e18
		mx[c] = -1e18
	}
	for yy := r.y; yy < r.y+r.h; yy++ {
		for xx := r.x; xx < r.x+r.w; xx++ {
			b := (yy*img.Cols + xx) * ch
			for c := 0; c < ch; c++ {
				v := float64(img.Data[b+c])
				if v < mn[c] {
					mn[c] = v
				}
				if v > mx[c] {
					mx[c] = v
				}
			}
		}
	}
	for c := 0; c < ch; c++ {
		if mx[c]-mn[c] > threshold {
			return false
		}
	}
	return true
}

// SplitAndMerge segments img with the classic quadtree split-and-merge method.
// The split phase recursively divides the image into square blocks until each
// block is homogeneous (per-channel intensity range <= threshold) or reaches the
// minimum block size minSize. The merge phase then unions spatially adjacent
// blocks whose mean colours differ by no more than threshold. The result is a
// [LabelMap] with contiguous labels.
//
// It panics if img is empty or minSize < 1.
func SplitAndMerge(img *cv.Mat, threshold float64, minSize int) *LabelMap {
	segment2requireNonEmpty(img, "SplitAndMerge")
	if minSize < 1 {
		panic("segment2: SplitAndMerge requires minSize >= 1")
	}
	rows, cols := img.Rows, img.Cols

	var leaves []segment2region
	var split func(r segment2region)
	split = func(r segment2region) {
		if r.w <= minSize && r.h <= minSize {
			leaves = append(leaves, r)
			return
		}
		if segment2homogeneous(img, r, threshold) {
			leaves = append(leaves, r)
			return
		}
		hw := (r.w + 1) / 2
		hh := (r.h + 1) / 2
		// Up to four children, clipped to the region.
		children := []segment2region{
			{r.x, r.y, hw, hh},
			{r.x + hw, r.y, r.w - hw, hh},
			{r.x, r.y + hh, hw, r.h - hh},
			{r.x + hw, r.y + hh, r.w - hw, r.h - hh},
		}
		for _, c := range children {
			if c.w > 0 && c.h > 0 {
				split(c)
			}
		}
	}
	split(segment2region{0, 0, cols, rows})

	// Assign each leaf a provisional label and record which leaf owns each
	// pixel.
	pixLeaf := make([]int, rows*cols)
	means := make([][]float64, len(leaves))
	ch := img.Channels
	for li, r := range leaves {
		m := make([]float64, ch)
		cnt := 0
		for yy := r.y; yy < r.y+r.h; yy++ {
			for xx := r.x; xx < r.x+r.w; xx++ {
				pixLeaf[yy*cols+xx] = li
				b := (yy*cols + xx) * ch
				for c := 0; c < ch; c++ {
					m[c] += float64(img.Data[b+c])
				}
				cnt++
			}
		}
		for c := 0; c < ch; c++ {
			m[c] /= float64(cnt)
		}
		means[li] = m
	}

	// Merge adjacent leaves with similar means.
	uf := segment2newUF(len(leaves))
	thr2 := threshold * threshold
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			li := pixLeaf[y*cols+x]
			if x+1 < cols {
				rj := pixLeaf[y*cols+x+1]
				if rj != li && segment2colorDist2(means[li], means[rj]) <= thr2 {
					uf.union(li, rj)
				}
			}
			if y+1 < rows {
				rj := pixLeaf[(y+1)*cols+x]
				if rj != li && segment2colorDist2(means[li], means[rj]) <= thr2 {
					uf.union(li, rj)
				}
			}
		}
	}

	lm := NewLabelMap(rows, cols)
	for i := range lm.Labels {
		lm.Labels[i] = uf.find(pixLeaf[i])
	}
	lm.NumLabels = len(leaves)
	lm.Compact()
	return lm
}
