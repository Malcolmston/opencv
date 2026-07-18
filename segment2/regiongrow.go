package segment2

import (
	cv "github.com/malcolmston/opencv"
)

// RegionGrow grows a single connected region outward from seed, adding a
// neighbouring pixel whenever its colour lies within tolerance (Euclidean colour
// distance) of the seed pixel's colour. It returns a single-channel mask that is
// 255 inside the region and 0 outside. conn selects 4- or 8-connectivity.
//
// It panics if img is empty, the seed is out of bounds, or conn is invalid.
func RegionGrow(img *cv.Mat, seed cv.Point, tolerance float64, conn Connectivity) *cv.Mat {
	segment2requireNonEmpty(img, "RegionGrow")
	if seed.X < 0 || seed.X >= img.Cols || seed.Y < 0 || seed.Y >= img.Rows {
		panic("segment2: RegionGrow seed out of bounds")
	}
	nb := segment2neighbors(conn)
	rows, cols := img.Rows, img.Cols
	mask := cv.NewMat(rows, cols, 1)
	seedCol := segment2colorAt(img, seed.X, seed.Y)
	tol2 := tolerance * tolerance

	stack := []int{seed.Y*cols + seed.X}
	mask.Data[seed.Y*cols+seed.X] = 255
	cur := make([]float64, img.Channels)
	for len(stack) > 0 {
		i := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		x := i % cols
		y := i / cols
		for _, o := range nb {
			nx, ny := x+o.dx, y+o.dy
			if nx < 0 || nx >= cols || ny < 0 || ny >= rows {
				continue
			}
			ni := ny*cols + nx
			if mask.Data[ni] != 0 {
				continue
			}
			segment2colorInto(img, nx, ny, cur)
			if segment2colorDist2(cur, seedCol) <= tol2 {
				mask.Data[ni] = 255
				stack = append(stack, ni)
			}
		}
	}
	return mask
}

// RegionGrowMean is like [RegionGrow] but tests each candidate pixel against the
// running mean colour of the region grown so far rather than the fixed seed
// colour. This adapts to smooth colour gradients. It returns the region mask.
//
// It panics if img is empty, the seed is out of bounds, or conn is invalid.
func RegionGrowMean(img *cv.Mat, seed cv.Point, tolerance float64, conn Connectivity) *cv.Mat {
	segment2requireNonEmpty(img, "RegionGrowMean")
	if seed.X < 0 || seed.X >= img.Cols || seed.Y < 0 || seed.Y >= img.Rows {
		panic("segment2: RegionGrowMean seed out of bounds")
	}
	nb := segment2neighbors(conn)
	rows, cols, ch := img.Rows, img.Cols, img.Channels
	mask := cv.NewMat(rows, cols, 1)
	tol2 := tolerance * tolerance

	mean := segment2colorAt(img, seed.X, seed.Y)
	count := 1.0
	stack := []int{seed.Y*cols + seed.X}
	mask.Data[seed.Y*cols+seed.X] = 255
	cur := make([]float64, ch)
	for len(stack) > 0 {
		i := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		x := i % cols
		y := i / cols
		for _, o := range nb {
			nx, ny := x+o.dx, y+o.dy
			if nx < 0 || nx >= cols || ny < 0 || ny >= rows {
				continue
			}
			ni := ny*cols + nx
			if mask.Data[ni] != 0 {
				continue
			}
			segment2colorInto(img, nx, ny, cur)
			if segment2colorDist2(cur, mean) <= tol2 {
				mask.Data[ni] = 255
				stack = append(stack, ni)
				count++
				for c := 0; c < ch; c++ {
					mean[c] += (cur[c] - mean[c]) / count
				}
			}
		}
	}
	return mask
}

// SeededRegionGrow grows several regions simultaneously from labelled seeds,
// producing a full partition of the image. seeds is a [LabelMap] the size of
// img in which positive labels mark seed pixels and 0 marks unassigned pixels.
// Pixels are assigned in order of increasing colour difference from the region
// mean, so competing regions meet along colour boundaries (Adams & Bischof
// seeded region growing). Every pixel receives a label. conn selects
// connectivity.
//
// It panics if img is empty or seeds does not match img in size.
func SeededRegionGrow(img *cv.Mat, seeds *LabelMap, conn Connectivity) *LabelMap {
	segment2requireNonEmpty(img, "SeededRegionGrow")
	if seeds.Rows != img.Rows || seeds.Cols != img.Cols {
		panic("segment2: SeededRegionGrow seed size mismatch")
	}
	nb := segment2neighbors(conn)
	rows, cols, ch := img.Rows, img.Cols, img.Channels
	out := seeds.Clone()

	// Region running means.
	maxL := 0
	for _, l := range out.Labels {
		if l > maxL {
			maxL = l
		}
	}
	means := make([][]float64, maxL+1)
	counts := make([]float64, maxL+1)
	for l := range means {
		means[l] = make([]float64, ch)
	}
	for i, l := range out.Labels {
		if l <= 0 {
			continue
		}
		b := i * ch
		for c := 0; c < ch; c++ {
			means[l][c] += float64(img.Data[b+c])
		}
		counts[l]++
	}
	for l := 1; l <= maxL; l++ {
		if counts[l] > 0 {
			for c := 0; c < ch; c++ {
				means[l][c] /= counts[l]
			}
		}
	}

	pq := &segment2candHeap{}
	pushNeighbours := func(i int) {
		x := i % cols
		y := i / cols
		l := out.Labels[i]
		cur := make([]float64, ch)
		for _, o := range nb {
			nx, ny := x+o.dx, y+o.dy
			if nx < 0 || nx >= cols || ny < 0 || ny >= rows {
				continue
			}
			ni := ny*cols + nx
			if out.Labels[ni] != 0 {
				continue
			}
			segment2colorInto(img, nx, ny, cur)
			diff := segment2colorDist(cur, means[l])
			pq.push(segment2cand{index: ni, label: l, diff: diff})
		}
	}
	// Initialise queue.
	for i, l := range out.Labels {
		if l > 0 {
			pushNeighbours(i)
		}
	}

	cur := make([]float64, ch)
	for pq.Len() > 0 {
		c := pq.pop()
		if out.Labels[c.index] != 0 {
			continue
		}
		out.Labels[c.index] = c.label
		segment2colorInto(img, c.index%cols, c.index/cols, cur)
		counts[c.label]++
		for ci := 0; ci < ch; ci++ {
			means[c.label][ci] += (cur[ci] - means[c.label][ci]) / counts[c.label]
		}
		pushNeighbours(c.index)
	}
	out.NumLabels = maxL + 1
	return out
}
