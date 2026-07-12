package segmentation

import (
	cv "github.com/malcolmston/opencv"
)

// MeanShiftSegmentation turns the edge-preserving smoothing of
// [MeanShiftFiltering] into a full region labelling, the segmentation stage that
// follows cv2.pyrMeanShiftFiltering in a classic mean-shift pipeline. It filters
// img in the joint spatial-range domain, groups spatially adjacent pixels whose
// filtered colours lie within sr of one another into connected regions, and
// merges regions smaller than minSize into their most similar neighbour.
//
// The result is a [LabelMap]: neighbouring pixels that collapsed onto the same
// colour mode share a label, so flat areas become single regions while genuine
// colour boundaries separate them. sp and sr are the spatial and range radii of
// the underlying mean shift; minSize suppresses tiny speckle regions (pass 0 or
// 1 to keep them all).
//
// img must be three-channel. It panics if img is empty or not three-channel.
func MeanShiftSegmentation(img *cv.Mat, sp, sr float64, minSize int) *LabelMap {
	if img.Empty() {
		panic("segmentation: MeanShiftSegmentation on empty image")
	}
	if img.Channels != 3 {
		panic("segmentation: MeanShiftSegmentation requires a 3-channel image")
	}
	filtered := MeanShiftFiltering(img, sp, sr)
	rows, cols := filtered.Rows, filtered.Cols
	n := rows * cols

	// Union adjacent pixels whose filtered colours are within sr.
	uf := newUnionFind(n)
	colorAt := func(idx int) [3]float64 {
		b := idx * 3
		return [3]float64{float64(filtered.Data[b]), float64(filtered.Data[b+1]), float64(filtered.Data[b+2])}
	}
	within := func(a, b int) bool {
		ca, cb := colorAt(a), colorAt(b)
		d := (ca[0]-cb[0])*(ca[0]-cb[0]) + (ca[1]-cb[1])*(ca[1]-cb[1]) + (ca[2]-cb[2])*(ca[2]-cb[2])
		return d <= sr*sr
	}
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			idx := y*cols + x
			if x+1 < cols && within(idx, idx+1) {
				uf.union(idx, idx+1)
			}
			if y+1 < rows && within(idx, idx+cols) {
				uf.union(idx, idx+cols)
			}
		}
	}

	raw := make([]int, n)
	for i := 0; i < n; i++ {
		raw[i] = uf.find(i)
	}
	labels, count := relabelConsecutive(raw)
	lm := &LabelMap{Rows: rows, Cols: cols, Count: count, Labels: labels}

	if minSize > 1 && count > 1 {
		rag := BuildRAG(lm, filtered)
		lm = rag.MergeBySize(minSize)
	}
	return lm
}
