package imgprocx

import cv "github.com/malcolmston/opencv"

// FloodFillOptions configures [FloodFill]. The zero value performs a 4-connected
// fixed-range fill with zero tolerance (only pixels exactly equal to the seed
// are filled) and no mask.
type FloodFillOptions struct {
	// Connectivity selects the neighbourhood: 8 for 8-connectivity, anything
	// else (including the zero value) for 4-connectivity.
	Connectivity int
	// FixedRange, when true, compares each candidate pixel against the fixed
	// seed value (OpenCV's fixed-range mode). When false the candidate is
	// compared against the neighbour it is reached from (floating-range mode),
	// so the fill follows gradual gradients.
	FixedRange bool
	// LoDiff and UpDiff are the maximal allowed downward and upward difference,
	// per channel, between a candidate pixel and the reference value.
	LoDiff, UpDiff float64
	// Mask, when non-nil, must be a single-channel image the same size as the
	// filled image. Pixels whose mask value is already non-zero act as barriers
	// that are never entered; every filled pixel has its mask set to MaskValue.
	Mask *cv.Mat
	// MaskValue is written into Mask for each filled pixel; a zero value is
	// treated as 1 so filled pixels are always marked non-zero.
	MaskValue uint8
	// MaskOnly, when true (and Mask is set), leaves the image unchanged and only
	// records the filled region in the mask (OpenCV's FLOODFILL_MASK_ONLY).
	MaskOnly bool
}

// FloodFill fills a connected region of img starting at seed, mirroring
// cv2.floodFill. A pixel joins the region when, for every channel, its value
// lies within [ref-LoDiff, ref+UpDiff] of a reference value — the fixed seed
// value in fixed-range mode or the already-accepted neighbour in floating-range
// mode (see [FloodFillOptions]). Accepted pixels are painted with newVal (unless
// MaskOnly is set) and, when a mask is supplied, marked in it.
//
// It returns the number of pixels filled and the bounding [cv.Rect] of the
// filled region (a zero-size rect when nothing is filled). The traversal is a
// deterministic breadth-first flood, so the outcome depends only on the inputs.
// It panics if seed lies outside img or if a supplied mask has the wrong size or
// is not single-channel.
func FloodFill(img *cv.Mat, seed cv.Point, newVal cv.Scalar, opts *FloodFillOptions) (area int, rect cv.Rect) {
	var o FloodFillOptions
	if opts != nil {
		o = *opts
	}
	if seed.X < 0 || seed.X >= img.Cols || seed.Y < 0 || seed.Y >= img.Rows {
		panic("imgprocx: FloodFill seed is out of bounds")
	}
	if o.Mask != nil {
		requireSingleChannel(o.Mask, "FloodFill mask")
		if o.Mask.Rows != img.Rows || o.Mask.Cols != img.Cols {
			panic("imgprocx: FloodFill mask must match the image size")
		}
	}
	maskVal := o.MaskValue
	if maskVal == 0 {
		maskVal = 1
	}
	rows, cols, ch := img.Rows, img.Cols, img.Channels

	// within reports whether the candidate at flat pixel index cand is within
	// tolerance of the reference pixel at flat index ref.
	within := func(refBase, candBase int) bool {
		for c := 0; c < ch; c++ {
			d := float64(img.Data[candBase+c]) - float64(img.Data[refBase+c])
			if d < -o.LoDiff || d > o.UpDiff {
				return false
			}
		}
		return true
	}

	visited := make([]bool, rows*cols)
	seedFlat := seed.Y*cols + seed.X
	seedBase := seedFlat * ch
	// A pixel already marked in the mask blocks the fill entirely.
	if o.Mask != nil && o.Mask.Data[seedFlat] != 0 {
		return 0, cv.Rect{}
	}

	queue := []int{seedFlat}
	visited[seedFlat] = true
	minX, minY, maxX, maxY := seed.X, seed.Y, seed.X, seed.Y

	neighbours := func(x, y int) [][2]int {
		n := [][2]int{{x - 1, y}, {x + 1, y}, {x, y - 1}, {x, y + 1}}
		if o.Connectivity == 8 {
			n = append(n, [2]int{x - 1, y - 1}, [2]int{x + 1, y - 1}, [2]int{x - 1, y + 1}, [2]int{x + 1, y + 1})
		}
		return n
	}

	for len(queue) > 0 {
		cur := queue[0]
		queue = queue[1:]
		cx := cur % cols
		cy := cur / cols
		area++
		if cx < minX {
			minX = cx
		}
		if cx > maxX {
			maxX = cx
		}
		if cy < minY {
			minY = cy
		}
		if cy > maxY {
			maxY = cy
		}
		refBase := seedBase
		if !o.FixedRange {
			refBase = cur * ch
		}
		for _, nb := range neighbours(cx, cy) {
			nx, ny := nb[0], nb[1]
			if nx < 0 || nx >= cols || ny < 0 || ny >= rows {
				continue
			}
			nFlat := ny*cols + nx
			if visited[nFlat] {
				continue
			}
			if o.Mask != nil && o.Mask.Data[nFlat] != 0 {
				continue
			}
			if !within(refBase, nFlat*ch) {
				continue
			}
			visited[nFlat] = true
			queue = append(queue, nFlat)
		}
	}

	// Second pass: commit the accepted pixels to the image and/or mask.
	for flat := 0; flat < rows*cols; flat++ {
		if !visited[flat] {
			continue
		}
		if o.Mask != nil {
			o.Mask.Data[flat] = maskVal
		}
		if o.Mask == nil || !o.MaskOnly {
			base := flat * ch
			for c := 0; c < ch; c++ {
				img.Data[base+c] = clampUint8(newVal[c])
			}
		}
	}
	rect = cv.Rect{X: minX, Y: minY, Width: maxX - minX + 1, Height: maxY - minY + 1}
	return area, rect
}
