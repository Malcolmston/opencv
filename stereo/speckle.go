package stereo

import cv "github.com/malcolmston/opencv"

// FilterSpecklesDisparity removes small speckles from a disparity map in place,
// mirroring OpenCV's cv::filterSpeckles. It groups connected pixels (4-connected)
// whose disparities differ by at most maxDiff into blobs; any blob smaller than
// maxSpeckleSize pixels is overwritten with newVal (usually [InvalidDisparity]).
// Pixels already holding [InvalidDisparity] are treated as background and skipped.
//
// disp is modified and also returned for convenience. It panics if disp is nil,
// empty, or not single-channel. Non-positive maxSpeckleSize or maxDiff leaves the
// map unchanged.
func FilterSpecklesDisparity(disp *cv.Mat, newVal uint8, maxSpeckleSize, maxDiff int) *cv.Mat {
	if disp == nil || disp.Empty() {
		panic("stereo: FilterSpecklesDisparity given a nil or empty disparity map")
	}
	if disp.Channels != 1 {
		panic("stereo: FilterSpecklesDisparity requires a single-channel disparity map")
	}
	if maxSpeckleSize <= 0 || maxDiff <= 0 {
		return disp
	}

	rows, cols := disp.Rows, disp.Cols
	n := rows * cols
	labels := make([]int, n) // 0 = unvisited
	label := 0
	queue := make([]int, 0, n)

	for start := 0; start < n; start++ {
		if labels[start] != 0 || disp.Data[start] == InvalidDisparity {
			continue
		}
		label++
		// Flood fill the blob connected to `start`.
		queue = queue[:0]
		queue = append(queue, start)
		labels[start] = label
		count := 0
		region := make([]int, 0, 16)
		region = append(region, start)
		for qi := 0; qi < len(queue); qi++ {
			p := queue[qi]
			count++
			py, px := p/cols, p%cols
			cur := int(disp.Data[p])
			// 4-connected neighbours.
			if px > 0 {
				visitSpeckleNeighbour(disp, labels, &queue, &region, p-1, cur, label, maxDiff)
			}
			if px < cols-1 {
				visitSpeckleNeighbour(disp, labels, &queue, &region, p+1, cur, label, maxDiff)
			}
			if py > 0 {
				visitSpeckleNeighbour(disp, labels, &queue, &region, p-cols, cur, label, maxDiff)
			}
			if py < rows-1 {
				visitSpeckleNeighbour(disp, labels, &queue, &region, p+cols, cur, label, maxDiff)
			}
		}
		if count < maxSpeckleSize {
			for _, p := range region {
				disp.Data[p] = newVal
			}
		}
	}
	return disp
}

// visitSpeckleNeighbour enqueues neighbour q into the current blob when it is a
// valid, unvisited pixel whose disparity is within maxDiff of cur.
func visitSpeckleNeighbour(disp *cv.Mat, labels []int, queue, region *[]int, q, cur, label, maxDiff int) {
	if labels[q] != 0 || disp.Data[q] == InvalidDisparity {
		return
	}
	diff := int(disp.Data[q]) - cur
	if diff < 0 {
		diff = -diff
	}
	if diff > maxDiff {
		return
	}
	labels[q] = label
	*queue = append(*queue, q)
	*region = append(*region, q)
}
