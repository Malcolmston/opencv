package features3

import cv "github.com/malcolmston/opencv"

// KeypointNMS applies greedy radial non-maximum suppression to a keypoint list.
// Keypoints are considered in descending response order; each accepted keypoint
// suppresses every remaining keypoint whose centre lies within minDistance
// pixels. When minDistance <= 0 the input is returned sorted by response with no
// suppression. The input slice is not modified.
func KeypointNMS(kps []KeyPoint, minDistance float64) []KeyPoint {
	sorted := make([]KeyPoint, len(kps))
	copy(sorted, kps)
	SortKeyPointsByResponse(sorted)
	if minDistance <= 0 {
		return sorted
	}
	minD2 := minDistance * minDistance
	var kept []KeyPoint
	for _, k := range sorted {
		ok := true
		for _, e := range kept {
			dx := k.Pt.X - e.Pt.X
			dy := k.Pt.Y - e.Pt.Y
			if dx*dx+dy*dy < minD2 {
				ok = false
				break
			}
		}
		if ok {
			kept = append(kept, k)
		}
	}
	return kept
}

// NonMaxSuppressionFloat scans a response image and returns the local maxima as
// keypoints. A pixel is a maximum when it is strictly greater than every other
// pixel within a (2*halfWindow+1) square window and its value is at least
// threshold. The window is clamped at image borders. Results are sorted by
// descending response.
func NonMaxSuppressionFloat(resp *cv.FloatMat, halfWindow int, threshold float64) []KeyPoint {
	if halfWindow < 1 {
		halfWindow = 1
	}
	rows, cols := resp.Rows, resp.Cols
	var kps []KeyPoint
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			v := resp.Data[y*cols+x]
			if v < threshold {
				continue
			}
			isMax := true
			for dy := -halfWindow; dy <= halfWindow && isMax; dy++ {
				ny := y + dy
				if ny < 0 || ny >= rows {
					continue
				}
				for dx := -halfWindow; dx <= halfWindow; dx++ {
					nx := x + dx
					if nx < 0 || nx >= cols || (dx == 0 && dy == 0) {
						continue
					}
					if resp.Data[ny*cols+nx] > v {
						isMax = false
						break
					}
				}
			}
			if isMax {
				kps = append(kps, NewKeyPoint(float64(x), float64(y), v))
			}
		}
	}
	SortKeyPointsByResponse(kps)
	return kps
}

// LocalMaxima returns the integer coordinates of the strict local maxima of a
// response image within a (2*halfWindow+1) window whose value is at least
// threshold. It is the point-only counterpart of [NonMaxSuppressionFloat].
func LocalMaxima(resp *cv.FloatMat, halfWindow int, threshold float64) []cv.Point {
	kps := NonMaxSuppressionFloat(resp, halfWindow, threshold)
	return ConvertKeyPointsToPoints(kps)
}
