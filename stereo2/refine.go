package stereo2

import (
	"fmt"
	"math"
	"sort"
)

// LeftRightCheck enforces left/right consistency between a left-referenced
// disparity map and a right-referenced one. For each valid left pixel (y,x) with
// disparity dL the matching right pixel lies at column x-round(dL); the pixel is
// kept only if that right pixel is valid and its disparity agrees with dL to
// within maxDiff. Inconsistent pixels — typically occlusions and mismatches —
// are set to [InvalidDisparity]. The input maps are not modified; a new map is
// returned. It panics if the maps differ in size.
func LeftRightCheck(left, right *DisparityMap, maxDiff float32) *DisparityMap {
	if left.Rows != right.Rows || left.Cols != right.Cols {
		panic(fmt.Sprintf("stereo2: LeftRightCheck size mismatch %dx%d vs %dx%d", left.Rows, left.Cols, right.Rows, right.Cols))
	}
	out := left.Clone()
	cols := left.Cols
	for y := 0; y < left.Rows; y++ {
		for x := 0; x < cols; x++ {
			dL := left.Data[y*cols+x]
			if dL < 0 || math.IsNaN(float64(dL)) {
				continue
			}
			xr := x - int(math.Round(float64(dL)))
			if xr < 0 || xr >= cols {
				out.Data[y*cols+x] = InvalidDisparity
				continue
			}
			dR := right.Data[y*cols+xr]
			if dR < 0 || math.IsNaN(float64(dR)) || absF32(dL-dR) > maxDiff {
				out.Data[y*cols+x] = InvalidDisparity
			}
		}
	}
	return out
}

// RefineSubpixelParabola extracts a sub-pixel disparity map from an aggregated
// cost volume. For each pixel the integer winner d* is found, and the true
// minimum is estimated by fitting a parabola through the costs at d*-1, d* and
// d*+1:
//
//	offset = (C(d*-1) - C(d*+1)) / (2*(C(d*-1) - 2*C(d*) + C(d*+1)))
//
// The refined disparity is MinDisparity + d* + offset. Winners at the first or
// last candidate, or with a non-positive parabola curvature, keep the integer
// value. uniquenessRatio, when positive, applies the uniqueness test.
func RefineSubpixelParabola(vol *CostVolume, uniquenessRatio int) *DisparityMap {
	out := NewDisparityMap(vol.Rows, vol.Cols)
	D := vol.Disparities
	for y := 0; y < vol.Rows; y++ {
		for x := 0; x < vol.Cols; x++ {
			best, bestC := vol.ArgMin(y, x)
			if bestC >= stereo2invalidThreshold {
				continue
			}
			if uniquenessRatio > 0 && !stereo2unique(vol, y, x, best, bestC, uniquenessRatio) {
				continue
			}
			d := float32(vol.MinDisparity + best)
			if best > 0 && best < D-1 {
				base := (y*vol.Cols + x) * D
				cm := vol.Data[base+best-1]
				c0 := vol.Data[base+best]
				cp := vol.Data[base+best+1]
				denom := cm - 2*c0 + cp
				if denom > 0 {
					off := (cm - cp) / (2 * denom)
					if off > -1 && off < 1 {
						d += off
					}
				}
			}
			out.Data[y*vol.Cols+x] = d
		}
	}
	return out
}

// MedianFilterDisparity replaces each valid disparity by the median of the valid
// disparities in its (2*radius+1) square neighbourhood, suppressing isolated
// outliers while preserving edges. Invalid pixels stay invalid and are ignored
// when computing a neighbour's median. It panics if radius is not positive.
func MedianFilterDisparity(disp *DisparityMap, radius int) *DisparityMap {
	if radius <= 0 {
		panic(fmt.Sprintf("stereo2: MedianFilterDisparity requires radius > 0, got %d", radius))
	}
	out := disp.Clone()
	cols := disp.Cols
	buf := make([]float32, 0, (2*radius+1)*(2*radius+1))
	for y := 0; y < disp.Rows; y++ {
		for x := 0; x < cols; x++ {
			v := disp.Data[y*cols+x]
			if v < 0 || math.IsNaN(float64(v)) {
				continue
			}
			buf = buf[:0]
			for j := -radius; j <= radius; j++ {
				yy := y + j
				if yy < 0 || yy >= disp.Rows {
					continue
				}
				for i := -radius; i <= radius; i++ {
					xx := x + i
					if xx < 0 || xx >= cols {
						continue
					}
					nv := disp.Data[yy*cols+xx]
					if nv >= 0 && !math.IsNaN(float64(nv)) {
						buf = append(buf, nv)
					}
				}
			}
			sort.Slice(buf, func(a, b int) bool { return buf[a] < buf[b] })
			out.Data[y*cols+x] = buf[len(buf)/2]
		}
	}
	return out
}

// SpeckleFilter removes small isolated disparity blobs ("speckles"). Pixels are
// grouped into connected components where two 4-adjacent valid pixels join iff
// their disparities differ by at most maxDiff; every component smaller than
// maxSpeckleSize pixels is set to [InvalidDisparity]. This is the standard
// post-filter for cleaning mismatched regions. It panics if maxSpeckleSize is
// not positive.
func SpeckleFilter(disp *DisparityMap, maxSpeckleSize int, maxDiff float32) *DisparityMap {
	if maxSpeckleSize <= 0 {
		panic(fmt.Sprintf("stereo2: SpeckleFilter requires maxSpeckleSize > 0, got %d", maxSpeckleSize))
	}
	out := disp.Clone()
	rows, cols := disp.Rows, disp.Cols
	labels := make([]int, rows*cols) // 0 = unlabelled
	label := 0
	stack := make([]int, 0, 256)
	for start := 0; start < rows*cols; start++ {
		v := disp.Data[start]
		if v < 0 || math.IsNaN(float64(v)) || labels[start] != 0 {
			continue
		}
		label++
		labels[start] = label
		stack = stack[:0]
		stack = append(stack, start)
		component := stack[:0:0] // separate slice recording members
		component = append(component, start)
		count := 1
		for len(stack) > 0 {
			cur := stack[len(stack)-1]
			stack = stack[:len(stack)-1]
			cy, cx := cur/cols, cur%cols
			cv := disp.Data[cur]
			neigh := [4][2]int{{cy - 1, cx}, {cy + 1, cx}, {cy, cx - 1}, {cy, cx + 1}}
			for _, n := range neigh {
				ny, nx := n[0], n[1]
				if ny < 0 || ny >= rows || nx < 0 || nx >= cols {
					continue
				}
				idx := ny*cols + nx
				if labels[idx] != 0 {
					continue
				}
				nv := disp.Data[idx]
				if nv < 0 || math.IsNaN(float64(nv)) {
					continue
				}
				if absF32(nv-cv) > maxDiff {
					continue
				}
				labels[idx] = label
				stack = append(stack, idx)
				component = append(component, idx)
				count++
			}
		}
		if count < maxSpeckleSize {
			for _, idx := range component {
				out.Data[idx] = InvalidDisparity
			}
		}
	}
	return out
}

// FillInvalidHorizontal fills each invalid pixel by propagating disparities
// horizontally: it takes the nearest valid disparity to the left and to the
// right on the same row and assigns the smaller of the two (the more distant,
// background surface — the standard occlusion-filling heuristic). Pixels with a
// valid neighbour on only one side take that value; rows with no valid pixel are
// left untouched. The input is not modified; a new map is returned.
func FillInvalidHorizontal(disp *DisparityMap) *DisparityMap {
	out := disp.Clone()
	cols := disp.Cols
	for y := 0; y < disp.Rows; y++ {
		row := y * cols
		for x := 0; x < cols; x++ {
			v := disp.Data[row+x]
			if v >= 0 && !math.IsNaN(float64(v)) {
				continue
			}
			left := InvalidDisparity
			for xl := x - 1; xl >= 0; xl-- {
				lv := disp.Data[row+xl]
				if lv >= 0 && !math.IsNaN(float64(lv)) {
					left = lv
					break
				}
			}
			right := InvalidDisparity
			for xr := x + 1; xr < cols; xr++ {
				rv := disp.Data[row+xr]
				if rv >= 0 && !math.IsNaN(float64(rv)) {
					right = rv
					break
				}
			}
			switch {
			case left >= 0 && right >= 0:
				if left < right {
					out.Data[row+x] = left
				} else {
					out.Data[row+x] = right
				}
			case left >= 0:
				out.Data[row+x] = left
			case right >= 0:
				out.Data[row+x] = right
			}
		}
	}
	return out
}

// absF32 returns the absolute value of v.
func absF32(v float32) float32 {
	if v < 0 {
		return -v
	}
	return v
}
