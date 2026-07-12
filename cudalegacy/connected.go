package cudalegacy

import (
	cv "github.com/malcolmston/opencv"
)

// Connectivity edge bits used by [ConnectivityMask] and [LabelComponents]. Each
// pixel of a connectivity mask is a uint8 whose set bits record which neighbours
// it is joined to. The four axial bits are always meaningful; the four diagonal
// bits are only set when 8-connectivity is requested.
const (
	MaskConnectEast  uint8 = 1 << 0 // neighbour at (x+1, y)
	MaskConnectSouth uint8 = 1 << 1 // neighbour at (x, y+1)
	MaskConnectWest  uint8 = 1 << 2 // neighbour at (x-1, y)
	MaskConnectNorth uint8 = 1 << 3 // neighbour at (x, y-1)
	MaskConnectSE    uint8 = 1 << 4 // neighbour at (x+1, y+1)
	MaskConnectSW    uint8 = 1 << 5 // neighbour at (x-1, y+1)
	MaskConnectNW    uint8 = 1 << 6 // neighbour at (x-1, y-1)
	MaskConnectNE    uint8 = 1 << 7 // neighbour at (x+1, y-1)
)

// ConnectivityMask is a CPU-backed mirror of OpenCV's
// cv::cuda::connectivityMask. It examines every pixel of image and produces a
// single-channel mask the same size in which each pixel's bits (see the
// MaskConnect* constants) record which of its neighbours are "similar enough" to
// be joined. Two neighbouring pixels are connected when the maximum absolute
// per-channel difference d between them satisfies loThreshold <= d <=
// hiThreshold.
//
// conn selects whether only the four axial neighbours ([cv.Connectivity4]) or
// also the four diagonals ([cv.Connectivity8]) are considered. The resulting
// mask is symmetric: if pixel A records a connection to B, then B records the
// reciprocal connection to A. It panics on a nil or empty image or an invalid
// connectivity. The stream is a no-op.
func ConnectivityMask(image *GpuMat, loThreshold, hiThreshold float64, conn cv.Connectivity, stream *Stream) *GpuMat {
	_ = stream
	m := requireMat(image, "ConnectivityMask")
	if conn != cv.Connectivity4 && conn != cv.Connectivity8 {
		panic("cudalegacy: ConnectivityMask connectivity must be 4 or 8")
	}
	rows, cols, ch := m.Rows, m.Cols, m.Channels
	out := cv.NewMat(rows, cols, 1)

	connected := func(y0, x0, y1, x1 int) bool {
		i0 := (y0*cols + x0) * ch
		i1 := (y1*cols + x1) * ch
		maxDiff := 0.0
		for c := 0; c < ch; c++ {
			d := float64(m.Data[i0+c]) - float64(m.Data[i1+c])
			if d < 0 {
				d = -d
			}
			if d > maxDiff {
				maxDiff = d
			}
		}
		return maxDiff >= loThreshold && maxDiff <= hiThreshold
	}

	set := func(y, x int, bit uint8) {
		out.Data[y*cols+x] |= bit
	}

	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			if x+1 < cols && connected(y, x, y, x+1) {
				set(y, x, MaskConnectEast)
				set(y, x+1, MaskConnectWest)
			}
			if y+1 < rows && connected(y, x, y+1, x) {
				set(y, x, MaskConnectSouth)
				set(y+1, x, MaskConnectNorth)
			}
			if conn == cv.Connectivity8 {
				if y+1 < rows && x+1 < cols && connected(y, x, y+1, x+1) {
					set(y, x, MaskConnectSE)
					set(y+1, x+1, MaskConnectNW)
				}
				if y+1 < rows && x-1 >= 0 && connected(y, x, y+1, x-1) {
					set(y, x, MaskConnectSW)
					set(y+1, x-1, MaskConnectNE)
				}
			}
		}
	}
	return GpuMatFromMat(out)
}

// LabelComponents is a CPU-backed mirror of OpenCV's cv::cuda::labelComponents.
// Given a connectivity mask produced by [ConnectivityMask], it groups pixels
// into connected components following only the edges the mask records, and
// returns a label per pixel in row-major order together with the number of
// components.
//
// Labels run 0..count-1 in raster order of first appearance. Because OpenCV's
// label image is CV_32S and the 8-bit root [cv.Mat] cannot represent 32-bit
// labels, the labels are returned as an []int32 slice rather than a GpuMat; use
// [RenderLabels] to obtain a viewable GpuMat. It panics on a nil or empty mask.
// The stream is a no-op.
func LabelComponents(mask *GpuMat, stream *Stream) (labels []int32, count int) {
	_ = stream
	m := requireMat(mask, "LabelComponents")
	rows, cols := m.Rows, m.Cols
	n := rows * cols

	parent := make([]int, n)
	for i := range parent {
		parent[i] = i
	}
	find := func(x int) int {
		for parent[x] != x {
			parent[x] = parent[parent[x]]
			x = parent[x]
		}
		return x
	}
	union := func(a, b int) {
		ra, rb := find(a), find(b)
		if ra == rb {
			return
		}
		if ra < rb {
			parent[rb] = ra
		} else {
			parent[ra] = rb
		}
	}

	// Only forward edges (East, South and the two downward diagonals) need to be
	// walked; the mask is symmetric so reverse edges are covered.
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			idx := y*cols + x
			bits := m.Data[idx]
			if bits&MaskConnectEast != 0 && x+1 < cols {
				union(idx, idx+1)
			}
			if bits&MaskConnectSouth != 0 && y+1 < rows {
				union(idx, idx+cols)
			}
			if bits&MaskConnectSE != 0 && y+1 < rows && x+1 < cols {
				union(idx, idx+cols+1)
			}
			if bits&MaskConnectSW != 0 && y+1 < rows && x-1 >= 0 {
				union(idx, idx+cols-1)
			}
		}
	}

	labels = make([]int32, n)
	remap := make(map[int]int32)
	for i := 0; i < n; i++ {
		root := find(i)
		lbl, ok := remap[root]
		if !ok {
			lbl = int32(count)
			remap[root] = lbl
			count++
		}
		labels[i] = lbl
	}
	return labels, count
}

// RenderLabels turns a label slice from [LabelComponents] into a viewable
// single-channel [GpuMat] of the given size, mapping each label to a distinct
// grey value (label modulo 256, with label 0 kept black). It panics if
// len(labels) != rows*cols or a dimension is non-positive.
func RenderLabels(labels []int32, rows, cols int) *GpuMat {
	if rows <= 0 || cols <= 0 {
		panic("cudalegacy: RenderLabels requires positive dimensions")
	}
	if len(labels) != rows*cols {
		panic("cudalegacy: RenderLabels label count does not match dimensions")
	}
	out := cv.NewMat(rows, cols, 1)
	for i, l := range labels {
		if l == 0 {
			continue
		}
		out.Data[i] = uint8(l % 256)
	}
	return GpuMatFromMat(out)
}
