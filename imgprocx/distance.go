package imgprocx

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// DistanceTransformWithLabels computes, for every pixel of a binary
// single-channel image, both the approximate Euclidean distance to the nearest
// zero (background) pixel and a label identifying that nearest background pixel,
// mirroring cv2.distanceTransformWithLabels. src is treated as binary: zero
// samples are background, non-zero samples are foreground.
//
// It returns the distances as a [cv.FloatMat] and a labels grid the same size
// as src. Each label is the flat index (row·Cols + col) of the nearest
// background pixel; every background pixel is labelled with its own index, so
// the label field is the discrete Voronoi diagram of the background pixels. It
// panics if src is not single-channel.
//
// Distances come from the two-pass 3×3 chamfer with weights 1 (orthogonal) and
// √2 (diagonal); the same relaxation propagates each pixel's label from the
// neighbour that produced its shortest distance, so distances and labels are
// always consistent.
func DistanceTransformWithLabels(src *cv.Mat) (dist *cv.FloatMat, labels [][]int) {
	requireSingleChannel(src, "DistanceTransformWithLabels")
	rows, cols := src.Rows, src.Cols
	const (
		orth = 1.0
		diag = math.Sqrt2
	)
	dist = cv.NewFloatMat(rows, cols)
	labels = make([][]int, rows)
	big := float64(rows+cols) * 2
	for y := 0; y < rows; y++ {
		labels[y] = make([]int, cols)
		for x := 0; x < cols; x++ {
			idx := y*cols + x
			if src.Data[idx] == 0 {
				dist.Data[idx] = 0
				labels[y][x] = idx
			} else {
				dist.Data[idx] = big
				labels[y][x] = -1
			}
		}
	}
	at := func(y, x int) float64 { return dist.Data[y*cols+x] }
	// relax updates (y,x) from neighbour (ny,nx) with edge weight w.
	relax := func(y, x, ny, nx int, w float64) {
		if at(ny, nx)+w < at(y, x) {
			dist.Data[y*cols+x] = at(ny, nx) + w
			labels[y][x] = labels[ny][nx]
		}
	}
	// Forward pass: top-left neighbours.
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			if src.Data[y*cols+x] == 0 {
				continue
			}
			if x > 0 {
				relax(y, x, y, x-1, orth)
			}
			if y > 0 {
				relax(y, x, y-1, x, orth)
			}
			if y > 0 && x > 0 {
				relax(y, x, y-1, x-1, diag)
			}
			if y > 0 && x < cols-1 {
				relax(y, x, y-1, x+1, diag)
			}
		}
	}
	// Backward pass: bottom-right neighbours.
	for y := rows - 1; y >= 0; y-- {
		for x := cols - 1; x >= 0; x-- {
			if src.Data[y*cols+x] == 0 {
				continue
			}
			if x < cols-1 {
				relax(y, x, y, x+1, orth)
			}
			if y < rows-1 {
				relax(y, x, y+1, x, orth)
			}
			if y < rows-1 && x < cols-1 {
				relax(y, x, y+1, x+1, diag)
			}
			if y < rows-1 && x > 0 {
				relax(y, x, y+1, x-1, diag)
			}
		}
	}
	return dist, labels
}
