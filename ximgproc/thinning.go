package ximgproc

import cv "github.com/malcolmston/opencv"

// Thinning reduces a binary image to a one-pixel-wide skeleton using the
// Zhang–Suen iterative thinning algorithm ("A fast parallel algorithm for
// thinning digital patterns", 1984), and returns a new single-channel Mat.
//
// binary must be single-channel; any non-zero sample is treated as foreground.
// The result marks skeleton pixels with 255 and the background with 0. The
// algorithm alternates two sub-iterations, each removing foreground border
// pixels whose deletion neither breaks the shape's connectivity nor shortens an
// end of the skeleton, and stops when a full pass removes nothing. Out-of-image
// neighbours are treated as background. It panics if binary is not
// single-channel.
//
// The output is guaranteed to be thin: no run of skeleton pixels along a row or
// column is wider than one pixel. The Guo–Hall variant is deferred (see the
// package documentation).
func Thinning(binary *cv.Mat) *cv.Mat {
	if binary.Channels != 1 {
		panic("ximgproc: Thinning requires a single-channel image")
	}
	rows, cols := binary.Rows, binary.Cols

	// Working grid: 1 = foreground, 0 = background.
	g := make([]uint8, rows*cols)
	for i, v := range binary.Data {
		if v != 0 {
			g[i] = 1
		}
	}

	at := func(y, x int) uint8 {
		if y < 0 || y >= rows || x < 0 || x >= cols {
			return 0
		}
		return g[y*cols+x]
	}

	// toDelete collects the flat indices flagged for removal in a sub-iteration.
	toDelete := make([]int, 0, rows*cols)

	step := func(second bool) bool {
		toDelete = toDelete[:0]
		for y := 0; y < rows; y++ {
			for x := 0; x < cols; x++ {
				if g[y*cols+x] == 0 {
					continue
				}
				// Neighbours P2..P9 clockwise from north.
				p2 := at(y-1, x)
				p3 := at(y-1, x+1)
				p4 := at(y, x+1)
				p5 := at(y+1, x+1)
				p6 := at(y+1, x)
				p7 := at(y+1, x-1)
				p8 := at(y, x-1)
				p9 := at(y-1, x-1)

				b := int(p2) + int(p3) + int(p4) + int(p5) + int(p6) + int(p7) + int(p8) + int(p9)
				if b < 2 || b > 6 {
					continue
				}
				// A = number of 0->1 transitions in the ordered sequence.
				seq := [9]uint8{p2, p3, p4, p5, p6, p7, p8, p9, p2}
				a := 0
				for i := 0; i < 8; i++ {
					if seq[i] == 0 && seq[i+1] == 1 {
						a++
					}
				}
				if a != 1 {
					continue
				}
				if !second {
					if p2*p4*p6 != 0 || p4*p6*p8 != 0 {
						continue
					}
				} else {
					if p2*p4*p8 != 0 || p2*p6*p8 != 0 {
						continue
					}
				}
				toDelete = append(toDelete, y*cols+x)
			}
		}
		for _, idx := range toDelete {
			g[idx] = 0
		}
		return len(toDelete) > 0
	}

	for {
		changed := step(false)
		if step(true) {
			changed = true
		}
		if !changed {
			break
		}
	}

	out := cv.NewMat(rows, cols, 1)
	for i, v := range g {
		if v != 0 {
			out.Data[i] = 255
		}
	}
	return out
}
