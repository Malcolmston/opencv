package morph2

import cv "github.com/malcolmston/opencv"

// toBits returns a foreground bit grid (1 = foreground, 0 = background) from a
// single-channel image, treating any non-zero sample as foreground.
func toBits(src *cv.Mat) ([]uint8, int, int) {
	rows, cols := src.Rows, src.Cols
	b := make([]uint8, rows*cols)
	for i, v := range src.Data {
		if v != 0 {
			b[i] = 1
		}
	}
	return b, rows, cols
}

// bitsToMat converts a foreground bit grid back to a binary image (0/255).
func bitsToMat(b []uint8, rows, cols int) *cv.Mat {
	out := cv.NewMat(rows, cols, 1)
	for i, v := range b {
		if v != 0 {
			out.Data[i] = 255
		}
	}
	return out
}

// eightNeighbourhood returns p2..p9 (N, NE, E, SE, S, SW, W, NW) of pixel
// (y, x); out-of-image neighbours are 0.
func eightNeighbourhood(b []uint8, y, x, rows, cols int) [8]uint8 {
	get := func(yy, xx int) uint8 {
		if yy < 0 || yy >= rows || xx < 0 || xx >= cols {
			return 0
		}
		return b[yy*cols+xx]
	}
	return [8]uint8{
		get(y-1, x),   // p2 N
		get(y-1, x+1), // p3 NE
		get(y, x+1),   // p4 E
		get(y+1, x+1), // p5 SE
		get(y+1, x),   // p6 S
		get(y+1, x-1), // p7 SW
		get(y, x-1),   // p8 W
		get(y-1, x-1), // p9 NW
	}
}

// transitions counts 0->1 transitions in the ordered cyclic sequence
// p2,p3,...,p9,p2.
func transitions(n [8]uint8) int {
	c := 0
	for i := 0; i < 8; i++ {
		if n[i] == 0 && n[(i+1)%8] == 1 {
			c++
		}
	}
	return c
}

// countNeighbours returns the number of foreground pixels among the eight
// neighbours.
func countNeighbours(n [8]uint8) int {
	c := 0
	for _, v := range n {
		c += int(v)
	}
	return c
}

// ZhangSuenThinning reduces the foreground of a binary image to a
// one-pixel-wide skeleton using the Zhang-Suen (1984) parallel thinning
// algorithm. The result is a binary image (0/255) that preserves the topology
// and approximate medial axis of the original shapes. It is deterministic and
// panics on multi-channel input.
func ZhangSuenThinning(src *cv.Mat) *cv.Mat {
	requireGray(src)
	b, rows, cols := toBits(src)
	changed := true
	marks := make([]int, 0, len(b))
	step := func(second bool) bool {
		marks = marks[:0]
		for y := 0; y < rows; y++ {
			for x := 0; x < cols; x++ {
				if b[y*cols+x] == 0 {
					continue
				}
				n := eightNeighbourhood(b, y, x, rows, cols)
				bc := countNeighbours(n)
				if bc < 2 || bc > 6 {
					continue
				}
				if transitions(n) != 1 {
					continue
				}
				p2, p4, p6, p8 := n[0], n[2], n[4], n[6]
				if !second {
					if p2*p4*p6 != 0 || p4*p6*p8 != 0 {
						continue
					}
				} else {
					if p2*p4*p8 != 0 || p2*p6*p8 != 0 {
						continue
					}
				}
				marks = append(marks, y*cols+x)
			}
		}
		for _, p := range marks {
			b[p] = 0
		}
		return len(marks) > 0
	}
	for changed {
		c1 := step(false)
		c2 := step(true)
		changed = c1 || c2
	}
	return bitsToMat(b, rows, cols)
}

// GuoHallThinning reduces the foreground of a binary image to a
// one-pixel-wide skeleton using the Guo-Hall (1989) parallel thinning
// algorithm, which tends to produce smoother, better-connected skeletons than
// Zhang-Suen. It is deterministic and panics on multi-channel input.
func GuoHallThinning(src *cv.Mat) *cv.Mat {
	requireGray(src)
	b, rows, cols := toBits(src)
	marks := make([]int, 0, len(b))
	step := func(second bool) bool {
		marks = marks[:0]
		for y := 0; y < rows; y++ {
			for x := 0; x < cols; x++ {
				if b[y*cols+x] == 0 {
					continue
				}
				n := eightNeighbourhood(b, y, x, rows, cols)
				// n = p2..p9 = N,NE,E,SE,S,SW,W,NW.
				p2, p3, p4, p5 := n[0], n[1], n[2], n[3]
				p6, p7, p8, p9 := n[4], n[5], n[6], n[7]
				// C: number of distinct 8-connected components of ones in a
				// clockwise ordering starting at p2, per Guo-Hall.
				c := b2i(p2 == 0 && (p3 == 1 || p4 == 1)) +
					b2i(p4 == 0 && (p5 == 1 || p6 == 1)) +
					b2i(p6 == 0 && (p7 == 1 || p8 == 1)) +
					b2i(p8 == 0 && (p9 == 1 || p2 == 1))
				if c != 1 {
					continue
				}
				n1 := b2i(p9 == 1 || p2 == 1) + b2i(p3 == 1 || p4 == 1) +
					b2i(p5 == 1 || p6 == 1) + b2i(p7 == 1 || p8 == 1)
				n2 := b2i(p2 == 1 || p3 == 1) + b2i(p4 == 1 || p5 == 1) +
					b2i(p6 == 1 || p7 == 1) + b2i(p8 == 1 || p9 == 1)
				nm := n1
				if n2 < nm {
					nm = n2
				}
				if nm < 2 || nm > 3 {
					continue
				}
				var m uint8
				if !second {
					m = (p6 | p7 | b2u8(p9 == 0)) & p8
				} else {
					m = (p2 | p3 | b2u8(p5 == 0)) & p4
				}
				if m == 0 {
					marks = append(marks, y*cols+x)
				}
			}
		}
		for _, p := range marks {
			b[p] = 0
		}
		return len(marks) > 0
	}
	changed := true
	for changed {
		c1 := step(false)
		c2 := step(true)
		changed = c1 || c2
	}
	return bitsToMat(b, rows, cols)
}

func b2i(v bool) int {
	if v {
		return 1
	}
	return 0
}

func b2u8(v bool) uint8 {
	if v {
		return 1
	}
	return 0
}

// Skeleton computes the morphological skeleton of a binary image by
// Lantuejoul's method: the union over successive erosions f⊖nB of the residual
// f⊖nB minus its opening by B. The element e is the structuring element B
// (typically a small cross or square). The skeleton is generally not connected;
// for a connected medial axis use [ZhangSuenThinning]. It panics on
// multi-channel input.
func Skeleton(src *cv.Mat, e *Element) *cv.Mat {
	requireGray(src)
	cur := Binarize(src, 0)
	skel := newLike(src)
	offs := e.offsets()
	for CountForeground(cur) > 0 {
		opened := flatMorph(binaryMin(cur, offs), offs, true) // open = dilate(erode)
		eroded := binaryMin(cur, offs)
		residual := Subtract(cur, opened)
		for i := range skel.Data {
			skel.Data[i] = maxU8(skel.Data[i], residual.Data[i])
		}
		cur = eroded
	}
	return skel
}

// binaryMin is an erosion that keeps 0/255 binary output using the flat min.
func binaryMin(src *cv.Mat, offs [][2]int) *cv.Mat {
	er := binaryErode(src, offs)
	return er
}

// Prune removes spur pixels (short branches) from a one-pixel-wide skeleton by
// iteratively deleting endpoints — foreground pixels with a single foreground
// neighbour — the given number of times. It is useful for cleaning small barbs
// left by thinning. It panics on multi-channel input.
func Prune(src *cv.Mat, iterations int) *cv.Mat {
	requireGray(src)
	b, rows, cols := toBits(src)
	if iterations < 1 {
		iterations = 1
	}
	for it := 0; it < iterations; it++ {
		var marks []int
		for y := 0; y < rows; y++ {
			for x := 0; x < cols; x++ {
				if b[y*cols+x] == 0 {
					continue
				}
				n := eightNeighbourhood(b, y, x, rows, cols)
				if countNeighbours(n) <= 1 {
					marks = append(marks, y*cols+x)
				}
			}
		}
		if len(marks) == 0 {
			break
		}
		for _, p := range marks {
			b[p] = 0
		}
	}
	return bitsToMat(b, rows, cols)
}

// FindEndpoints returns a binary image marking the endpoints of a
// one-pixel-wide skeleton: foreground pixels with exactly one foreground
// neighbour. It panics on multi-channel input.
func FindEndpoints(src *cv.Mat) *cv.Mat {
	requireGray(src)
	b, rows, cols := toBits(src)
	out := cv.NewMat(rows, cols, 1)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			if b[y*cols+x] == 0 {
				continue
			}
			n := eightNeighbourhood(b, y, x, rows, cols)
			if countNeighbours(n) == 1 {
				out.Data[y*cols+x] = 255
			}
		}
	}
	return out
}

// FindBranchPoints returns a binary image marking the branch points of a
// one-pixel-wide skeleton: foreground pixels with three or more foreground
// neighbours. It panics on multi-channel input.
func FindBranchPoints(src *cv.Mat) *cv.Mat {
	requireGray(src)
	b, rows, cols := toBits(src)
	out := cv.NewMat(rows, cols, 1)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			if b[y*cols+x] == 0 {
				continue
			}
			n := eightNeighbourhood(b, y, x, rows, cols)
			if countNeighbours(n) >= 3 {
				out.Data[y*cols+x] = 255
			}
		}
	}
	return out
}
