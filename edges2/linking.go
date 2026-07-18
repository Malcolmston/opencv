package edges2

import cv "github.com/malcolmston/opencv"

// Point is an integer pixel coordinate with column X and row Y.
type Point struct {
	// X is the column coordinate.
	X int
	// Y is the row coordinate.
	Y int
}

// EdgeChain is an ordered list of connected edge pixels produced by
// [LinkEdges].
type EdgeChain []Point

// Length returns the number of pixels in the chain.
func (c EdgeChain) Length() int { return len(c) }

// BoundingBox returns the inclusive axis-aligned bounding box of the chain as
// its top-left and bottom-right corners. It panics on an empty chain.
func (c EdgeChain) BoundingBox() (min, max Point) {
	if len(c) == 0 {
		panic("edges2: EdgeChain.BoundingBox on empty chain")
	}
	min = c[0]
	max = c[0]
	for _, p := range c {
		if p.X < min.X {
			min.X = p.X
		}
		if p.Y < min.Y {
			min.Y = p.Y
		}
		if p.X > max.X {
			max.X = p.X
		}
		if p.Y > max.Y {
			max.Y = p.Y
		}
	}
	return min, max
}

// LinkEdges groups the foreground pixels of a binary edge image into connected
// [EdgeChain] chains using 8-connectivity. Each edge pixel appears in exactly
// one chain. Within a chain the pixels are ordered by a depth-first walk that
// prefers continuing along the current direction, so the ordering follows the
// contour; branch points simply attach their neighbours in scan order. Chains
// with fewer than minLength pixels are discarded. Scanning is top-to-bottom,
// left-to-right, making the result deterministic. It panics on multi-channel
// input.
func LinkEdges(edges *cv.Mat, minLength int) []EdgeChain {
	edges2RequireGray(edges, "LinkEdges")
	rows, cols := edges.Rows, edges.Cols
	visited := make([]bool, rows*cols)
	neigh := [8][2]int{
		{-1, -1}, {-1, 0}, {-1, 1},
		{0, -1}, {0, 1},
		{1, -1}, {1, 0}, {1, 1},
	}
	var chains []EdgeChain
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			i := y*cols + x
			if edges.Data[i] == 0 || visited[i] {
				continue
			}
			// Depth-first traversal of this connected component.
			var chain EdgeChain
			stack := []Point{{X: x, Y: y}}
			visited[i] = true
			for len(stack) > 0 {
				p := stack[len(stack)-1]
				stack = stack[:len(stack)-1]
				chain = append(chain, p)
				for _, d := range neigh {
					ny := p.Y + d[0]
					nx := p.X + d[1]
					if ny < 0 || ny >= rows || nx < 0 || nx >= cols {
						continue
					}
					ni := ny*cols + nx
					if edges.Data[ni] != 0 && !visited[ni] {
						visited[ni] = true
						stack = append(stack, Point{X: nx, Y: ny})
					}
				}
			}
			if len(chain) >= minLength {
				chains = append(chains, chain)
			}
		}
	}
	return chains
}
