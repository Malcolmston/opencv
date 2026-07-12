package segmentation

import (
	"fmt"

	cv "github.com/malcolmston/opencv"
)

// FloodFill fills a connected region of img, starting at seed, with newVal and
// returns the number of pixels filled together with the bounding rectangle of
// the filled area. Like cv2.floodFill it mutates img in place.
//
// A candidate pixel joins the region when, for every channel c, its original
// sample lies within the tolerance band of the neighbour it spreads from:
//
//	neighbour[c] - loDiff[c] <= candidate[c] <= neighbour[c] + hiDiff[c]
//
// This is OpenCV's default floating-range comparison (each pixel is tested
// against its already-filled neighbour, not against the seed), which lets the
// fill track smooth gradients while still stopping at a sharp colour edge.
// Comparisons always use the original image data, so overwriting pixels with
// newVal during the fill does not affect the tolerance test. Only the first
// img.Channels components of newVal, loDiff and hiDiff are used.
//
// connectivity selects the neighbourhood and must be 4 (edge neighbours) or 8
// (edge and corner neighbours); any other value panics. The fill is performed
// with an explicit stack, so it is safe on large regions. It panics if img is
// empty or seed lies outside the image.
//
// The returned rectangle uses the root package's inclusive convention: a single
// filled pixel yields a 1x1 rectangle. When nothing is filled (which cannot
// happen for an in-bounds seed, since the seed itself always qualifies) count is
// zero and the rectangle is the zero [cv.Rect].
func FloodFill(img *cv.Mat, seed cv.Point, newVal cv.Scalar, loDiff, hiDiff cv.Scalar, connectivity int) (count int, rect cv.Rect) {
	if img.Empty() {
		panic("segmentation: FloodFill on empty image")
	}
	sx, sy := seed.X, seed.Y
	if sx < 0 || sx >= img.Cols || sy < 0 || sy >= img.Rows {
		panic(fmt.Sprintf("segmentation: FloodFill seed (%d,%d) out of bounds for %dx%d", sx, sy, img.Cols, img.Rows))
	}
	var neigh []offset
	switch connectivity {
	case 4:
		neigh = neighbors4
	case 8:
		neigh = neighbors8
	default:
		panic(fmt.Sprintf("segmentation: FloodFill connectivity must be 4 or 8, got %d", connectivity))
	}

	ch := img.Channels
	// Snapshot the original samples so tolerance tests are unaffected by the
	// pixels we overwrite as the fill proceeds.
	orig := make([]uint8, len(img.Data))
	copy(orig, img.Data)

	fill := make([]uint8, ch)
	for c := 0; c < ch; c++ {
		fill[c] = clampU8(newVal[c])
	}

	cols, rows := img.Cols, img.Rows
	visited := make([]bool, rows*cols)

	writePixel := func(x, y int) {
		i := (y*cols + x) * ch
		copy(img.Data[i:i+ch], fill)
	}

	minX, minY := sx, sy
	maxX, maxY := sx, sy

	stack := []cv.Point{{X: sx, Y: sy}}
	visited[sy*cols+sx] = true
	writePixel(sx, sy)
	count = 1

	for len(stack) > 0 {
		p := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		pBase := (p.Y*cols + p.X) * ch
		for _, o := range neigh {
			nx, ny := p.X+o.dx, p.Y+o.dy
			if nx < 0 || nx >= cols || ny < 0 || ny >= rows {
				continue
			}
			vi := ny*cols + nx
			if visited[vi] {
				continue
			}
			nBase := vi * ch
			if !withinTolerance(orig[pBase:pBase+ch], orig[nBase:nBase+ch], loDiff, hiDiff) {
				continue
			}
			visited[vi] = true
			writePixel(nx, ny)
			count++
			if nx < minX {
				minX = nx
			}
			if nx > maxX {
				maxX = nx
			}
			if ny < minY {
				minY = ny
			}
			if ny > maxY {
				maxY = ny
			}
			stack = append(stack, cv.Point{X: nx, Y: ny})
		}
	}

	rect = cv.Rect{X: minX, Y: minY, Width: maxX - minX + 1, Height: maxY - minY + 1}
	return count, rect
}

// withinTolerance reports whether candidate is within [ref-lo, ref+hi] on every
// channel, comparing the original samples of the source (ref) and destination
// (candidate) pixels.
func withinTolerance(ref, candidate []uint8, lo, hi cv.Scalar) bool {
	for c := range ref {
		d := float64(candidate[c]) - float64(ref[c])
		if d < -lo[c] || d > hi[c] {
			return false
		}
	}
	return true
}
