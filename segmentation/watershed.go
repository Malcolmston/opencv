package segmentation

import (
	"container/heap"
	"math"

	cv "github.com/malcolmston/opencv"
)

// WatershedMarker is the sentinel label written to watershed-line pixels by
// [Watershed]. OpenCV stores -1 in a signed CV_32S marker image; because
// [cv.Mat] holds unsigned 8-bit samples this port uses 255 instead, which also
// caps usable basin labels at 254.
const WatershedMarker = 255

// Watershed performs marker-controlled watershed segmentation of img using the
// seed labels in markers, returning a new single-channel [cv.Mat] of labels.
//
// markers is read from channel 0: a value of 0 means "unknown" (to be filled)
// and any value k in [1, 254] marks a seed belonging to region k. img may be
// single- or three-channel; three-channel input is converted to grayscale and a
// 3x3 Sobel gradient magnitude is used as the flooding relief.
//
// The algorithm is Meyer's flooding: every unknown pixel adjacent to a labelled
// pixel is pushed onto a priority queue keyed by gradient magnitude (lowest
// first, ties broken by insertion order for determinism). Pixels are popped in
// order of increasing relief; a popped pixel adopts the label of its labelled
// 4-neighbours when they all agree, and otherwise becomes a watershed-line pixel
// tagged [WatershedMarker]. The result is a fresh Mat and markers is not
// modified.
//
// It panics if img is empty, if markers is nil or does not match the image
// dimensions.
func Watershed(img *cv.Mat, markers *cv.Mat) *cv.Mat {
	if img.Empty() {
		panic("segmentation: Watershed on empty image")
	}
	if markers == nil || markers.Rows != img.Rows || markers.Cols != img.Cols {
		panic("segmentation: Watershed markers must match the image dimensions")
	}

	rows, cols := img.Rows, img.Cols
	n := rows * cols
	mag := gradientMagnitude(img)

	const (
		unlabeled = 0
		boundary  = -1
	)
	labels := make([]int, n)
	for i := 0; i < n; i++ {
		labels[i] = int(markers.Data[i*markers.Channels])
	}

	inQueue := make([]bool, n)
	pq := &pixelHeap{}
	heap.Init(pq)
	seq := 0
	push := func(x, y int) {
		idx := y*cols + x
		if inQueue[idx] || labels[idx] != unlabeled {
			return
		}
		inQueue[idx] = true
		heap.Push(pq, pixelItem{priority: mag[idx], seq: seq, x: x, y: y})
		seq++
	}

	// Seed the queue with unknown pixels bordering a marker.
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			if labels[y*cols+x] == unlabeled {
				continue
			}
			for _, o := range neighbors4 {
				nx, ny := x+o.dx, y+o.dy
				if nx >= 0 && nx < cols && ny >= 0 && ny < rows {
					push(nx, ny)
				}
			}
		}
	}

	for pq.Len() > 0 {
		it := heap.Pop(pq).(pixelItem)
		idx := it.y*cols + it.x

		found := unlabeled
		conflict := false
		for _, o := range neighbors4 {
			nx, ny := it.x+o.dx, it.y+o.dy
			if nx < 0 || nx >= cols || ny < 0 || ny >= rows {
				continue
			}
			nl := labels[ny*cols+nx]
			if nl == unlabeled || nl == boundary {
				continue
			}
			if found == unlabeled {
				found = nl
			} else if found != nl {
				conflict = true
			}
		}
		if conflict {
			labels[idx] = boundary
			continue
		}
		if found == unlabeled {
			// Every labelled neighbour turned into a watershed line before this
			// pixel was popped. Clear its queued flag so it can be reconsidered
			// if a neighbour later adopts a basin label.
			inQueue[idx] = false
			continue
		}
		labels[idx] = found
		for _, o := range neighbors4 {
			nx, ny := it.x+o.dx, it.y+o.dy
			if nx >= 0 && nx < cols && ny >= 0 && ny < rows {
				push(nx, ny)
			}
		}
	}

	out := cv.NewMat(rows, cols, 1)
	for i := 0; i < n; i++ {
		switch {
		case labels[i] == boundary:
			out.Data[i] = WatershedMarker
		case labels[i] < 0 || labels[i] > 254:
			out.Data[i] = WatershedMarker
		default:
			out.Data[i] = uint8(labels[i])
		}
	}
	return out
}

// gradientMagnitude returns the per-pixel Sobel gradient magnitude of img as a
// flat row-major slice of length Rows*Cols. Three-channel input is reduced to
// grayscale first.
func gradientMagnitude(img *cv.Mat) []float64 {
	gray := img
	if img.Channels != 1 {
		gray = cv.CvtColor(img, cv.ColorRGB2Gray)
	}
	gx := cv.SobelFloat(gray, 1, 0, 3)[0]
	gy := cv.SobelFloat(gray, 0, 1, 3)[0]
	mag := make([]float64, len(gx))
	for i := range gx {
		mag[i] = math.Hypot(gx[i], gy[i])
	}
	return mag
}

// pixelItem is a queue entry for the flooding heap. seq preserves insertion
// order so equal-priority pixels are popped deterministically.
type pixelItem struct {
	priority float64
	seq      int
	x, y     int
}

// pixelHeap is a min-heap of pixelItem ordered by priority then insertion order.
type pixelHeap []pixelItem

func (h pixelHeap) Len() int { return len(h) }
func (h pixelHeap) Less(i, j int) bool {
	if h[i].priority != h[j].priority {
		return h[i].priority < h[j].priority
	}
	return h[i].seq < h[j].seq
}
func (h pixelHeap) Swap(i, j int) { h[i], h[j] = h[j], h[i] }
func (h *pixelHeap) Push(x any)   { *h = append(*h, x.(pixelItem)) }
func (h *pixelHeap) Pop() any {
	old := *h
	n := len(old)
	it := old[n-1]
	*h = old[:n-1]
	return it
}
