package segmentation

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// DistanceTransform returns the exact Euclidean distance from every foreground
// pixel of binary to the nearest background pixel, as a flat row-major slice of
// length Rows*Cols. A pixel is foreground when its channel-0 sample is non-zero;
// background pixels get distance 0. This matches cv2.distanceTransform with
// DIST_L2.
//
// The computation is the two-pass separable algorithm of Felzenszwalb &
// Huttenlocher ("Distance Transforms of Sampled Functions", 2012): a lower
// envelope of parabolas is swept along the columns and then the rows, so the
// result is exact (not a chamfer approximation) in linear time.
//
// It panics if binary is empty.
func DistanceTransform(binary *cv.Mat) []float64 {
	if binary.Empty() {
		panic("segmentation: DistanceTransform on empty image")
	}
	rows, cols, ch := binary.Rows, binary.Cols, binary.Channels
	const inf = 1e20

	// Squared-distance field: 0 on background, +inf on foreground.
	f := make([]float64, rows*cols)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			if binary.Data[(y*cols+x)*ch] != 0 {
				f[y*cols+x] = inf
			}
		}
	}

	// Distance transform along columns, then along rows.
	col := make([]float64, rows)
	for x := 0; x < cols; x++ {
		for y := 0; y < rows; y++ {
			col[y] = f[y*cols+x]
		}
		dt1d(col)
		for y := 0; y < rows; y++ {
			f[y*cols+x] = col[y]
		}
	}
	row := make([]float64, cols)
	for y := 0; y < rows; y++ {
		copy(row, f[y*cols:y*cols+cols])
		dt1d(row)
		copy(f[y*cols:y*cols+cols], row)
	}

	for i := range f {
		f[i] = math.Sqrt(f[i])
	}
	return f
}

// dt1d computes the 1-D squared-distance transform of f in place using the lower
// envelope of parabolas rooted at each sample.
func dt1d(f []float64) {
	n := len(f)
	if n == 0 {
		return
	}
	const inf = 1e20
	d := make([]float64, n)
	v := make([]int, n)       // locations of parabolas in the lower envelope
	z := make([]float64, n+1) // boundaries between parabolas
	k := 0
	v[0] = 0
	z[0] = -inf
	z[1] = inf
	for q := 1; q < n; q++ {
		s := ((f[q] + float64(q)*float64(q)) - (f[v[k]] + float64(v[k])*float64(v[k]))) / (2*float64(q) - 2*float64(v[k]))
		for s <= z[k] {
			k--
			s = ((f[q] + float64(q)*float64(q)) - (f[v[k]] + float64(v[k])*float64(v[k]))) / (2*float64(q) - 2*float64(v[k]))
		}
		k++
		v[k] = q
		z[k] = s
		z[k+1] = inf
	}
	k = 0
	for q := 0; q < n; q++ {
		for z[k+1] < float64(q) {
			k++
		}
		dx := float64(q) - float64(v[k])
		d[q] = dx*dx + f[v[k]]
	}
	copy(f, d)
}

// DistanceTransformWatershed splits the foreground of a binary mask into
// separate regions using distance-transform-seeded watershed, the standard
// recipe for separating touching blobs (e.g. overlapping circles). It returns a
// [LabelMap] in which label 0 is the background, each connected blob receives a
// distinct positive label, and watershed-line pixels between two blobs are
// assigned to the numerically smaller neighbouring label.
//
// A pixel of binary is foreground when its channel-0 sample is non-zero. The
// distance transform of the foreground is computed; pixels whose distance
// exceeds seedRatio times the per-blob maximum become "sure foreground" seeds,
// which are connected-component-labelled to form the markers. The negative
// distance is then flooded (peaks become basins) so that basins grow outward
// from each seed and meet along the thin necks that join touching blobs.
//
// seedRatio must be in (0, 1]; typical values are 0.5-0.7. It panics if binary
// is empty.
func DistanceTransformWatershed(binary *cv.Mat, seedRatio float64) *LabelMap {
	if binary.Empty() {
		panic("segmentation: DistanceTransformWatershed on empty image")
	}
	if seedRatio <= 0 || seedRatio > 1 {
		panic("segmentation: DistanceTransformWatershed seedRatio must be in (0, 1]")
	}
	rows, cols, ch := binary.Rows, binary.Cols, binary.Channels
	n := rows * cols
	dist := DistanceTransform(binary)

	fg := make([]bool, n)
	globalMax := 0.0
	for i := 0; i < n; i++ {
		if binary.Data[i*ch] != 0 {
			fg[i] = true
			if dist[i] > globalMax {
				globalMax = dist[i]
			}
		}
	}

	// Seed pixels: distance above a fraction of the global peak. Using the global
	// peak keeps the threshold well-defined for symmetric blobs of similar size.
	seedThresh := seedRatio * globalMax
	seedMask := cv.NewMat(rows, cols, 1)
	for i := 0; i < n; i++ {
		if fg[i] && dist[i] >= seedThresh {
			seedMask.Data[i] = 255
		}
	}
	// Label the seeds via a local connected-component pass (8-connectivity).
	seedLabels, _ := connectedComponents(seedMask)

	seed := make([]int, n)
	for i := 0; i < n; i++ {
		if seedLabels[i] > 0 {
			seed[i] = seedLabels[i]
		}
	}

	// Flood the inverted distance so basins expand from peaks toward the necks.
	relief := make([]float64, n)
	for i := 0; i < n; i++ {
		relief[i] = globalMax - dist[i]
	}
	flooded := priorityFlood(relief, seed, fg, rows, cols)

	// Resolve watershed-line pixels to the smallest labelled 4-neighbour so every
	// foreground pixel carries a region label.
	raw := make([]int, n)
	for i := 0; i < n; i++ {
		if !fg[i] {
			raw[i] = 0
			continue
		}
		l := flooded[i]
		if l > 0 {
			raw[i] = l
			continue
		}
		best := 0
		y, x := i/cols, i%cols
		for _, o := range neighbors4 {
			nx, ny := x+o.dx, y+o.dy
			if nx < 0 || nx >= cols || ny < 0 || ny >= rows {
				continue
			}
			nl := flooded[ny*cols+nx]
			if nl > 0 && (best == 0 || nl < best) {
				best = nl
			}
		}
		raw[i] = best
	}

	labels, count := relabelConsecutive(raw)
	return &LabelMap{Rows: rows, Cols: cols, Count: count, Labels: labels}
}

// connectedComponents labels the non-zero pixels of mask with 8-connectivity,
// returning a per-pixel label slice (0 = background, positive = component) and
// the number of components. It is a small self-contained pass so this package
// does not depend on sibling cv subpackages.
func connectedComponents(mask *cv.Mat) (labels []int, count int) {
	rows, cols, ch := mask.Rows, mask.Cols, mask.Channels
	n := rows * cols
	labels = make([]int, n)
	next := 0
	stack := make([]int, 0, 64)
	for start := 0; start < n; start++ {
		if mask.Data[start*ch] == 0 || labels[start] != 0 {
			continue
		}
		next++
		labels[start] = next
		stack = stack[:0]
		stack = append(stack, start)
		for len(stack) > 0 {
			idx := stack[len(stack)-1]
			stack = stack[:len(stack)-1]
			y, x := idx/cols, idx%cols
			for _, o := range neighbors8 {
				nx, ny := x+o.dx, y+o.dy
				if nx < 0 || nx >= cols || ny < 0 || ny >= rows {
					continue
				}
				nidx := ny*cols + nx
				if mask.Data[nidx*ch] == 0 || labels[nidx] != 0 {
					continue
				}
				labels[nidx] = next
				stack = append(stack, nidx)
			}
		}
	}
	return labels, next
}
