package segmentation

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// KMeansSegmentation clusters the pixels of img into k colour groups with
// deterministic Lloyd's k-means and returns a [LabelMap] whose label is the
// cluster index of each pixel, together with the final cluster mean colours
// (each a channels-length slice). This is colour quantisation used as
// segmentation: pixels sharing a colour cluster share a label regardless of
// their spatial position, matching cv2.kmeans applied to pixel colours.
//
// Initial centres are chosen deterministically by the k-means++ furthest-point
// rule seeded from the mean colour, so repeated calls on the same image give
// identical results without any random dependence. img may have any number of
// channels; k must be >= 1 and is reduced to the pixel count when larger.
//
// It panics if img is empty or k < 1.
func KMeansSegmentation(img *cv.Mat, k, iterations int) (*LabelMap, [][]float64) {
	if img.Empty() {
		panic("segmentation: KMeansSegmentation on empty image")
	}
	if k < 1 {
		panic("segmentation: KMeansSegmentation requires k >= 1")
	}
	rows, cols, ch := img.Rows, img.Cols, img.Channels
	n := rows * cols
	if k > n {
		k = n
	}
	if iterations < 1 {
		iterations = 1
	}

	pix := make([][]float64, n)
	for i := 0; i < n; i++ {
		p := make([]float64, ch)
		b := i * ch
		for c := 0; c < ch; c++ {
			p[c] = float64(img.Data[b+c])
		}
		pix[i] = p
	}

	centers := kmeansPlusPlusInit(pix, k, ch)
	assign := make([]int, n)

	for it := 0; it < iterations; it++ {
		changed := false
		for i := 0; i < n; i++ {
			best, bestD := 0, math.MaxFloat64
			for c := 0; c < k; c++ {
				d := sqDistN(pix[i], centers[c])
				if d < bestD {
					bestD, best = d, c
				}
			}
			if assign[i] != best {
				assign[i] = best
				changed = true
			}
		}
		sum := make([][]float64, k)
		cnt := make([]int, k)
		for c := range sum {
			sum[c] = make([]float64, ch)
		}
		for i := 0; i < n; i++ {
			c := assign[i]
			cnt[c]++
			for d := 0; d < ch; d++ {
				sum[c][d] += pix[i][d]
			}
		}
		for c := 0; c < k; c++ {
			if cnt[c] == 0 {
				continue
			}
			for d := 0; d < ch; d++ {
				centers[c][d] = sum[c][d] / float64(cnt[c])
			}
		}
		if !changed && it > 0 {
			break
		}
	}

	labels, count := relabelConsecutive(assign)
	// Reorder the returned centres to align with the renumbered labels.
	orderedCenters := make([][]float64, count)
	seen := make([]bool, count)
	for i := 0; i < n; i++ {
		l := labels[i]
		if !seen[l] {
			seen[l] = true
			orderedCenters[l] = append([]float64(nil), centers[assign[i]]...)
		}
	}
	return &LabelMap{Rows: rows, Cols: cols, Count: count, Labels: labels}, orderedCenters
}

// kmeansPlusPlusInit selects k initial centres deterministically: the first is
// the overall mean colour and each subsequent centre is the pixel furthest (in
// squared colour distance) from the centres chosen so far, with ties broken by
// the lowest pixel index.
func kmeansPlusPlusInit(pix [][]float64, k, ch int) [][]float64 {
	n := len(pix)
	centers := make([][]float64, 0, k)
	mean := make([]float64, ch)
	for i := 0; i < n; i++ {
		for c := 0; c < ch; c++ {
			mean[c] += pix[i][c]
		}
	}
	for c := 0; c < ch; c++ {
		mean[c] /= float64(n)
	}
	// Seed 0: pixel closest to the mean colour.
	best, bestD := 0, math.MaxFloat64
	for i := 0; i < n; i++ {
		if d := sqDistN(pix[i], mean); d < bestD {
			bestD, best = d, i
		}
	}
	centers = append(centers, append([]float64(nil), pix[best]...))

	dist := make([]float64, n)
	for i := range dist {
		dist[i] = sqDistN(pix[i], centers[0])
	}
	for len(centers) < k {
		far, farD := 0, -1.0
		for i := 0; i < n; i++ {
			if dist[i] > farD {
				farD, far = dist[i], i
			}
		}
		centers = append(centers, append([]float64(nil), pix[far]...))
		newC := centers[len(centers)-1]
		for i := 0; i < n; i++ {
			if d := sqDistN(pix[i], newC); d < dist[i] {
				dist[i] = d
			}
		}
	}
	return centers
}

// sqDistN is the squared Euclidean distance between two equal-length vectors.
func sqDistN(a, b []float64) float64 {
	var s float64
	for i := range a {
		d := a[i] - b[i]
		s += d * d
	}
	return s
}
