package segment2

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// KMeansResult holds the outcome of colour-space k-means clustering.
type KMeansResult struct {
	// Labels is the per-pixel cluster assignment as a [LabelMap].
	Labels *LabelMap
	// Centers holds the final cluster mean colours, one channels-length slice
	// per cluster, indexed by cluster label.
	Centers [][]float64
	// Compactness is the total within-cluster squared colour distance, the
	// objective k-means minimises. Lower is tighter.
	Compactness float64
}

// segment2kmeansInit chooses k initial centres deterministically with the
// k-means++ furthest-point rule seeded from the global mean colour, so results
// are reproducible without any random dependence.
func segment2kmeansInit(pix [][]float64, k int) [][]float64 {
	n := len(pix)
	ch := len(pix[0])
	centers := make([][]float64, 0, k)

	mean := make([]float64, ch)
	for _, p := range pix {
		for c := 0; c < ch; c++ {
			mean[c] += p[c]
		}
	}
	for c := 0; c < ch; c++ {
		mean[c] /= float64(n)
	}
	// First centre: the pixel furthest from the mean (deterministic tie-break
	// by lowest index).
	best, bestD := 0, -1.0
	for i, p := range pix {
		d := segment2colorDist2(p, mean)
		if d > bestD {
			bestD, best = d, i
		}
	}
	centers = append(centers, append([]float64(nil), pix[best]...))

	dist := make([]float64, n)
	for i := range dist {
		dist[i] = math.Inf(1)
	}
	for len(centers) < k {
		last := centers[len(centers)-1]
		far, farD := 0, -1.0
		for i, p := range pix {
			d := segment2colorDist2(p, last)
			if d < dist[i] {
				dist[i] = d
			}
			if dist[i] > farD {
				farD, far = dist[i], i
			}
		}
		centers = append(centers, append([]float64(nil), pix[far]...))
	}
	return centers
}

// segment2kmeans runs Lloyd's algorithm on flat pixel vectors and returns the
// assignment, centres and compactness.
func segment2kmeans(pix [][]float64, k, iterations int) ([]int, [][]float64, float64) {
	n := len(pix)
	ch := len(pix[0])
	if k > n {
		k = n
	}
	if iterations < 1 {
		iterations = 1
	}
	centers := segment2kmeansInit(pix, k)
	assign := make([]int, n)

	for it := 0; it < iterations; it++ {
		changed := false
		for i, p := range pix {
			best, bestD := 0, math.Inf(1)
			for c, ctr := range centers {
				d := segment2colorDist2(p, ctr)
				if d < bestD {
					bestD, best = d, c
				}
			}
			if assign[i] != best {
				assign[i] = best
				changed = true
			}
		}
		sums := make([][]float64, k)
		cnt := make([]int, k)
		for c := range sums {
			sums[c] = make([]float64, ch)
		}
		for i, p := range pix {
			a := assign[i]
			for c := 0; c < ch; c++ {
				sums[a][c] += p[c]
			}
			cnt[a]++
		}
		for c := 0; c < k; c++ {
			if cnt[c] == 0 {
				continue
			}
			for j := 0; j < ch; j++ {
				centers[c][j] = sums[c][j] / float64(cnt[c])
			}
		}
		if !changed && it > 0 {
			break
		}
	}

	var compact float64
	for i, p := range pix {
		compact += segment2colorDist2(p, centers[assign[i]])
	}
	return assign, centers, compact
}

// KMeansSegment clusters the pixels of img into k colour groups with
// deterministic Lloyd's k-means and returns a [KMeansResult] whose Labels give
// each pixel its cluster index. This is colour quantisation used as
// segmentation: pixels that share a colour cluster share a label regardless of
// spatial position, matching cv2.kmeans applied to pixel colours.
//
// Initial centres are chosen deterministically (k-means++ furthest-point seeded
// from the mean colour), so repeated calls give identical results. img may have
// any number of channels; k is clamped to the pixel count when larger.
//
// It panics if img is empty or k < 1.
func KMeansSegment(img *cv.Mat, k, iterations int) *KMeansResult {
	segment2requireNonEmpty(img, "KMeansSegment")
	if k < 1 {
		panic("segment2: KMeansSegment requires k >= 1")
	}
	n := img.Rows * img.Cols
	ch := img.Channels
	pix := make([][]float64, n)
	for i := 0; i < n; i++ {
		p := make([]float64, ch)
		b := i * ch
		for c := 0; c < ch; c++ {
			p[c] = float64(img.Data[b+c])
		}
		pix[i] = p
	}
	assign, centers, compact := segment2kmeans(pix, k, iterations)

	lm := NewLabelMap(img.Rows, img.Cols)
	maxL := 0
	for i, a := range assign {
		lm.Labels[i] = a
		if a > maxL {
			maxL = a
		}
	}
	lm.NumLabels = maxL + 1
	return &KMeansResult{Labels: lm, Centers: centers, Compactness: compact}
}

// QuantizeColors reduces img to k colours by k-means clustering and returns a
// new [cv.Mat] of the same shape in which every pixel is replaced by the mean
// colour of its cluster. It is the image-valued companion to [KMeansSegment].
//
// It panics if img is empty or k < 1.
func QuantizeColors(img *cv.Mat, k, iterations int) *cv.Mat {
	res := KMeansSegment(img, k, iterations)
	out := cv.NewMat(img.Rows, img.Cols, img.Channels)
	ch := img.Channels
	for i, a := range res.Labels.Labels {
		b := i * ch
		ctr := res.Centers[a]
		for c := 0; c < ch; c++ {
			out.Data[b+c] = segment2clampU8(ctr[c])
		}
	}
	return out
}
