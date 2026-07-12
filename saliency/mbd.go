package saliency

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// MinimumBarrierSaliency implements the Minimum Barrier Distance (MBD) salient
// object detector of Zhang, Sclaroff, Lin, Shen, Price & Mech, "Minimum Barrier
// Salient Object Detection at 80 FPS" (ICCV 2015).
//
// Saliency is the minimum barrier distance from every pixel to the image
// boundary, where the barrier cost of a path is the difference between the
// highest and lowest intensity encountered along it (max minus min), not the
// sum of gradients. Because the image border is treated as a background seed
// set, pixels that can only be reached by crossing a strong intensity ridge —
// the interior of a distinct object — receive a large distance and therefore a
// high saliency, while background connected smoothly to the border stays dark.
//
// The distance field is computed with the paper's fast approximation: a small
// fixed number of alternating raster-scan (forward/backward) passes that
// propagate, for each pixel, the lowest barrier cost found so far together with
// the running max and min of the corresponding path.
//
// Construct one with [NewMinimumBarrierSaliency]. It satisfies [StaticSaliency].
type MinimumBarrierSaliency struct {
	// Passes is the number of raster-scan sweeps (each pass alternates
	// direction). More passes tighten the approximation; the default is 3.
	Passes int
	// BlurRadius is the radius of a final box smoothing applied to the distance
	// field (0 disables it). The default is 3.
	BlurRadius int
}

// NewMinimumBarrierSaliency returns a detector with three scan passes and a
// smoothing radius of three.
func NewMinimumBarrierSaliency() *MinimumBarrierSaliency {
	return &MinimumBarrierSaliency{Passes: 3, BlurRadius: 3}
}

// ComputeSaliency returns the minimum-barrier saliency map of img: a
// single-channel [cv.Mat] the same size as img, normalised to [0,255]. It
// panics if img is nil or empty.
func (m *MinimumBarrierSaliency) ComputeSaliency(img *cv.Mat) *cv.Mat {
	gray := grayPlane(img)
	rows, cols := gray.rows, gray.cols
	n := rows * cols

	const inf = math.MaxFloat64
	dist := make([]float64, n)
	up := make([]float64, n) // running max along the best path
	lo := make([]float64, n) // running min along the best path
	for i := 0; i < n; i++ {
		v := gray.data[i]
		y, x := i/cols, i%cols
		if y == 0 || y == rows-1 || x == 0 || x == cols-1 {
			dist[i] = 0
		} else {
			dist[i] = inf
		}
		up[i] = v
		lo[i] = v
	}

	// relax tries to improve pixel i using already-visited neighbour j.
	relax := func(i, j int) {
		v := gray.data[i]
		nu := math.Max(up[j], v)
		nl := math.Min(lo[j], v)
		cand := nu - nl
		if cand < dist[i] {
			dist[i] = cand
			up[i] = nu
			lo[i] = nl
		}
	}

	passes := m.Passes
	if passes < 1 {
		passes = 3
	}
	for p := 0; p < passes; p++ {
		// Forward pass: consider up and left neighbours.
		for y := 0; y < rows; y++ {
			for x := 0; x < cols; x++ {
				i := y*cols + x
				if y > 0 {
					relax(i, i-cols)
				}
				if x > 0 {
					relax(i, i-1)
				}
			}
		}
		// Backward pass: consider down and right neighbours.
		for y := rows - 1; y >= 0; y-- {
			for x := cols - 1; x >= 0; x-- {
				i := y*cols + x
				if y < rows-1 {
					relax(i, i+cols)
				}
				if x < cols-1 {
					relax(i, i+1)
				}
			}
		}
	}

	out := newPlane(rows, cols)
	copy(out.data, dist)
	if m.BlurRadius > 0 {
		out = meanBlur(out, m.BlurRadius)
	}
	return out.normalizedMat()
}
