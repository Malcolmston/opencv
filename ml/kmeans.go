package ml

import (
	"math"
	"math/rand"
)

// KMeans partitions data into k clusters with Lloyd's algorithm using k-means++
// seeding. It returns, for every input sample, the index of the cluster it was
// assigned to, and the k cluster centres (centroids). Iteration stops when no
// sample changes cluster or after maxIter passes, whichever comes first.
//
// The k-means++ seeding draws on randomness; seed makes the whole run
// reproducible. It panics if data is empty or ragged, or if k is not in
// [1, len(data)].
func KMeans(data [][]float64, k, maxIter int, seed int64) (labels []int, centers [][]float64) {
	if len(data) == 0 {
		panic("ml: KMeans given no data")
	}
	if k <= 0 || k > len(data) {
		panic("ml: KMeans requires 1 <= k <= len(data)")
	}
	dim := len(data[0])
	for _, d := range data {
		if len(d) != dim {
			panic("ml: KMeans requires all samples to have the same length")
		}
	}
	if maxIter <= 0 {
		maxIter = 100
	}
	rng := rand.New(rand.NewSource(seed))

	centers = kmeansPlusPlusInit(data, k, dim, rng)
	labels = make([]int, len(data))
	for i := range labels {
		labels[i] = -1
	}

	for iter := 0; iter < maxIter; iter++ {
		changed := false
		// Assignment step.
		for i, s := range data {
			best, bestDist := 0, math.Inf(1)
			for c := 0; c < k; c++ {
				if d := squaredEuclidean(s, centers[c]); d < bestDist {
					bestDist = d
					best = c
				}
			}
			if labels[i] != best {
				labels[i] = best
				changed = true
			}
		}
		// Update step.
		sums := make([][]float64, k)
		counts := make([]int, k)
		for c := 0; c < k; c++ {
			sums[c] = make([]float64, dim)
		}
		for i, s := range data {
			c := labels[i]
			counts[c]++
			for j, v := range s {
				sums[c][j] += v
			}
		}
		for c := 0; c < k; c++ {
			if counts[c] == 0 {
				// Re-seed an empty cluster at the point farthest from its
				// current centre so the partition stays deterministic.
				centers[c] = farthestPoint(data, centers, rng)
				continue
			}
			for j := 0; j < dim; j++ {
				centers[c][j] = sums[c][j] / float64(counts[c])
			}
		}
		if !changed && iter > 0 {
			break
		}
	}
	return labels, centers
}

// kmeansPlusPlusInit chooses k initial centres with the D²-weighted k-means++
// procedure.
func kmeansPlusPlusInit(data [][]float64, k, dim int, rng *rand.Rand) [][]float64 {
	centers := make([][]float64, 0, k)
	first := rng.Intn(len(data))
	centers = append(centers, cloneSample(data[first]))

	dist2 := make([]float64, len(data))
	for i := range dist2 {
		dist2[i] = math.Inf(1)
	}
	for len(centers) < k {
		last := centers[len(centers)-1]
		var total float64
		for i, s := range data {
			if d := squaredEuclidean(s, last); d < dist2[i] {
				dist2[i] = d
			}
			total += dist2[i]
		}
		next := chooseWeighted(dist2, total, rng)
		centers = append(centers, cloneSample(data[next]))
	}
	return centers
}

// chooseWeighted returns an index drawn with probability proportional to
// weights. When the total weight is zero (all points coincide with a centre) it
// falls back to a uniform choice.
func chooseWeighted(weights []float64, total float64, rng *rand.Rand) int {
	if total <= 0 {
		return rng.Intn(len(weights))
	}
	target := rng.Float64() * total
	var acc float64
	for i, w := range weights {
		acc += w
		if acc >= target {
			return i
		}
	}
	return len(weights) - 1
}

// farthestPoint returns a copy of the sample whose nearest existing centre is
// the most distant, used to revive empty clusters. rng breaks ties.
func farthestPoint(data, centers [][]float64, rng *rand.Rand) []float64 {
	bestIdx, bestDist := rng.Intn(len(data)), -1.0
	for i, s := range data {
		nearest := math.Inf(1)
		for _, c := range centers {
			if d := squaredEuclidean(s, c); d < nearest {
				nearest = d
			}
		}
		if nearest > bestDist {
			bestDist = nearest
			bestIdx = i
		}
	}
	return cloneSample(data[bestIdx])
}
