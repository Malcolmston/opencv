package ml2

import (
	"errors"
	"math/rand"
)

// KMeans partitions unlabelled data into k clusters with Lloyd's algorithm.
// Cluster centres are seeded deterministically with k-means++ using the
// configured seed, then refined by alternating assignment and mean-update steps
// until convergence or MaxIter is reached.
type KMeans struct {
	k       int
	maxIter int
	seed    int64

	// Centroids holds the final cluster centres, one row per cluster; it is
	// populated by Fit.
	Centroids [][]float64
	// Labels holds the cluster index assigned to each training sample; it is
	// populated by Fit.
	Labels []int
	// Inertia is the sum of squared distances of samples to their assigned
	// centroid after the final iteration.
	Inertia float64
}

// NewKMeans returns a KMeans configured for k clusters, at most maxIter
// refinement iterations and the given random seed for k-means++ initialisation.
// It panics if k < 1 or maxIter < 1.
func NewKMeans(k, maxIter int, seed int64) *KMeans {
	if k < 1 {
		panic("ml2: NewKMeans requires k >= 1")
	}
	if maxIter < 1 {
		panic("ml2: NewKMeans requires maxIter >= 1")
	}
	return &KMeans{k: k, maxIter: maxIter, seed: seed}
}

// kmeansPlusPlus chooses k initial centres via the k-means++ scheme, spreading
// them out in proportion to squared distance from the nearest chosen centre.
func (m *KMeans) kmeansPlusPlus(x [][]float64, rng *rand.Rand) [][]float64 {
	n := len(x)
	centres := make([][]float64, 0, m.k)
	first := rng.Intn(n)
	centres = append(centres, cloneVec(x[first]))
	d2 := make([]float64, n)
	for len(centres) < m.k {
		var total float64
		for i, p := range x {
			best := ml2squaredEuclidean(p, centres[0])
			for _, c := range centres[1:] {
				if dd := ml2squaredEuclidean(p, c); dd < best {
					best = dd
				}
			}
			d2[i] = best
			total += best
		}
		if total == 0 {
			// All remaining points coincide with a centre; pad by repeating.
			centres = append(centres, cloneVec(x[rng.Intn(n)]))
			continue
		}
		target := rng.Float64() * total
		var cum float64
		chosen := n - 1
		for i := 0; i < n; i++ {
			cum += d2[i]
			if cum >= target {
				chosen = i
				break
			}
		}
		centres = append(centres, cloneVec(x[chosen]))
	}
	return centres
}

// Fit clusters x, populating Centroids, Labels and Inertia. It returns an error
// if there are fewer samples than clusters. Assignment ties are broken toward
// the lower cluster index; empty clusters keep their previous centre.
func (m *KMeans) Fit(x [][]float64) error {
	if len(x) < m.k {
		return errors.New("ml2: KMeans.Fit requires at least k samples")
	}
	rng := rand.New(rand.NewSource(m.seed))
	centres := m.kmeansPlusPlus(x, rng)
	d := len(x[0])
	labels := make([]int, len(x))
	for iter := 0; iter < m.maxIter; iter++ {
		changed := false
		for i, p := range x {
			best, bestD := 0, ml2squaredEuclidean(p, centres[0])
			for c := 1; c < m.k; c++ {
				if dd := ml2squaredEuclidean(p, centres[c]); dd < bestD {
					bestD, best = dd, c
				}
			}
			if labels[i] != best {
				changed = true
			}
			labels[i] = best
		}
		// Recompute centres as the mean of assigned points.
		sums := make([][]float64, m.k)
		counts := make([]int, m.k)
		for c := 0; c < m.k; c++ {
			sums[c] = make([]float64, d)
		}
		for i, p := range x {
			c := labels[i]
			counts[c]++
			for j := 0; j < d; j++ {
				sums[c][j] += p[j]
			}
		}
		for c := 0; c < m.k; c++ {
			if counts[c] == 0 {
				continue
			}
			for j := 0; j < d; j++ {
				centres[c][j] = sums[c][j] / float64(counts[c])
			}
		}
		if !changed && iter > 0 {
			break
		}
	}
	var inertia float64
	for i, p := range x {
		inertia += ml2squaredEuclidean(p, centres[labels[i]])
	}
	m.Centroids, m.Labels, m.Inertia = centres, labels, inertia
	return nil
}

// Predict returns the index of the nearest centroid to sample. It panics before
// Fit.
func (m *KMeans) Predict(sample []float64) int {
	if m.Centroids == nil {
		panic("ml2: KMeans.Predict before Fit")
	}
	best, bestD := 0, ml2squaredEuclidean(sample, m.Centroids[0])
	for c := 1; c < len(m.Centroids); c++ {
		if dd := ml2squaredEuclidean(sample, m.Centroids[c]); dd < bestD {
			bestD, best = dd, c
		}
	}
	return best
}

// PredictBatch assigns every sample in x to its nearest centroid.
func (m *KMeans) PredictBatch(x [][]float64) []int {
	out := make([]int, len(x))
	for i, s := range x {
		out[i] = m.Predict(s)
	}
	return out
}

// cloneVec returns a copy of v.
func cloneVec(v []float64) []float64 {
	out := make([]float64, len(v))
	copy(out, v)
	return out
}
