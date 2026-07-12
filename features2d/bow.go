package features2d

import (
	"math"
	"math/rand"
	"sort"
)

// DescriptorMatcher is the subset of matcher behaviour used by
// [BOWImgDescriptorExtractor]: mapping each query descriptor to its single best
// train descriptor. Both [BFMatcher] and [FlannBasedMatcher] implement it.
type DescriptorMatcher interface {
	Match(query, train Descriptors) []DMatch
}

// BOWKMeansTrainer clusters float feature descriptors into a visual vocabulary
// with k-means, mirroring OpenCV's cv::BOWKMeansTrainer. Descriptors are added
// with [BOWKMeansTrainer.Add]; [BOWKMeansTrainer.Cluster] returns the cluster
// centres as float [Descriptors] — the "bag of visual words". Initialisation
// uses k-means++ with a fixed seed and Lloyd iterations, so the vocabulary is
// deterministic. The zero value is not usable; construct one with
// [NewBOWKMeansTrainer].
type BOWKMeansTrainer struct {
	clusterCount int
	maxIter      int
	epsilon      float64
	descriptors  [][]float64
}

// NewBOWKMeansTrainer returns a trainer that will produce a vocabulary of
// clusterCount words. It panics if clusterCount < 1. maxIter and epsilon bound
// the Lloyd iteration (pass 0 for the defaults: 100 iterations, epsilon 1e-4).
func NewBOWKMeansTrainer(clusterCount, maxIter int, epsilon float64) *BOWKMeansTrainer {
	if clusterCount < 1 {
		panic("features2d: BOWKMeansTrainer requires clusterCount >= 1")
	}
	if maxIter <= 0 {
		maxIter = 100
	}
	if epsilon <= 0 {
		epsilon = 1e-4
	}
	return &BOWKMeansTrainer{clusterCount: clusterCount, maxIter: maxIter, epsilon: epsilon}
}

// Add appends the float descriptor rows of one image (or any batch) to the
// training set. It panics if the descriptors are not float.
func (t *BOWKMeansTrainer) Add(desc Descriptors) {
	if desc.Float == nil {
		if desc.Len() == 0 {
			return
		}
		panic("features2d: BOWKMeansTrainer requires float descriptors")
	}
	for _, row := range desc.Float {
		cp := make([]float64, len(row))
		copy(cp, row)
		t.descriptors = append(t.descriptors, cp)
	}
}

// DescriptorsCount returns the number of descriptors added so far.
func (t *BOWKMeansTrainer) DescriptorsCount() int { return len(t.descriptors) }

// Cluster runs k-means over the accumulated descriptors and returns the cluster
// centres as float [Descriptors]. If fewer descriptors than clusters were added,
// every descriptor becomes its own centre. It panics if no descriptors were
// added.
func (t *BOWKMeansTrainer) Cluster() Descriptors {
	if len(t.descriptors) == 0 {
		panic("features2d: BOWKMeansTrainer.Cluster called with no descriptors")
	}
	k := t.clusterCount
	if k > len(t.descriptors) {
		k = len(t.descriptors)
	}
	centers := kmeans(t.descriptors, k, t.maxIter, t.epsilon)
	return NewFloatDescriptors(centers)
}

// kmeans runs deterministic k-means++ initialisation followed by Lloyd
// iterations, returning k cluster centres.
func kmeans(points [][]float64, k, maxIter int, epsilon float64) [][]float64 {
	rng := rand.New(rand.NewSource(0xb0d5eed))
	dim := len(points[0])
	centers := kmeansPlusPlusInit(points, k, rng)

	assign := make([]int, len(points))
	for iter := 0; iter < maxIter; iter++ {
		// Assignment step.
		for i, p := range points {
			best, bestD := 0, math.Inf(1)
			for c := range centers {
				d := squaredL2(p, centers[c])
				if d < bestD {
					bestD, best = d, c
				}
			}
			assign[i] = best
		}
		// Update step.
		newCenters := make([][]float64, k)
		counts := make([]int, k)
		for c := range newCenters {
			newCenters[c] = make([]float64, dim)
		}
		for i, p := range points {
			c := assign[i]
			counts[c]++
			for j := 0; j < dim; j++ {
				newCenters[c][j] += p[j]
			}
		}
		var shift float64
		for c := 0; c < k; c++ {
			if counts[c] == 0 {
				// Re-seed an empty cluster to the farthest point for stability.
				newCenters[c] = append([]float64(nil), farthestPoint(points, centers)...)
				continue
			}
			for j := 0; j < dim; j++ {
				newCenters[c][j] /= float64(counts[c])
			}
			shift += squaredL2(newCenters[c], centers[c])
		}
		centers = newCenters
		if shift < epsilon*epsilon {
			break
		}
	}
	return centers
}

// kmeansPlusPlusInit selects k initial centres with the k-means++ scheme.
func kmeansPlusPlusInit(points [][]float64, k int, rng *rand.Rand) [][]float64 {
	centers := make([][]float64, 0, k)
	first := rng.Intn(len(points))
	centers = append(centers, append([]float64(nil), points[first]...))
	dist := make([]float64, len(points))
	for i := range dist {
		dist[i] = math.Inf(1)
	}
	for len(centers) < k {
		var total float64
		last := centers[len(centers)-1]
		for i, p := range points {
			d := squaredL2(p, last)
			if d < dist[i] {
				dist[i] = d
			}
			total += dist[i]
		}
		if total == 0 {
			// All remaining points coincide with a centre; pad with copies.
			centers = append(centers, append([]float64(nil), points[rng.Intn(len(points))]...))
			continue
		}
		target := rng.Float64() * total
		var acc float64
		chosen := len(points) - 1
		for i := range points {
			acc += dist[i]
			if acc >= target {
				chosen = i
				break
			}
		}
		centers = append(centers, append([]float64(nil), points[chosen]...))
	}
	return centers
}

// farthestPoint returns the point with the greatest distance to its nearest
// existing centre.
func farthestPoint(points, centers [][]float64) []float64 {
	best, bestD := points[0], -1.0
	for _, p := range points {
		nearest := math.Inf(1)
		for _, c := range centers {
			if d := squaredL2(p, c); d < nearest {
				nearest = d
			}
		}
		if nearest > bestD {
			bestD, best = nearest, p
		}
	}
	return best
}

// BOWImgDescriptorExtractor computes the bag-of-visual-words descriptor of an
// image from its feature descriptors and a vocabulary, mirroring OpenCV's
// cv::BOWImgDescriptorExtractor. Each input descriptor is assigned to its
// nearest vocabulary word (via the configured [DescriptorMatcher]) and the
// result is the L1-normalised histogram of word occurrences, with one bin per
// vocabulary word. The zero value is not usable; construct one with
// [NewBOWImgDescriptorExtractor].
type BOWImgDescriptorExtractor struct {
	matcher    DescriptorMatcher
	vocabulary Descriptors
}

// NewBOWImgDescriptorExtractor returns an extractor that assigns descriptors to
// words using the given matcher (typically a float [BFMatcher] or a
// [FlannBasedMatcher]).
func NewBOWImgDescriptorExtractor(matcher DescriptorMatcher) *BOWImgDescriptorExtractor {
	return &BOWImgDescriptorExtractor{matcher: matcher}
}

// SetVocabulary stores the visual vocabulary (float cluster centres, as returned
// by [BOWKMeansTrainer.Cluster]). It panics if the vocabulary is not float.
func (e *BOWImgDescriptorExtractor) SetVocabulary(vocabulary Descriptors) {
	if vocabulary.Float == nil {
		panic("features2d: BOWImgDescriptorExtractor vocabulary must be float")
	}
	e.vocabulary = vocabulary
}

// VocabularySize returns the number of visual words (histogram bins).
func (e *BOWImgDescriptorExtractor) VocabularySize() int { return e.vocabulary.Len() }

// Compute returns the bag-of-words histogram for the given image descriptors: an
// L1-normalised vector of length VocabularySize. An all-zero vector is returned
// when there are no descriptors. It panics if no vocabulary has been set or the
// descriptors are not float.
func (e *BOWImgDescriptorExtractor) Compute(desc Descriptors) []float64 {
	if e.vocabulary.Len() == 0 {
		panic("features2d: BOWImgDescriptorExtractor has no vocabulary")
	}
	hist := make([]float64, e.vocabulary.Len())
	if desc.Len() == 0 {
		return hist
	}
	if desc.Float == nil {
		panic("features2d: BOWImgDescriptorExtractor requires float descriptors")
	}
	matches := e.matcher.Match(desc, e.vocabulary)
	for _, mt := range matches {
		if mt.TrainIdx >= 0 && mt.TrainIdx < len(hist) {
			hist[mt.TrainIdx]++
		}
	}
	var sum float64
	for _, v := range hist {
		sum += v
	}
	if sum > 0 {
		for i := range hist {
			hist[i] /= sum
		}
	}
	return hist
}

// wordAssignments returns, for the given descriptors, the vocabulary word index
// each was assigned to (used by tests and debugging). It is deterministic.
func (e *BOWImgDescriptorExtractor) wordAssignments(desc Descriptors) []int {
	matches := e.matcher.Match(desc, e.vocabulary)
	sort.SliceStable(matches, func(i, j int) bool { return matches[i].QueryIdx < matches[j].QueryIdx })
	out := make([]int, len(matches))
	for i, mt := range matches {
		out[i] = mt.TrainIdx
	}
	return out
}
