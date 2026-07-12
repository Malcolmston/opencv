package xfeatures2d

import (
	"math"
	"math/rand"
	"sort"

	cv "github.com/malcolmston/opencv"
)

// SignaturePoint is one cluster of a PCT (Position-Color-Texture) signature: a
// weighted centroid in feature space. X and Y are the centroid image position in
// pixels; Weight is the fraction of sampled points that fall in the cluster
// (weights sum to 1 over a signature); Features is the full centroid feature
// vector (normalised position, intensity and texture) used by [SQFD].
type SignaturePoint struct {
	X, Y     float64
	Weight   float64
	Features []float64
}

// PCTSignatures computes Position-Color-Texture signatures, a port of OpenCV's
// cv::xfeatures2d::PCTSignatures for single-channel (intensity) images.
//
// A fixed, deterministically sampled set of points is measured for a small
// feature vector — normalised position, intensity, local contrast and local
// entropy. Those samples are clustered with a few iterations of k-means; each
// surviving cluster becomes a weighted [SignaturePoint]. The resulting variable
// -length signature is a compact, translation-tolerant summary of the image
// well suited to the [SQFD] distance. Two signatures of different length can be
// compared directly, unlike fixed-length descriptors.
type PCTSignatures struct {
	// SampleCount is the number of points sampled from the image.
	SampleCount int
	// ClusterCount is the maximum number of clusters (signature points).
	ClusterCount int
	// Iterations is the number of k-means refinement iterations.
	Iterations int
	// MinWeight drops clusters whose relative weight is below this fraction.
	MinWeight float64
	// WindowRadius is the half size of the neighbourhood used for the contrast
	// and entropy texture features.
	WindowRadius int

	seed int64
}

// NewPCTSignatures returns a PCTSignatures extractor with sensible defaults
// (400 samples, up to 8 clusters, 8 iterations).
func NewPCTSignatures() *PCTSignatures {
	return &PCTSignatures{
		SampleCount:  400,
		ClusterCount: 8,
		Iterations:   8,
		MinWeight:    0.02,
		WindowRadius: 3,
		seed:         0x9c75,
	}
}

const pctFeatureDim = 5 // x, y, intensity, contrast, entropy

// sampleFeatures deterministically samples points and returns their feature
// vectors (each length pctFeatureDim, all components in [0,1]).
func (p *PCTSignatures) sampleFeatures(gray *cv.Mat) [][]float64 {
	rows, cols := gray.Rows, gray.Cols
	rng := rand.New(rand.NewSource(p.seed))
	feats := make([][]float64, 0, p.SampleCount)
	for i := 0; i < p.SampleCount; i++ {
		x := rng.Intn(cols)
		y := rng.Intn(rows)
		intensity := grayAtClamped(gray, x, y) / 255
		contrast, entropy := p.texture(gray, x, y)
		feats = append(feats, []float64{
			float64(x) / float64(cols-1),
			float64(y) / float64(rows-1),
			intensity,
			contrast,
			entropy,
		})
	}
	return feats
}

// texture returns the local contrast (mean absolute deviation from the centre,
// normalised) and entropy (of a coarse 8-bin intensity histogram, normalised) in
// the window around (x, y).
func (p *PCTSignatures) texture(gray *cv.Mat, x, y int) (contrast, entropy float64) {
	r := p.WindowRadius
	centre := grayAtClamped(gray, x, y)
	var sumAbs float64
	var count float64
	var hist [8]float64
	for dy := -r; dy <= r; dy++ {
		for dx := -r; dx <= r; dx++ {
			v := grayAtClamped(gray, x+dx, y+dy)
			sumAbs += math.Abs(v - centre)
			bin := int(v / 32)
			if bin > 7 {
				bin = 7
			}
			hist[bin]++
			count++
		}
	}
	contrast = sumAbs / count / 255
	for _, h := range hist {
		if h > 0 {
			pr := h / count
			entropy -= pr * math.Log2(pr)
		}
	}
	entropy /= 3 // log2(8) = 3, so entropy is normalised to [0,1]
	return contrast, entropy
}

// ComputeSignature samples and clusters img and returns its PCT signature as a
// slice of weighted [SignaturePoint]s (weights sum to 1). img may be single- or
// three-channel; a colour image is converted to gray.
func (p *PCTSignatures) ComputeSignature(img *cv.Mat) []SignaturePoint {
	gray := toGray(img)
	cols := gray.Cols
	rows := gray.Rows
	feats := p.sampleFeatures(gray)
	if len(feats) == 0 {
		return nil
	}
	k := p.ClusterCount
	if k > len(feats) {
		k = len(feats)
	}
	// Deterministic k-means initialisation: evenly spaced samples.
	centres := make([][]float64, k)
	for c := 0; c < k; c++ {
		idx := c * len(feats) / k
		centres[c] = append([]float64(nil), feats[idx]...)
	}
	assign := make([]int, len(feats))
	for iter := 0; iter < p.Iterations; iter++ {
		// Assignment step.
		for i, f := range feats {
			best := 0
			bestD := math.Inf(1)
			for c := 0; c < k; c++ {
				d := featDist2(f, centres[c])
				if d < bestD {
					bestD = d
					best = c
				}
			}
			assign[i] = best
		}
		// Update step.
		sums := make([][]float64, k)
		counts := make([]int, k)
		for c := 0; c < k; c++ {
			sums[c] = make([]float64, pctFeatureDim)
		}
		for i, f := range feats {
			c := assign[i]
			counts[c]++
			for d := 0; d < pctFeatureDim; d++ {
				sums[c][d] += f[d]
			}
		}
		for c := 0; c < k; c++ {
			if counts[c] == 0 {
				continue
			}
			for d := 0; d < pctFeatureDim; d++ {
				centres[c][d] = sums[c][d] / float64(counts[c])
			}
		}
	}
	// Build signature: weight = fraction of points, drop tiny clusters.
	counts := make([]int, k)
	for _, a := range assign {
		counts[a]++
	}
	total := float64(len(feats))
	var sig []SignaturePoint
	for c := 0; c < k; c++ {
		w := float64(counts[c]) / total
		if counts[c] == 0 || w < p.MinWeight {
			continue
		}
		sig = append(sig, SignaturePoint{
			X:        centres[c][0] * float64(cols-1),
			Y:        centres[c][1] * float64(rows-1),
			Weight:   w,
			Features: append([]float64(nil), centres[c]...),
		})
	}
	// Renormalise weights to sum to 1 after dropping, and sort for determinism.
	var wsum float64
	for _, s := range sig {
		wsum += s.Weight
	}
	if wsum > 0 {
		for i := range sig {
			sig[i].Weight /= wsum
		}
	}
	sort.Slice(sig, func(a, b int) bool {
		if sig[a].X != sig[b].X {
			return sig[a].X < sig[b].X
		}
		return sig[a].Y < sig[b].Y
	})
	return sig
}

// featDist2 returns the squared Euclidean distance between two feature vectors.
func featDist2(a, b []float64) float64 {
	var s float64
	for i := range a {
		d := a[i] - b[i]
		s += d * d
	}
	return s
}

// SQFD returns the Signature Quadratic Form Distance between two PCT signatures,
// the distance used with [PCTSignatures] (OpenCV's
// cv::xfeatures2d::PCTSignaturesSQFD). The similarity between two cluster
// centroids is the Gaussian kernel exp(-alpha·d²) of their feature-space
// distance; a larger alpha makes the kernel more local. The distance is
// sqrt(w·A·w) over the concatenated signed weight vector, and is 0 for identical
// signatures and larger for dissimilar ones.
func SQFD(a, b []SignaturePoint, alpha float64) float64 {
	n := len(a) + len(b)
	if n == 0 {
		return 0
	}
	feats := make([][]float64, 0, n)
	weights := make([]float64, 0, n)
	for _, s := range a {
		feats = append(feats, s.Features)
		weights = append(weights, s.Weight)
	}
	for _, s := range b {
		feats = append(feats, s.Features)
		weights = append(weights, -s.Weight)
	}
	var acc float64
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			d2 := featDist2(feats[i], feats[j])
			sim := math.Exp(-alpha * d2)
			acc += weights[i] * weights[j] * sim
		}
	}
	if acc < 0 {
		acc = 0
	}
	return math.Sqrt(acc)
}
