package cudafeatures2d

import (
	"github.com/malcolmston/opencv/features2d"
)

// DescriptorMatcher is a CPU-backed mirror of cv::cuda::DescriptorMatcher. It
// compares device descriptor [GpuMat]s (as produced by [ORB]) by delegating to
// features2d.BFMatcher.
//
// Construct one with [CreateBFMatcher].
type DescriptorMatcher struct {
	norm NormType
	impl *features2d.BFMatcher
}

// CreateBFMatcher returns a brute-force DescriptorMatcher using the given norm.
// Use [NormHamming] for ORB/BRIEF binary descriptors (the usual case) and
// [NormL2] for float descriptors. It mirrors
// cv::cuda::DescriptorMatcher::createBFMatcher.
func CreateBFMatcher(norm NormType) *DescriptorMatcher {
	return &DescriptorMatcher{norm: norm, impl: features2d.NewBFMatcher(norm)}
}

// buildDescriptors converts a device descriptor GpuMat into the features2d
// Descriptors representation the matcher's norm expects. Byte-packed rows become
// binary descriptors for NormHamming, or per-byte float samples for NormL2.
func (m *DescriptorMatcher) buildDescriptors(g *GpuMat) features2d.Descriptors {
	rows := descriptorsFromGpuMat(g)
	if m.norm == NormHamming {
		return features2d.NewBinaryDescriptors(rows)
	}
	frows := make([][]float64, len(rows))
	for i, r := range rows {
		fr := make([]float64, len(r))
		for j, b := range r {
			fr[j] = float64(b)
		}
		frows[i] = fr
	}
	return features2d.NewFloatDescriptors(frows)
}

// Match returns the single best train match for each query descriptor, in query
// order. It mirrors cv::cuda::DescriptorMatcher::match. Query and train are
// device descriptor [GpuMat]s. An empty input yields a nil result.
func (m *DescriptorMatcher) Match(query, train *GpuMat) []DMatch {
	if query.Empty() || train.Empty() {
		return nil
	}
	return m.impl.Match(m.buildDescriptors(query), m.buildDescriptors(train))
}

// KnnMatch returns, for each query descriptor, its k nearest train descriptors
// sorted by ascending distance. It mirrors
// cv::cuda::DescriptorMatcher::knnMatch. It panics if k < 1.
func (m *DescriptorMatcher) KnnMatch(query, train *GpuMat, k int) [][]DMatch {
	if query.Empty() || train.Empty() {
		return nil
	}
	return m.impl.KnnMatch(m.buildDescriptors(query), m.buildDescriptors(train), k)
}

// RadiusMatch returns, for each query descriptor, every train descriptor within
// maxDistance, sorted by ascending distance. It mirrors
// cv::cuda::DescriptorMatcher::radiusMatch. Rows with no match in range are
// present but empty.
func (m *DescriptorMatcher) RadiusMatch(query, train *GpuMat, maxDistance float64) [][]DMatch {
	if query.Empty() || train.Empty() {
		return nil
	}
	q := m.buildDescriptors(query)
	t := m.buildDescriptors(train)
	// Rank all train descriptors per query, then keep those within the radius.
	knn := m.impl.KnnMatch(q, t, t.Len())
	out := make([][]DMatch, len(knn))
	for i, row := range knn {
		var kept []DMatch
		for _, dm := range row {
			if dm.Distance <= maxDistance {
				kept = append(kept, dm)
			}
		}
		out[i] = kept
	}
	return out
}

// MatchConvert flattens a raw knn/radius match result (the per-query slices
// returned by [DescriptorMatcher.KnnMatch] or [DescriptorMatcher.RadiusMatch])
// into a single []DMatch, taking the best (first) match of each non-empty row.
// It mirrors the cv::cuda::DescriptorMatcher matchConvert helpers that turn the
// device match matrix into a flat vector of DMatch.
func (m *DescriptorMatcher) MatchConvert(raw [][]DMatch) []DMatch {
	var out []DMatch
	for _, row := range raw {
		if len(row) == 0 {
			continue
		}
		out = append(out, row[0])
	}
	return out
}
