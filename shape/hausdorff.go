package shape

import (
	"math"
	"sort"

	cv "github.com/malcolmston/opencv"
)

// Distance flags for [HausdorffDistanceExtractor], selecting the ground metric
// used between individual points. The values match OpenCV's cv::DistanceTypes.
const (
	// HausdorffL1 uses the Manhattan (city-block) distance |dx| + |dy|.
	HausdorffL1 = 1
	// HausdorffL2 uses the Euclidean distance √(dx² + dy²).
	HausdorffL2 = 2
)

// DirectedHausdorff returns the directed Hausdorff distance from point set a to
// point set b: the largest, over points of a, of the distance from that point to
// its nearest neighbour in b. It is not symmetric. metric selects the ground
// distance ([HausdorffL1] or [HausdorffL2]); it panics on an unknown metric or an
// empty input.
func DirectedHausdorff(a, b []Point2D, metric int) float64 {
	dists := directedNearest(a, b, metric)
	var worst float64
	for _, d := range dists {
		if d > worst {
			worst = d
		}
	}
	return worst
}

// directedNearest returns, for each point of a, the distance to its nearest
// neighbour in b under the given metric.
func directedNearest(a, b []Point2D, metric int) []float64 {
	if len(a) == 0 || len(b) == 0 {
		panic("shape: Hausdorff distance on empty point set")
	}
	ground := groundMetric(metric)
	out := make([]float64, len(a))
	for i, p := range a {
		best := math.Inf(1)
		for _, q := range b {
			if d := ground(p, q); d < best {
				best = d
			}
		}
		out[i] = best
	}
	return out
}

// groundMetric returns the point-to-point distance function for a metric flag.
func groundMetric(metric int) func(a, b Point2D) float64 {
	switch metric {
	case HausdorffL1:
		return func(a, b Point2D) float64 { return math.Abs(a.X-b.X) + math.Abs(a.Y-b.Y) }
	case HausdorffL2:
		return dist2D
	default:
		panic("shape: unknown Hausdorff metric")
	}
}

// HausdorffDistanceExtractor measures the dissimilarity of two shapes by the
// (partial) Hausdorff distance between their point sets, mirroring OpenCV's
// cv::HausdorffDistanceExtractor. The symmetric Hausdorff distance is the larger
// of the two directed distances between the shapes.
//
// RankProportion selects a partial Hausdorff distance robust to outliers: rather
// than the maximum nearest-neighbour distance, the value at the given rank
// quantile of the sorted nearest-neighbour distances is used (1.0 is the classic
// maximum; smaller values ignore that fraction of the worst matches). The zero
// value uses the Euclidean metric and the classic maximum.
type HausdorffDistanceExtractor struct {
	// DistanceFlag selects the ground metric ([HausdorffL1] or [HausdorffL2];
	// default Euclidean).
	DistanceFlag int
	// RankProportion is the quantile (0,1] of nearest-neighbour distances taken
	// as the directed distance (default 1.0, the maximum).
	RankProportion float64
}

// NewHausdorffDistanceExtractor returns an extractor using the given ground
// metric and rank proportion (pass 1.0 for the classic Hausdorff distance).
func NewHausdorffDistanceExtractor(distanceFlag int, rankProportion float64) *HausdorffDistanceExtractor {
	return &HausdorffDistanceExtractor{DistanceFlag: distanceFlag, RankProportion: rankProportion}
}

func (e *HausdorffDistanceExtractor) metric() int {
	if e.DistanceFlag == 0 {
		return HausdorffL2
	}
	return e.DistanceFlag
}

func (e *HausdorffDistanceExtractor) rank() float64 {
	if e.RankProportion <= 0 || e.RankProportion > 1 {
		return 1.0
	}
	return e.RankProportion
}

// ComputeDistance returns the symmetric (partial) Hausdorff distance between
// contours c1 and c2. It panics if either contour is empty.
func (e *HausdorffDistanceExtractor) ComputeDistance(c1, c2 []cv.Point) float64 {
	a := FloatPoints(c1)
	b := FloatPoints(c2)
	metric := e.metric()
	rank := e.rank()
	d1 := rankedDistance(directedNearest(a, b, metric), rank)
	d2 := rankedDistance(directedNearest(b, a, metric), rank)
	return math.Max(d1, d2)
}

// rankedDistance returns the value at the given rank quantile of the sorted
// nearest-neighbour distances.
func rankedDistance(dists []float64, rank float64) float64 {
	if len(dists) == 0 {
		return 0
	}
	sorted := make([]float64, len(dists))
	copy(sorted, dists)
	sort.Float64s(sorted)
	idx := int(math.Ceil(rank*float64(len(sorted)))) - 1
	if idx < 0 {
		idx = 0
	}
	if idx >= len(sorted) {
		idx = len(sorted) - 1
	}
	return sorted[idx]
}
