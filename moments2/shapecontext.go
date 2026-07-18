package moments2

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// ShapeContextHistogram is the log-polar histogram computed at one reference
// point of a shape. It is stored row-major with RadialBins rows and AngularBins
// columns; the value at radial bin r and angular bin a is at index
// r*AngularBins + a. Its entries sum to the number of other points that fell
// inside the histogram's radial range.
type ShapeContextHistogram struct {
	// RadialBins is the number of log-distance bins.
	RadialBins int
	// AngularBins is the number of angular bins spanning the full circle.
	AngularBins int
	// Counts holds the per-bin counts in row-major order.
	Counts []float64
}

// ComputeShapeContext computes the shape context descriptor of a point set: for
// each reference point it builds a log-polar histogram of the relative
// positions of all the other points. Distances are normalized by their mean and
// binned logarithmically between innerRadius and outerRadius (as fractions of
// that mean distance); angles are binned uniformly over the full circle. The
// returned slice is parallel to points. It panics if there are fewer than two
// points or the bin counts are not positive.
func ComputeShapeContext(points []cv.Point2f, radialBins, angularBins int, innerRadius, outerRadius float64) []ShapeContextHistogram {
	n := len(points)
	if n < 2 {
		panic("moments2: ComputeShapeContext requires at least two points")
	}
	if radialBins < 1 || angularBins < 1 {
		panic("moments2: ComputeShapeContext requires positive bin counts")
	}
	// Mean pairwise distance for scale normalization.
	var meanDist float64
	var pairs int
	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			meanDist += math.Hypot(points[i].X-points[j].X, points[i].Y-points[j].Y)
			pairs++
		}
	}
	if pairs > 0 {
		meanDist /= float64(pairs)
	}
	if meanDist == 0 {
		meanDist = 1
	}
	logInner := math.Log(innerRadius)
	logOuter := math.Log(outerRadius)
	out := make([]ShapeContextHistogram, n)
	for i := 0; i < n; i++ {
		h := ShapeContextHistogram{
			RadialBins:  radialBins,
			AngularBins: angularBins,
			Counts:      make([]float64, radialBins*angularBins),
		}
		for j := 0; j < n; j++ {
			if j == i {
				continue
			}
			dx := points[j].X - points[i].X
			dy := points[j].Y - points[i].Y
			dist := math.Hypot(dx, dy) / meanDist
			if dist <= 0 {
				continue
			}
			logD := math.Log(dist)
			if logD < logInner || logD > logOuter {
				continue
			}
			rBin := int((logD - logInner) / (logOuter - logInner) * float64(radialBins))
			if rBin >= radialBins {
				rBin = radialBins - 1
			}
			if rBin < 0 {
				rBin = 0
			}
			theta := math.Atan2(dy, dx)
			if theta < 0 {
				theta += 2 * math.Pi
			}
			aBin := int(theta / (2 * math.Pi) * float64(angularBins))
			if aBin >= angularBins {
				aBin = angularBins - 1
			}
			h.Counts[rBin*angularBins+aBin]++
		}
		out[i] = h
	}
	return out
}

// ShapeContextCost returns the chi-square dissimilarity between two shape context
// histograms, the standard local matching cost, in the range [0, 1]. Both
// histograms are normalized to sum to one before comparison so that points with
// different numbers of neighbours remain comparable. It panics if the histograms
// have different dimensions.
func ShapeContextCost(a, b ShapeContextHistogram) float64 {
	if a.RadialBins != b.RadialBins || a.AngularBins != b.AngularBins {
		panic("moments2: ShapeContextCost histogram dimension mismatch")
	}
	var sumA, sumB float64
	for i := range a.Counts {
		sumA += a.Counts[i]
		sumB += b.Counts[i]
	}
	if sumA == 0 || sumB == 0 {
		return 0
	}
	var cost float64
	for i := range a.Counts {
		pa := a.Counts[i] / sumA
		pb := b.Counts[i] / sumB
		denom := pa + pb
		if denom > 0 {
			d := pa - pb
			cost += d * d / denom
		}
	}
	return 0.5 * cost
}

// ShapeContextMatchCost returns the mean over all reference points of the best
// per-point [ShapeContextCost] between two shape context descriptors, a global
// dissimilarity between two shapes. For each histogram in a it takes the minimum
// cost against any histogram in b. It returns 0 if either descriptor is empty.
func ShapeContextMatchCost(a, b []ShapeContextHistogram) float64 {
	if len(a) == 0 || len(b) == 0 {
		return 0
	}
	var total float64
	for i := range a {
		best := math.Inf(1)
		for j := range b {
			if c := ShapeContextCost(a[i], b[j]); c < best {
				best = c
			}
		}
		total += best
	}
	return total / float64(len(a))
}
