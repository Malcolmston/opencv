package shape

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// ShapeContext computes the shape-context descriptor of a point set: for each
// point, a coarse log-polar histogram of the relative positions of all the other
// points, as introduced by Belongie, Malik and Puzicha. The histogram bins
// distances on a logarithmic scale (so nearby structure is described more finely
// than distant structure) and directions on a uniform angular scale.
//
// Distances are normalised by the mean inter-point distance, making the
// descriptor invariant to scale; enabling RotationInvariant additionally
// measures angles relative to the point set's dominant orientation, making it
// invariant to rotation. The zero value uses sensible defaults (12 angular bins,
// 5 radial bins, radial extent 0.125–2 of the mean distance) via its accessor
// methods, or configure the fields explicitly.
type ShapeContext struct {
	// AngularBins is the number of orientation bins (default 12).
	AngularBins int
	// RadialBins is the number of log-distance bins (default 5).
	RadialBins int
	// InnerRadius is the smallest binned distance as a fraction of the mean
	// inter-point distance (default 0.125).
	InnerRadius float64
	// OuterRadius is the largest binned distance as a fraction of the mean
	// inter-point distance (default 2.0).
	OuterRadius float64
	// RotationInvariant measures angles relative to the dominant orientation.
	RotationInvariant bool
}

func (sc ShapeContext) angularBins() int {
	if sc.AngularBins > 0 {
		return sc.AngularBins
	}
	return 12
}

func (sc ShapeContext) radialBins() int {
	if sc.RadialBins > 0 {
		return sc.RadialBins
	}
	return 5
}

func (sc ShapeContext) innerRadius() float64 {
	if sc.InnerRadius > 0 {
		return sc.InnerRadius
	}
	return 0.125
}

func (sc ShapeContext) outerRadius() float64 {
	if sc.OuterRadius > 0 {
		return sc.OuterRadius
	}
	return 2.0
}

// Descriptor returns the shape-context histograms of the point set: one
// normalised histogram per point, each of length RadialBins×AngularBins, laid
// out radial-major (all angular bins of radial ring 0, then ring 1, …). Each
// histogram sums to 1 (or to 0 for a single-point input). The result is
// deterministic.
func (sc ShapeContext) Descriptor(points []Point2D) [][]float64 {
	n := len(points)
	aBins := sc.angularBins()
	rBins := sc.radialBins()
	hist := make([][]float64, n)
	for i := range hist {
		hist[i] = make([]float64, aBins*rBins)
	}
	if n < 2 {
		return hist
	}

	// Mean inter-point distance for scale normalisation.
	var sum float64
	var cnt int
	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			sum += dist2D(points[i], points[j])
			cnt++
		}
	}
	meanDist := sum / float64(cnt)
	if meanDist < 1e-12 {
		return hist
	}

	orient := 0.0
	if sc.RotationInvariant {
		orient = dominantOrientation(points)
	}

	logInner := math.Log(sc.innerRadius())
	logOuter := math.Log(sc.outerRadius())
	span := logOuter - logInner
	twoPi := 2 * math.Pi
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			if i == j {
				continue
			}
			dx := points[j].X - points[i].X
			dy := points[j].Y - points[i].Y
			r := math.Hypot(dx, dy) / meanDist
			if r < 1e-12 {
				continue
			}
			// Radial bin on a log scale, clamped into range.
			rb := int((math.Log(r) - logInner) / span * float64(rBins))
			if rb < 0 {
				rb = 0
			} else if rb >= rBins {
				rb = rBins - 1
			}
			theta := math.Atan2(dy, dx) - orient
			theta = math.Mod(theta, twoPi)
			if theta < 0 {
				theta += twoPi
			}
			ab := int(theta / twoPi * float64(aBins))
			if ab < 0 {
				ab = 0
			} else if ab >= aBins {
				ab = aBins - 1
			}
			hist[i][rb*aBins+ab]++
		}
		// Normalise this point's histogram to unit mass.
		inv := 1.0 / float64(n-1)
		for k := range hist[i] {
			hist[i][k] *= inv
		}
	}
	return hist
}

// dominantOrientation returns the angle (radians) of the point set's principal
// axis, used for rotation-invariant shape contexts.
func dominantOrientation(points []Point2D) float64 {
	var mx, my float64
	for _, p := range points {
		mx += p.X
		my += p.Y
	}
	n := float64(len(points))
	mx /= n
	my /= n
	var sxx, sxy, syy float64
	for _, p := range points {
		dx := p.X - mx
		dy := p.Y - my
		sxx += dx * dx
		sxy += dx * dy
		syy += dy * dy
	}
	// Angle of the eigenvector of the larger eigenvalue.
	return 0.5 * math.Atan2(2*sxy, sxx-syy)
}

// chiSquareCost returns the χ² histogram distance ½·Σ (a−b)²/(a+b) between two
// shape-context histograms.
func chiSquareCost(a, b []float64) float64 {
	var c float64
	for i := range a {
		s := a[i] + b[i]
		if s > 1e-12 {
			d := a[i] - b[i]
			c += d * d / s
		}
	}
	return 0.5 * c
}

// ShapeContextDistanceExtractor measures the dissimilarity of two shapes with
// the shape-context matching pipeline of Belongie, Malik and Puzicha, mirroring
// OpenCV's cv::ShapeContextDistanceExtractor. Each contour is resampled to a
// fixed number of points, described by shape contexts, and the two point sets are
// put into optimal correspondence by solving a linear assignment problem
// (Hungarian algorithm) over their χ² histogram costs. A thin-plate spline is
// fitted to the matched pairs, and the final distance combines the average
// matching cost with the bending energy of that spline.
//
// The zero value is usable with defaults; use [NewShapeContextDistanceExtractor]
// to override them. Matching is scale-invariant, and rotation-invariant when
// RotationInvariant is set, so a shape compares as far more similar to a rotated
// and scaled copy of itself than to a genuinely different shape. All steps are
// deterministic.
type ShapeContextDistanceExtractor struct {
	// NPoints is the number of sample points each contour is resampled to
	// (default 100).
	NPoints int
	// AngularBins and RadialBins configure the shape-context histograms.
	AngularBins int
	RadialBins  int
	// RotationInvariant enables rotation-invariant shape contexts.
	RotationInvariant bool
	// BendingEnergyWeight weights the thin-plate-spline bending energy in the
	// final distance (default 0.3).
	BendingEnergyWeight float64
	// Regularization is the thin-plate spline smoothing parameter used when
	// fitting the matched correspondences (default 0, exact interpolation).
	Regularization float64
}

// NewShapeContextDistanceExtractor returns an extractor that resamples each
// contour to nPoints and, when rotationInvariant is true, uses rotation-invariant
// shape contexts.
func NewShapeContextDistanceExtractor(nPoints int, rotationInvariant bool) *ShapeContextDistanceExtractor {
	return &ShapeContextDistanceExtractor{
		NPoints:             nPoints,
		RotationInvariant:   rotationInvariant,
		BendingEnergyWeight: 0.3,
	}
}

func (e *ShapeContextDistanceExtractor) nPoints() int {
	if e.NPoints > 0 {
		return e.NPoints
	}
	return 100
}

func (e *ShapeContextDistanceExtractor) bendingWeight() float64 {
	if e.BendingEnergyWeight != 0 {
		return e.BendingEnergyWeight
	}
	return 0.3
}

// ComputeDistance returns the shape-context dissimilarity between contours c1 and
// c2: 0 for identical shapes and growing as they differ. It panics if either
// contour has fewer than two points.
func (e *ShapeContextDistanceExtractor) ComputeDistance(c1, c2 []cv.Point) float64 {
	if len(c1) < 2 || len(c2) < 2 {
		panic("shape: ShapeContextDistanceExtractor.ComputeDistance needs at least two points per contour")
	}
	n := e.nPoints()
	p1 := resampleContour(c1, n)
	p2 := resampleContour(c2, n)

	sc := ShapeContext{
		AngularBins:       e.AngularBins,
		RadialBins:        e.RadialBins,
		RotationInvariant: e.RotationInvariant,
	}
	h1 := sc.Descriptor(p1)
	h2 := sc.Descriptor(p2)

	// Cost matrix of χ² histogram distances, then optimal assignment.
	cost := make([][]float64, len(p1))
	for i := range p1 {
		cost[i] = make([]float64, len(p2))
		for j := range p2 {
			cost[i][j] = chiSquareCost(h1[i], h2[j])
		}
	}
	assign, total := SolveAssignment(cost)

	matchedSrc := make([]Point2D, 0, len(p1))
	matchedDst := make([]Point2D, 0, len(p1))
	var matched int
	for i, j := range assign {
		if j < 0 {
			continue
		}
		matchedSrc = append(matchedSrc, p1[i])
		matchedDst = append(matchedDst, p2[j])
		matched++
	}
	scCost := 0.0
	if matched > 0 {
		scCost = total / float64(matched)
	}

	// Thin-plate spline bending energy of the matched correspondence.
	var bending float64
	if matched >= 3 {
		tps := &ThinPlateSplineShapeTransformer{Regularization: e.Regularization}
		if safeEstimate(tps, matchedSrc, matchedDst) {
			bending = tps.BendingEnergy()
		}
	}
	// Guard against numerical blow-up of the bending term on odd inputs.
	if math.IsNaN(bending) || math.IsInf(bending, 0) {
		bending = 0
	}
	return scCost + e.bendingWeight()*bending
}

// safeEstimate fits the transformer, recovering from a degeneracy panic and
// reporting whether the fit succeeded.
func safeEstimate(tps *ThinPlateSplineShapeTransformer, src, dst []Point2D) (ok bool) {
	defer func() {
		if recover() != nil {
			ok = false
		}
	}()
	tps.EstimateTransformation(src, dst)
	return true
}

// resampleContour resamples a contour to exactly n points, evenly spaced along
// its arc length, treating it as a closed polygon. It is deterministic.
func resampleContour(contour []cv.Point, n int) []Point2D {
	m := len(contour)
	pts := FloatPoints(contour)
	if m <= 1 || n <= 0 {
		out := make([]Point2D, 0, n)
		for i := 0; i < n; i++ {
			if m == 0 {
				out = append(out, Point2D{})
			} else {
				out = append(out, pts[0])
			}
		}
		return out
	}
	// Cumulative arc length around the closed polygon.
	cum := make([]float64, m+1)
	for i := 0; i < m; i++ {
		cum[i+1] = cum[i] + dist2D(pts[i], pts[(i+1)%m])
	}
	perimeter := cum[m]
	if perimeter < 1e-12 {
		out := make([]Point2D, n)
		for i := range out {
			out[i] = pts[0]
		}
		return out
	}
	out := make([]Point2D, n)
	seg := 0
	for i := 0; i < n; i++ {
		target := perimeter * float64(i) / float64(n)
		for seg < m && cum[seg+1] < target {
			seg++
		}
		if seg >= m {
			seg = m - 1
		}
		segLen := cum[seg+1] - cum[seg]
		t := 0.0
		if segLen > 1e-12 {
			t = (target - cum[seg]) / segLen
		}
		a := pts[seg]
		b := pts[(seg+1)%m]
		out[i] = Point2D{X: a.X + t*(b.X-a.X), Y: a.Y + t*(b.Y-a.Y)}
	}
	return out
}
