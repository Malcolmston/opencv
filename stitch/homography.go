package stitch

import (
	"math"
	"math/rand"
)

// PointF is a floating-point image coordinate (x is the column, y is the row).
type PointF struct {
	// X is the horizontal (column) coordinate.
	X float64
	// Y is the vertical (row) coordinate.
	Y float64
}

// Sub returns the vector from q to p (p minus q).
func (p PointF) Sub(q PointF) PointF {
	return PointF{p.X - q.X, p.Y - q.Y}
}

// Distance returns the Euclidean distance between p and q.
func (p PointF) Distance(q PointF) float64 {
	return math.Hypot(p.X-q.X, p.Y-q.Y)
}

// Match is a single point correspondence between two images: Src is the point
// in the source image and Dst the matching point in the destination image.
type Match struct {
	// Src is the location of the feature in the source image.
	Src PointF
	// Dst is the location of the corresponding feature in the destination image.
	Dst PointF
}

// Homography is a 3×3 projective transform in row-major order. It maps a source
// image point p to a destination point p' by p' = H·p in homogeneous
// coordinates, followed by a perspective divide.
type Homography [3][3]float64

// IdentityHomography returns the identity transform, which maps every point to
// itself.
func IdentityHomography() Homography {
	return Homography{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}
}

// TranslationHomography returns a homography that shifts points by (tx, ty).
func TranslationHomography(tx, ty float64) Homography {
	return Homography{{1, 0, tx}, {0, 1, ty}, {0, 0, 1}}
}

// ScaleHomography returns a homography that scales the x axis by sx and the y
// axis by sy about the origin.
func ScaleHomography(sx, sy float64) Homography {
	return Homography{{sx, 0, 0}, {0, sy, 0}, {0, 0, 1}}
}

// ApplyXY transforms the point (x, y) and returns the mapped coordinates. If the
// homogeneous denominator is zero (a point mapped to infinity) it returns the
// numerators unchanged.
func (h Homography) ApplyXY(x, y float64) (float64, float64) {
	nx := h[0][0]*x + h[0][1]*y + h[0][2]
	ny := h[1][0]*x + h[1][1]*y + h[1][2]
	w := h[2][0]*x + h[2][1]*y + h[2][2]
	if w == 0 {
		return nx, ny
	}
	return nx / w, ny / w
}

// Apply transforms the point p and returns the mapped point.
func (h Homography) Apply(p PointF) PointF {
	x, y := h.ApplyXY(p.X, p.Y)
	return PointF{x, y}
}

// Mul returns the composition h·other, whose action first applies other and then
// h.
func (h Homography) Mul(other Homography) Homography {
	var r Homography
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			var s float64
			for k := 0; k < 3; k++ {
				s += h[i][k] * other[k][j]
			}
			r[i][j] = s
		}
	}
	return r
}

// Det returns the determinant of the 3×3 matrix.
func (h Homography) Det() float64 {
	return h[0][0]*(h[1][1]*h[2][2]-h[1][2]*h[2][1]) -
		h[0][1]*(h[1][0]*h[2][2]-h[1][2]*h[2][0]) +
		h[0][2]*(h[1][0]*h[2][1]-h[1][1]*h[2][0])
}

// Inverse returns the inverse transform and true, or the zero homography and
// false if the matrix is singular.
func (h Homography) Inverse() (Homography, bool) {
	d := h.Det()
	if math.Abs(d) < 1e-18 {
		return Homography{}, false
	}
	inv := 1 / d
	var r Homography
	r[0][0] = (h[1][1]*h[2][2] - h[1][2]*h[2][1]) * inv
	r[0][1] = (h[0][2]*h[2][1] - h[0][1]*h[2][2]) * inv
	r[0][2] = (h[0][1]*h[1][2] - h[0][2]*h[1][1]) * inv
	r[1][0] = (h[1][2]*h[2][0] - h[1][0]*h[2][2]) * inv
	r[1][1] = (h[0][0]*h[2][2] - h[0][2]*h[2][0]) * inv
	r[1][2] = (h[0][2]*h[1][0] - h[0][0]*h[1][2]) * inv
	r[2][0] = (h[1][0]*h[2][1] - h[1][1]*h[2][0]) * inv
	r[2][1] = (h[0][1]*h[2][0] - h[0][0]*h[2][1]) * inv
	r[2][2] = (h[0][0]*h[1][1] - h[0][1]*h[1][0]) * inv
	return r, true
}

// Normalize scales the matrix so its bottom-right entry equals 1, giving a
// canonical representation of the same transform. If that entry is zero the
// matrix is returned unchanged.
func (h Homography) Normalize() Homography {
	if h[2][2] == 0 {
		return h
	}
	inv := 1 / h[2][2]
	var r Homography
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			r[i][j] = h[i][j] * inv
		}
	}
	return r
}

// TransferPoints applies the homography to every point in pts and returns the
// mapped points in a new slice.
func (h Homography) TransferPoints(pts []PointF) []PointF {
	out := make([]PointF, len(pts))
	for i, p := range pts {
		out[i] = h.Apply(p)
	}
	return out
}

// ReprojectionError returns the Euclidean distance in the destination image
// between the mapped source point of m and its observed destination point.
func ReprojectionError(h Homography, m Match) float64 {
	return h.Apply(m.Src).Distance(m.Dst)
}

// MeanReprojectionError returns the average [ReprojectionError] over matches. It
// returns 0 for an empty slice.
func MeanReprojectionError(h Homography, matches []Match) float64 {
	if len(matches) == 0 {
		return 0
	}
	var s float64
	for _, m := range matches {
		s += ReprojectionError(h, m)
	}
	return s / float64(len(matches))
}

// similarityNormalize returns the isotropic-scaling similarity transform that
// moves the centroid of pts to the origin and scales so the mean distance to
// the origin is sqrt(2), together with the normalized points. This is Hartley's
// preconditioning for the DLT.
func similarityNormalize(pts []PointF) (Homography, []PointF) {
	n := len(pts)
	var cx, cy float64
	for _, p := range pts {
		cx += p.X
		cy += p.Y
	}
	cx /= float64(n)
	cy /= float64(n)
	var meanDist float64
	for _, p := range pts {
		meanDist += math.Hypot(p.X-cx, p.Y-cy)
	}
	meanDist /= float64(n)
	scale := 1.0
	if meanDist > 1e-12 {
		scale = math.Sqrt2 / meanDist
	}
	t := Homography{
		{scale, 0, -scale * cx},
		{0, scale, -scale * cy},
		{0, 0, 1},
	}
	out := make([]PointF, n)
	for i, p := range pts {
		out[i] = PointF{scale*p.X - scale*cx, scale*p.Y - scale*cy}
	}
	return t, out
}

// EstimateHomographyDLT estimates the homography that best maps the source
// points to the destination points of matches, in the total least-squares sense,
// using the normalized Direct Linear Transform. It needs at least four
// correspondences and returns false if there are too few or the configuration is
// degenerate (for example all points collinear).
func EstimateHomographyDLT(matches []Match) (Homography, bool) {
	if len(matches) < 4 {
		return Homography{}, false
	}
	src := make([]PointF, len(matches))
	dst := make([]PointF, len(matches))
	for i, m := range matches {
		src[i] = m.Src
		dst[i] = m.Dst
	}
	ts, ns := similarityNormalize(src)
	td, nd := similarityNormalize(dst)

	// Solve for 8 unknowns h0..h7 with h8 fixed to 1, in normalized space.
	n := len(matches)
	a := make([][]float64, 2*n)
	b := make([]float64, 2*n)
	for i := 0; i < n; i++ {
		x, y := ns[i].X, ns[i].Y
		xp, yp := nd[i].X, nd[i].Y
		a[2*i] = []float64{x, y, 1, 0, 0, 0, -x * xp, -y * xp}
		b[2*i] = xp
		a[2*i+1] = []float64{0, 0, 0, x, y, 1, -x * yp, -y * yp}
		b[2*i+1] = yp
	}
	// Normal equations: (Aᵀ A) h = Aᵀ b, an 8×8 symmetric system.
	ata := make([][]float64, 8)
	atb := make([]float64, 8)
	for i := 0; i < 8; i++ {
		ata[i] = make([]float64, 8)
	}
	for r := 0; r < 2*n; r++ {
		for i := 0; i < 8; i++ {
			atb[i] += a[r][i] * b[r]
			for j := 0; j < 8; j++ {
				ata[i][j] += a[r][i] * a[r][j]
			}
		}
	}
	sol, ok := solveLinear(ata, atb)
	if !ok {
		return Homography{}, false
	}
	hn := Homography{
		{sol[0], sol[1], sol[2]},
		{sol[3], sol[4], sol[5]},
		{sol[6], sol[7], 1},
	}
	// Denormalize: H = Td⁻¹ · Hn · Ts.
	tdInv, ok := td.Inverse()
	if !ok {
		return Homography{}, false
	}
	h := tdInv.Mul(hn).Mul(ts)
	return h.Normalize(), true
}

// EstimateHomographyRANSAC robustly estimates the homography between the source
// and destination points of matches in the presence of outliers. It repeatedly
// fits a candidate homography to a random minimal sample of four
// correspondences, scores it by the number of inliers whose [ReprojectionError]
// is below threshold pixels, and keeps the best consensus set, finally refitting
// on all inliers. seed makes the sampling deterministic.
//
// It returns the estimated homography, the indices into matches of the inliers,
// and true; or false if there are fewer than four matches or no acceptable model
// was found.
func EstimateHomographyRANSAC(matches []Match, threshold float64, iterations int, seed int64) (Homography, []int, bool) {
	if len(matches) < 4 {
		return Homography{}, nil, false
	}
	if iterations < 1 {
		iterations = 1
	}
	rng := rand.New(rand.NewSource(seed))
	n := len(matches)
	bestInliers := []int(nil)
	var bestModel Homography
	thr2 := threshold * threshold
	for it := 0; it < iterations; it++ {
		// Draw four distinct indices.
		var idx [4]int
		chooseFour(rng, n, &idx)
		sample := []Match{matches[idx[0]], matches[idx[1]], matches[idx[2]], matches[idx[3]]}
		model, ok := EstimateHomographyDLT(sample)
		if !ok {
			continue
		}
		var inliers []int
		for i, m := range matches {
			mp := model.Apply(m.Src)
			dx := mp.X - m.Dst.X
			dy := mp.Y - m.Dst.Y
			if dx*dx+dy*dy <= thr2 {
				inliers = append(inliers, i)
			}
		}
		if len(inliers) > len(bestInliers) {
			bestInliers = inliers
			bestModel = model
		}
	}
	if len(bestInliers) < 4 {
		return Homography{}, nil, false
	}
	refit := make([]Match, len(bestInliers))
	for i, idx := range bestInliers {
		refit[i] = matches[idx]
	}
	if model, ok := EstimateHomographyDLT(refit); ok {
		bestModel = model
		// Recompute inliers with the refined model.
		var inliers []int
		for i, m := range matches {
			mp := model.Apply(m.Src)
			dx := mp.X - m.Dst.X
			dy := mp.Y - m.Dst.Y
			if dx*dx+dy*dy <= thr2 {
				inliers = append(inliers, i)
			}
		}
		if len(inliers) >= len(bestInliers) {
			bestInliers = inliers
		}
	}
	return bestModel, bestInliers, true
}

// chooseFour fills idx with four distinct indices in [0,n) drawn from rng.
func chooseFour(rng *rand.Rand, n int, idx *[4]int) {
	for k := 0; k < 4; k++ {
		for {
			c := rng.Intn(n)
			dup := false
			for j := 0; j < k; j++ {
				if idx[j] == c {
					dup = true
					break
				}
			}
			if !dup {
				idx[k] = c
				break
			}
		}
	}
}
