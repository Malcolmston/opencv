package shape

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// FitLine fits a straight line to the point set by total least squares and
// returns it in OpenCV's (vx, vy, x0, y0) form: (vx, vy) is a unit vector along
// the line and (x0, y0) is a point on it (the centroid of the input).
//
// The direction is the first principal component of the mean-centred points —
// the eigenvector of their 2×2 covariance matrix with the larger eigenvalue —
// so the fit minimises the sum of squared perpendicular (orthogonal) distances,
// not vertical residuals. The returned direction is normalised and its sign is
// canonicalised (vx ≥ 0, and vy ≥ 0 when vx == 0) so the result is
// deterministic. Fewer than two distinct points leave the direction undefined
// and it returns the unit x-axis through the single point (or the origin for an
// empty set).
func FitLine(pts []cv.Point) (vx, vy, x0, y0 float64) {
	n := len(pts)
	if n == 0 {
		return 1, 0, 0, 0
	}
	var mx, my float64
	for _, p := range pts {
		mx += float64(p.X)
		my += float64(p.Y)
	}
	mx /= float64(n)
	my /= float64(n)

	// Covariance (scatter) matrix of the centred points.
	var sxx, sxy, syy float64
	for _, p := range pts {
		dx := float64(p.X) - mx
		dy := float64(p.Y) - my
		sxx += dx * dx
		sxy += dx * dy
		syy += dy * dy
	}
	if sxx+syy < epsGeom {
		// All points coincide: direction is undefined, return the x-axis.
		return 1, 0, mx, my
	}

	// Larger eigenvalue of [[sxx, sxy],[sxy, syy]] and its eigenvector.
	tr := sxx + syy
	det := sxx*syy - sxy*sxy
	disc := math.Sqrt(math.Max(0, tr*tr/4-det))
	lambda := tr/2 + disc
	// Eigenvector of the symmetric matrix for eigenvalue lambda.
	ex := sxy
	ey := lambda - sxx
	if math.Hypot(ex, ey) < epsGeom {
		// Off-diagonal is ~0: axis-aligned scatter, pick the dominant axis.
		if sxx >= syy {
			ex, ey = 1, 0
		} else {
			ex, ey = 0, 1
		}
	}
	norm := math.Hypot(ex, ey)
	ex, ey = ex/norm, ey/norm
	// Canonicalise the sign for determinism.
	if ex < 0 || (ex == 0 && ey < 0) {
		ex, ey = -ex, -ey
	}
	return ex, ey, mx, my
}

// FitEllipse fits an ellipse to the point set with the direct least-squares
// method of Fitzgibbon, Pilu and Fisher, using the numerically stable
// reformulation of Halir and Flusser. It returns the ellipse as a
// [cv.RotatedRect] whose centre is the ellipse centre, whose Width and Height
// are the full axis lengths along the rotated x- and y-axes, and whose Angle is
// the rotation of that x-axis in degrees.
//
// At least five points are required to determine a general conic; with fewer,
// or when the points are degenerate (collinear) so no proper ellipse exists, a
// zero-value [cv.RotatedRect] is returned. Points are internally recentred and
// scaled for conditioning, so the fit is stable for coordinates far from the
// origin.
func FitEllipse(pts []cv.Point) cv.RotatedRect {
	n := len(pts)
	if n < 5 {
		return cv.RotatedRect{}
	}
	// Normalise coordinates (centre and scale) for numerical conditioning.
	var mx, my float64
	for _, p := range pts {
		mx += float64(p.X)
		my += float64(p.Y)
	}
	mx /= float64(n)
	my /= float64(n)
	var s float64
	for _, p := range pts {
		s += math.Hypot(float64(p.X)-mx, float64(p.Y)-my)
	}
	s /= float64(n)
	if s < epsGeom {
		return cv.RotatedRect{}
	}
	scale := 1 / s

	// Build the two design blocks D1 = [x^2, xy, y^2], D2 = [x, y, 1].
	var s1, s2, s3 [3][3]float64
	for _, p := range pts {
		x := (float64(p.X) - mx) * scale
		y := (float64(p.Y) - my) * scale
		d1 := [3]float64{x * x, x * y, y * y}
		d2 := [3]float64{x, y, 1}
		for a := 0; a < 3; a++ {
			for b := 0; b < 3; b++ {
				s1[a][b] += d1[a] * d1[b]
				s2[a][b] += d1[a] * d2[b]
				s3[a][b] += d2[a] * d2[b]
			}
		}
	}
	s3inv, ok := inv3(s3)
	if !ok {
		return cv.RotatedRect{}
	}
	// T = -S3^{-1} S2^T.
	var t [3][3]float64
	for a := 0; a < 3; a++ {
		for b := 0; b < 3; b++ {
			var sum float64
			for k := 0; k < 3; k++ {
				sum += s3inv[a][k] * s2[b][k] // s2[b][k] is (S2^T)[k][b]
			}
			t[a][b] = -sum
		}
	}
	// M = S1 + S2 T.
	var m [3][3]float64
	for a := 0; a < 3; a++ {
		for b := 0; b < 3; b++ {
			sum := s1[a][b]
			for k := 0; k < 3; k++ {
				sum += s2[a][k] * t[k][b]
			}
			m[a][b] = sum
		}
	}
	// Premultiply by C1^{-1} = [[0,0,0.5],[0,-1,0],[0.5,0,0]].
	var m2 [3][3]float64
	for b := 0; b < 3; b++ {
		m2[0][b] = 0.5 * m[2][b]
		m2[1][b] = -m[1][b]
		m2[2][b] = 0.5 * m[0][b]
	}

	// The ellipse solution is the eigenvector a1 of M2 with 4*a*c - b^2 > 0.
	a1, ok := ellipseEigenvector(m2)
	if !ok {
		return cv.RotatedRect{}
	}
	// a2 = T a1 gives the linear conic terms.
	var a2 [3]float64
	for a := 0; a < 3; a++ {
		a2[a] = t[a][0]*a1[0] + t[a][1]*a1[1] + t[a][2]*a1[2]
	}
	// Conic in the normalised frame: A x^2 + B xy + C y^2 + D x + E y + F = 0.
	A, B, C := a1[0], a1[1], a1[2]
	D, E, F := a2[0], a2[1], a2[2]

	rr, ok := conicToRotatedRect(A, B, C, D, E, F)
	if !ok {
		return cv.RotatedRect{}
	}
	// Undo the normalisation: map centre and axes back to pixel coordinates.
	rr.CenterX = rr.CenterX/scale + mx
	rr.CenterY = rr.CenterY/scale + my
	rr.Width /= scale
	rr.Height /= scale
	return rr
}

// conicToRotatedRect converts the conic A x^2 + B xy + C y^2 + D x + E y + F = 0
// (B is the full xy coefficient) into a rotated-rectangle description of the
// ellipse. It reports false when the conic is not a real ellipse.
func conicToRotatedRect(A, B, C, D, E, F float64) (cv.RotatedRect, bool) {
	det := 4*A*C - B*B // = -discriminant; positive for an ellipse
	if det <= 0 {
		return cv.RotatedRect{}, false
	}
	// Centre solves [[A, B/2],[B/2, C]] c = -[D/2, E/2].
	cx := (B*E - 2*C*D) / det
	cy := (B*D - 2*A*E) / det
	// Constant of the centred conic.
	fc := A*cx*cx + B*cx*cy + C*cy*cy + D*cx + E*cy + F
	// Eigenvalues of [[A, B/2],[B/2, C]].
	tr := A + C
	disc := math.Sqrt(math.Max(0, (A-C)*(A-C)+B*B))
	l1 := (tr - disc) / 2 // smaller eigenvalue -> longer (major) axis
	l2 := (tr + disc) / 2
	if l1 == 0 || l2 == 0 {
		return cv.RotatedRect{}, false
	}
	// Semi-axis^2 = -fc / lambda; both must be positive for a real ellipse.
	sa2 := -fc / l1
	sb2 := -fc / l2
	if sa2 <= 0 || sb2 <= 0 {
		return cv.RotatedRect{}, false
	}
	semiMajor := math.Sqrt(sa2)
	semiMinor := math.Sqrt(sb2)
	// Orient the angle along the major axis so Width (2*semiMajor) runs along it.
	theta := 0.5 * math.Atan2(B, A-C)
	c0, s0 := math.Cos(theta), math.Sin(theta)
	lam0 := A*c0*c0 + B*c0*s0 + C*s0*s0
	if math.Abs(lam0-l1) > math.Abs(lam0-l2) {
		// theta lies along the minor axis; rotate to the major axis.
		theta += math.Pi / 2
	}
	return cv.RotatedRect{
		CenterX: cx,
		CenterY: cy,
		Width:   2 * semiMajor,
		Height:  2 * semiMinor,
		Angle:   theta * 180 / math.Pi,
	}, true
}

// ellipseEigenvector returns the eigenvector of the 3×3 matrix m satisfying the
// ellipse constraint 4*a*c - b^2 > 0 (the Fitzgibbon condition), where the
// vector is (a, b, c). It reports false if no such real eigenvector exists.
func ellipseEigenvector(m [3][3]float64) ([3]float64, bool) {
	roots := eigenvalues3(m)
	best := [3]float64{}
	found := false
	bestCond := 0.0
	for _, lambda := range roots {
		v, ok := nullVector3(m, lambda)
		if !ok {
			continue
		}
		cond := 4*v[0]*v[2] - v[1]*v[1]
		if cond > 0 && (!found || cond > bestCond) {
			best = v
			bestCond = cond
			found = true
		}
	}
	return best, found
}

// eigenvalues3 returns the real eigenvalues of a general 3×3 matrix as the real
// roots of its characteristic polynomial.
func eigenvalues3(m [3][3]float64) []float64 {
	tr := m[0][0] + m[1][1] + m[2][2]
	// Sum of principal 2×2 minors.
	minorSum := m[0][0]*m[1][1] - m[0][1]*m[1][0] +
		m[0][0]*m[2][2] - m[0][2]*m[2][0] +
		m[1][1]*m[2][2] - m[1][2]*m[2][1]
	det := det3(m)
	// λ^3 - tr λ^2 + minorSum λ - det = 0.
	return realCubicRoots(1, -tr, minorSum, -det)
}

// nullVector3 returns a unit null-space vector of (m - lambda*I), i.e. an
// eigenvector of m for eigenvalue lambda, via the largest cross product of two
// rows. It reports false if the matrix is effectively full rank.
func nullVector3(m [3][3]float64, lambda float64) ([3]float64, bool) {
	b := m
	b[0][0] -= lambda
	b[1][1] -= lambda
	b[2][2] -= lambda
	rows := [3][3]float64{b[0], b[1], b[2]}
	best := [3]float64{}
	bestMag := 0.0
	for i := 0; i < 3; i++ {
		for j := i + 1; j < 3; j++ {
			c := cross3(rows[i], rows[j])
			if mag := c[0]*c[0] + c[1]*c[1] + c[2]*c[2]; mag > bestMag {
				bestMag = mag
				best = c
			}
		}
	}
	if bestMag < 1e-18 {
		return [3]float64{}, false
	}
	norm := math.Sqrt(bestMag)
	best[0] /= norm
	best[1] /= norm
	best[2] /= norm
	return best, true
}

// cross3 returns the cross product of two 3-vectors.
func cross3(a, b [3]float64) [3]float64 {
	return [3]float64{
		a[1]*b[2] - a[2]*b[1],
		a[2]*b[0] - a[0]*b[2],
		a[0]*b[1] - a[1]*b[0],
	}
}

// realCubicRoots returns the real roots of a x^3 + b x^2 + c x + d = 0 (a != 0).
func realCubicRoots(a, b, c, d float64) []float64 {
	if math.Abs(a) < 1e-18 {
		return realQuadraticRoots(b, c, d)
	}
	// Normalise to x^3 + px^2 + qx + r.
	p := b / a
	q := c / a
	r := d / a
	// Depressed cubic t^3 + At + Bd via x = t - p/3.
	shift := p / 3
	Ad := q - p*p/3
	Bd := 2*p*p*p/27 - p*q/3 + r
	disc := Bd*Bd/4 + Ad*Ad*Ad/27
	var roots []float64
	if disc > 1e-14 {
		sq := math.Sqrt(disc)
		u := math.Cbrt(-Bd/2 + sq)
		v := math.Cbrt(-Bd/2 - sq)
		roots = append(roots, u+v-shift)
	} else if disc < -1e-14 {
		// Three distinct real roots (trigonometric solution).
		m := 2 * math.Sqrt(-Ad/3)
		theta := math.Acos(clamp(3*Bd/(Ad*m), -1, 1)) / 3
		for k := 0; k < 3; k++ {
			roots = append(roots, m*math.Cos(theta-2*math.Pi*float64(k)/3)-shift)
		}
	} else {
		// Repeated roots.
		if math.Abs(Ad) < 1e-14 && math.Abs(Bd) < 1e-14 {
			roots = append(roots, -shift)
		} else {
			u := math.Cbrt(-Bd / 2)
			roots = append(roots, 2*u-shift, -u-shift)
		}
	}
	return roots
}

// realQuadraticRoots returns the real roots of b x^2 + c x + d = 0.
func realQuadraticRoots(b, c, d float64) []float64 {
	if math.Abs(b) < 1e-18 {
		if math.Abs(c) < 1e-18 {
			return nil
		}
		return []float64{-d / c}
	}
	disc := c*c - 4*b*d
	if disc < 0 {
		return nil
	}
	sq := math.Sqrt(disc)
	return []float64{(-c + sq) / (2 * b), (-c - sq) / (2 * b)}
}

// clamp constrains v to [lo, hi].
func clamp(v, lo, hi float64) float64 {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

// det3 returns the determinant of a 3×3 matrix.
func det3(m [3][3]float64) float64 {
	return m[0][0]*(m[1][1]*m[2][2]-m[1][2]*m[2][1]) -
		m[0][1]*(m[1][0]*m[2][2]-m[1][2]*m[2][0]) +
		m[0][2]*(m[1][0]*m[2][1]-m[1][1]*m[2][0])
}

// inv3 returns the inverse of a 3×3 matrix and reports whether it is invertible.
func inv3(m [3][3]float64) ([3][3]float64, bool) {
	d := det3(m)
	if math.Abs(d) < 1e-18 {
		return [3][3]float64{}, false
	}
	invd := 1 / d
	var out [3][3]float64
	out[0][0] = (m[1][1]*m[2][2] - m[1][2]*m[2][1]) * invd
	out[0][1] = (m[0][2]*m[2][1] - m[0][1]*m[2][2]) * invd
	out[0][2] = (m[0][1]*m[1][2] - m[0][2]*m[1][1]) * invd
	out[1][0] = (m[1][2]*m[2][0] - m[1][0]*m[2][2]) * invd
	out[1][1] = (m[0][0]*m[2][2] - m[0][2]*m[2][0]) * invd
	out[1][2] = (m[0][2]*m[1][0] - m[0][0]*m[1][2]) * invd
	out[2][0] = (m[1][0]*m[2][1] - m[1][1]*m[2][0]) * invd
	out[2][1] = (m[0][1]*m[2][0] - m[0][0]*m[2][1]) * invd
	out[2][2] = (m[0][0]*m[1][1] - m[0][1]*m[1][0]) * invd
	return out, true
}
