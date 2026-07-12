package shape

import (
	"math"
	"sort"

	cv "github.com/malcolmston/opencv"
)

// Distance types for [FitLineRobust], selecting the M-estimator loss whose
// influence function reweights each point during iteratively reweighted least
// squares. The numeric values match OpenCV's cv::DistanceTypes.
const (
	// DistL1 uses ρ(r) = |r| (least absolute deviation): weight 1/|r|.
	DistL1 = 1
	// DistL2 uses ρ(r) = r²/2 (ordinary total least squares): every weight 1.
	DistL2 = 2
	// DistL12 uses the L1–L2 (fair-like) loss 2(√(1+r²/2) − 1).
	DistL12 = 4
	// DistFair uses Fair's loss with tuning constant 1.3998.
	DistFair = 5
	// DistWelsch uses Welsch's redescending loss with tuning constant 2.9846.
	DistWelsch = 6
	// DistHuber uses Huber's loss with tuning constant 1.345.
	DistHuber = 7
)

// FitLineRobust fits a straight line to the point set with a robust
// M-estimator, returning it in OpenCV's (vx, vy, x0, y0) form: (vx, vy) is a unit
// direction and (x0, y0) a point on the line (the weighted centroid). distType
// selects the loss ([DistL1], [DistL2], [DistL12], [DistFair], [DistWelsch] or
// [DistHuber]); [DistL2] reproduces the plain total-least-squares [FitLine].
//
// The fit uses iteratively reweighted least squares: it starts from the
// ordinary total-least-squares direction, then repeatedly recomputes each
// point's orthogonal residual, derives a weight from the chosen loss (scaled by a
// robust estimate of the residual spread, the normalised median absolute
// deviation), and refits by weighted total least squares until the direction
// stops changing. Redescending losses (Welsch) and the others down-weight
// outliers so a few gross errors no longer dominate the fit. The iteration is
// deterministic and the direction sign is canonicalised as in [FitLine].
//
// It panics on an unknown distType.
func FitLineRobust(pts []cv.Point, distType int) (vx, vy, x0, y0 float64) {
	if distType == DistL2 {
		return FitLine(pts)
	}
	switch distType {
	case DistL1, DistL12, DistFair, DistWelsch, DistHuber:
	default:
		panic("shape: FitLineRobust unknown distType")
	}
	n := len(pts)
	if n == 0 {
		return 1, 0, 0, 0
	}
	fp := make([]fpoint, n)
	for i, p := range pts {
		fp[i] = fpoint{float64(p.X), float64(p.Y)}
	}
	// Initial estimate: unweighted total least squares.
	vx, vy, x0, y0 = FitLine(pts)
	weights := make([]float64, n)
	residuals := make([]float64, n)
	for iter := 0; iter < 50; iter++ {
		// Orthogonal residual of each point to the current line.
		nx, ny := -vy, vx // line normal
		for i, p := range fp {
			residuals[i] = math.Abs(nx*(p.x-x0) + ny*(p.y-y0))
		}
		scale := robustScale(residuals)
		if scale < 1e-12 {
			scale = 1e-12
		}
		for i := range fp {
			weights[i] = lossWeight(distType, residuals[i]/scale)
		}
		nvx, nvy, nx0, ny0, ok := weightedTLS(fp, weights)
		if !ok {
			break
		}
		// Keep the direction continuous with the previous iterate.
		if nvx*vx+nvy*vy < 0 {
			nvx, nvy = -nvx, -nvy
		}
		change := math.Hypot(nvx-vx, nvy-vy)
		vx, vy, x0, y0 = nvx, nvy, nx0, ny0
		if change < 1e-10 {
			break
		}
	}
	// Canonicalise sign for determinism, as FitLine does.
	if vx < 0 || (vx == 0 && vy < 0) {
		vx, vy = -vx, -vy
	}
	return vx, vy, x0, y0
}

// weightedTLS fits a line by weighted total least squares: the weighted centroid
// and the dominant eigenvector of the weighted scatter matrix. It reports false
// when the total weight or scatter is degenerate.
func weightedTLS(pts []fpoint, w []float64) (vx, vy, x0, y0 float64, ok bool) {
	var sw, mx, my float64
	for i, p := range pts {
		sw += w[i]
		mx += w[i] * p.x
		my += w[i] * p.y
	}
	if sw < 1e-12 {
		return 0, 0, 0, 0, false
	}
	mx /= sw
	my /= sw
	var sxx, sxy, syy float64
	for i, p := range pts {
		dx := p.x - mx
		dy := p.y - my
		sxx += w[i] * dx * dx
		sxy += w[i] * dx * dy
		syy += w[i] * dy * dy
	}
	if sxx+syy < 1e-15 {
		return 1, 0, mx, my, true
	}
	tr := sxx + syy
	det := sxx*syy - sxy*sxy
	disc := math.Sqrt(math.Max(0, tr*tr/4-det))
	lambda := tr/2 + disc
	ex := sxy
	ey := lambda - sxx
	if math.Hypot(ex, ey) < 1e-12 {
		if sxx >= syy {
			ex, ey = 1, 0
		} else {
			ex, ey = 0, 1
		}
	}
	norm := math.Hypot(ex, ey)
	return ex / norm, ey / norm, mx, my, true
}

// robustScale returns a robust estimate of residual spread: the median absolute
// residual divided by 0.6745 (making it consistent with the standard deviation
// for Gaussian noise).
func robustScale(residuals []float64) float64 {
	tmp := make([]float64, len(residuals))
	copy(tmp, residuals)
	sort.Float64s(tmp)
	n := len(tmp)
	var med float64
	if n%2 == 1 {
		med = tmp[n/2]
	} else {
		med = (tmp[n/2-1] + tmp[n/2]) / 2
	}
	return med / 0.6745
}

// lossWeight returns the IRLS weight ρ'(t)/t for the chosen loss at normalised
// residual t.
func lossWeight(distType int, t float64) float64 {
	at := math.Abs(t)
	switch distType {
	case DistL1:
		if at < 1e-9 {
			return 1e9
		}
		return 1 / at
	case DistL12:
		return 1 / math.Sqrt(1+t*t/2)
	case DistFair:
		const c = 1.3998
		return 1 / (1 + at/c)
	case DistWelsch:
		const c = 2.9846
		return math.Exp(-(t * t) / (c * c))
	case DistHuber:
		const c = 1.345
		if at <= c {
			return 1
		}
		return c / at
	default:
		return 1
	}
}
