package hdr

import (
	"errors"
	"math"

	cv "github.com/malcolmston/opencv"
)

// CameraResponse is a recovered camera response function (CRF). For each
// channel it holds a 256-entry lookup table: Curve[c][z] is the linear scene
// radiance (in arbitrary but consistent units) that corresponds to the 8-bit
// pixel value z in channel c.
//
// The table is monotonically non-decreasing for a well-behaved camera. Its
// absolute scale is arbitrary — only ratios between entries are meaningful —
// because calibration fixes the response up to a single global factor.
type CameraResponse struct {
	// Channels is the number of channels the response was calibrated for.
	Channels int
	// Curve holds one 256-entry lookup table per channel, indexed by pixel
	// value. Values are strictly positive.
	Curve [][]float64
}

// logCurve returns the natural logarithm of Curve[c], i.e. the Debevec g(z).
func (r *CameraResponse) logCurve(c int) []float64 {
	g := make([]float64, 256)
	for z := 0; z < 256; z++ {
		g[z] = math.Log(r.Curve[c][z])
	}
	return g
}

// LinearResponse returns a CameraResponse whose curve is the identity (linear)
// mapping for the given number of channels: Curve[c][z] = z+1. It is a useful
// default when the true response is already known to be linear.
func LinearResponse(channels int) *CameraResponse {
	curve := make([][]float64, channels)
	for c := range curve {
		curve[c] = make([]float64, 256)
		for z := 0; z < 256; z++ {
			curve[c][z] = float64(z) + 1
		}
	}
	return &CameraResponse{Channels: channels, Curve: curve}
}

// hat is the Debevec triangular ("hat") weighting: it peaks at the middle of
// the 8-bit range and is never zero, so no sample is discarded outright.
func hat(z int) float64 {
	if z <= 127 {
		return float64(z) + 1
	}
	return float64(256 - z)
}

// DefaultLambda is the smoothness weight used by [CalibrateDebevec] when the
// caller passes a non-positive lambda.
const DefaultLambda = 20.0

// DefaultSamples is the number of pixel locations [CalibrateDebevec] samples
// when the caller passes a non-positive sample count.
const DefaultSamples = 70

func validateStack(images []*cv.Mat, times []float64) error {
	if len(images) < 2 {
		return errors.New("hdr: need at least two exposures")
	}
	if len(images) != len(times) {
		return errors.New("hdr: number of images and exposure times differ")
	}
	rows, cols, ch := images[0].Rows, images[0].Cols, images[0].Channels
	for i, m := range images {
		if m == nil || m.Empty() {
			return errors.New("hdr: nil or empty image in stack")
		}
		if m.Rows != rows || m.Cols != cols || m.Channels != ch {
			return errors.New("hdr: all images must share dimensions and channel count")
		}
		if times[i] <= 0 {
			return errors.New("hdr: exposure times must be strictly positive")
		}
	}
	return nil
}

// samplePixels returns a deterministic, evenly spaced set of pixel indices
// (into the flat per-pixel grid, ignoring channels). Sampling on a fixed grid
// keeps calibration reproducible.
func samplePixels(rows, cols, n int) []int {
	total := rows * cols
	if n >= total {
		out := make([]int, total)
		for i := range out {
			out[i] = i
		}
		return out
	}
	out := make([]int, n)
	// Spread indices across the whole image with a fixed stride.
	for i := 0; i < n; i++ {
		// Map i into [0,total) as evenly as possible; add half-step offset so
		// the samples are centred rather than clustered at the origin.
		out[i] = (i*total + total/2) / n
	}
	return out
}

// CalibrateDebevec recovers the camera response function from a bracket of LDR
// images and their exposure times using the Debevec & Malik least-squares
// method. It solves, per channel, for the log response g(z) and per-sample log
// irradiance jointly, minimising the exposure data term plus a second-order
// smoothness term (weighted by lambda) subject to fixing g at the middle of the
// range. samples controls how many pixel locations are drawn; pass zero for
// [DefaultSamples]. lambda is the smoothness weight; pass zero for
// [DefaultLambda].
//
// The images must be aligned, share dimensions and channel count, and be listed
// with matching, strictly positive exposure times. The returned response has a
// monotonically non-decreasing curve for a well-behaved camera.
func CalibrateDebevec(images []*cv.Mat, times []float64, samples, lambda int) (*CameraResponse, error) {
	if err := validateStack(images, times); err != nil {
		return nil, err
	}
	nSamp := samples
	if nSamp <= 0 {
		nSamp = DefaultSamples
	}
	lam := float64(lambda)
	if lam <= 0 {
		lam = DefaultLambda
	}
	rows, cols, ch := images[0].Rows, images[0].Cols, images[0].Channels
	pixels := samplePixels(rows, cols, nSamp)
	logTimes := make([]float64, len(times))
	for j, t := range times {
		logTimes[j] = math.Log(t)
	}

	curve := make([][]float64, ch)
	for c := 0; c < ch; c++ {
		g := solveDebevecChannel(images, logTimes, pixels, c, lam)
		lut := make([]float64, 256)
		for z := 0; z < 256; z++ {
			lut[z] = math.Exp(g[z])
		}
		curve[c] = lut
	}
	return &CameraResponse{Channels: ch, Curve: curve}, nil
}

// solveDebevecChannel builds and solves the Debevec linear system for a single
// channel, returning the recovered log response g(z), z=0..255.
func solveDebevecChannel(images []*cv.Mat, logTimes []float64, pixels []int, c int, lambda float64) []float64 {
	nImg := len(images)
	nSamp := len(pixels)
	nUnknown := 256 + nSamp // g(0..255) then lnE(0..nSamp-1)

	// Number of equations: data terms + smoothness (z=1..254) + 1 constraint.
	nEq := nImg*nSamp + 254 + 1

	// Dense design matrix A (nEq x nUnknown) and RHS b.
	a := make([][]float64, nEq)
	for i := range a {
		a[i] = make([]float64, nUnknown)
	}
	b := make([]float64, nEq)

	row := 0
	ch := images[0].Channels
	// Data-fitting equations.
	for i, pix := range pixels {
		for j := 0; j < nImg; j++ {
			z := int(images[j].Data[pix*ch+c])
			w := hat(z)
			a[row][z] = w
			a[row][256+i] = -w
			b[row] = w * logTimes[j]
			row++
		}
	}
	// Smoothness equations.
	for z := 1; z <= 254; z++ {
		w := lambda * hat(z)
		a[row][z-1] = w
		a[row][z] = -2 * w
		a[row][z+1] = w
		row++
	}
	// Fix the curve at the middle of the range: g(127) = 0.
	a[row][127] = 1

	x := solveLeastSquares(a, b, nUnknown)
	g := make([]float64, 256)
	copy(g, x[:256])
	return g
}

// CalibrateRobertson recovers the camera response with Robertson's iterative
// maximum-likelihood method: it alternates between estimating per-pixel
// radiance from the current response and re-estimating the response from that
// radiance, for the given number of iterations (pass zero for a sensible
// default). It shares the sampling and validation of [CalibrateDebevec] and
// returns a response normalised so that Curve[c][128] = 1.
func CalibrateRobertson(images []*cv.Mat, times []float64, samples, iterations int) (*CameraResponse, error) {
	if err := validateStack(images, times); err != nil {
		return nil, err
	}
	nSamp := samples
	if nSamp <= 0 {
		nSamp = DefaultSamples
	}
	iters := iterations
	if iters <= 0 {
		iters = 15
	}
	rows, cols, ch := images[0].Rows, images[0].Cols, images[0].Channels
	pixels := samplePixels(rows, cols, nSamp)

	curve := make([][]float64, ch)
	for c := 0; c < ch; c++ {
		curve[c] = solveRobertsonChannel(images, times, pixels, c, iters)
	}
	return &CameraResponse{Channels: ch, Curve: curve}, nil
}

// solveRobertsonChannel runs Robertson's alternating optimisation for one
// channel and returns the inverse-response lookup I(z), normalised to I(128)=1.
func solveRobertsonChannel(images []*cv.Mat, times []float64, pixels []int, c, iters int) []float64 {
	nImg := len(images)
	nSamp := len(pixels)
	ch := images[0].Channels

	// I(z): inverse response, initialised linearly to z/128.
	inv := make([]float64, 256)
	for z := 0; z < 256; z++ {
		inv[z] = float64(z) / 128.0
		if inv[z] <= 0 {
			inv[z] = 1e-4
		}
	}
	radiance := make([]float64, nSamp)

	for it := 0; it < iters; it++ {
		// Radiance estimate: weighted least squares over exposures.
		for i, pix := range pixels {
			var num, den float64
			for j := 0; j < nImg; j++ {
				z := int(images[j].Data[pix*ch+c])
				w := hat(z)
				num += w * inv[z] * times[j]
				den += w * times[j] * times[j]
			}
			if den > 0 {
				radiance[i] = num / den
			}
		}
		// Response update: average of E*t over all samples mapping to each z.
		sum := make([]float64, 256)
		cnt := make([]float64, 256)
		for i, pix := range pixels {
			for j := 0; j < nImg; j++ {
				z := int(images[j].Data[pix*ch+c])
				sum[z] += radiance[i] * times[j]
				cnt[z]++
			}
		}
		for z := 0; z < 256; z++ {
			if cnt[z] > 0 {
				inv[z] = sum[z] / cnt[z]
			}
		}
		// Enforce monotonicity by filling gaps and clamping decreases; this
		// stabilises poorly observed bins.
		fillMonotone(inv)
		// Normalise so I(128) = 1.
		mid := inv[128]
		if mid <= 0 {
			mid = 1
		}
		for z := 0; z < 256; z++ {
			inv[z] /= mid
			if inv[z] <= 0 {
				inv[z] = 1e-6
			}
		}
	}
	return inv
}

// fillMonotone repairs an inverse-response table in place: it forward-fills any
// non-positive (unobserved) bins and enforces a non-decreasing sequence.
func fillMonotone(inv []float64) {
	last := 1e-6
	for z := 0; z < len(inv); z++ {
		if inv[z] <= 0 || math.IsNaN(inv[z]) {
			inv[z] = last
		}
		if inv[z] < last {
			inv[z] = last
		}
		last = inv[z]
	}
}

// solveLeastSquares solves the overdetermined system a·x = b in the
// least-squares sense via the normal equations (aᵀa)x = aᵀb, using Gaussian
// elimination with partial pivoting on the n×n normal matrix. A tiny Tikhonov
// term is added to the diagonal for numerical stability.
func solveLeastSquares(a [][]float64, b []float64, n int) []float64 {
	// Normal matrix M = aᵀa (n×n) and rhs r = aᵀb.
	m := make([][]float64, n)
	for i := range m {
		m[i] = make([]float64, n)
	}
	r := make([]float64, n)
	for _, arow := range a {
		// Only visit non-zero entries to keep this affordable for the sparse
		// design matrix.
		nz := make([]int, 0, 4)
		for k := 0; k < n; k++ {
			if arow[k] != 0 {
				nz = append(nz, k)
			}
		}
		for _, i := range nz {
			mi := m[i]
			ai := arow[i]
			for _, k := range nz {
				mi[k] += ai * arow[k]
			}
		}
	}
	for ri, arow := range a {
		if b[ri] == 0 {
			continue
		}
		for k := 0; k < n; k++ {
			if arow[k] != 0 {
				r[k] += arow[k] * b[ri]
			}
		}
	}
	for i := 0; i < n; i++ {
		m[i][i] += 1e-8
	}
	return gaussianSolve(m, r)
}

// gaussianSolve solves m·x = r for x with partial-pivot Gaussian elimination.
func gaussianSolve(m [][]float64, r []float64) []float64 {
	n := len(r)
	for col := 0; col < n; col++ {
		// Partial pivot.
		piv := col
		best := math.Abs(m[col][col])
		for k := col + 1; k < n; k++ {
			if v := math.Abs(m[k][col]); v > best {
				best = v
				piv = k
			}
		}
		if piv != col {
			m[col], m[piv] = m[piv], m[col]
			r[col], r[piv] = r[piv], r[col]
		}
		diag := m[col][col]
		if math.Abs(diag) < 1e-300 {
			continue
		}
		for k := col + 1; k < n; k++ {
			f := m[k][col] / diag
			if f == 0 {
				continue
			}
			mk := m[k]
			mc := m[col]
			for j := col; j < n; j++ {
				mk[j] -= f * mc[j]
			}
			r[k] -= f * r[col]
		}
	}
	x := make([]float64, n)
	for i := n - 1; i >= 0; i-- {
		sum := r[i]
		mi := m[i]
		for j := i + 1; j < n; j++ {
			sum -= mi[j] * x[j]
		}
		if math.Abs(mi[i]) < 1e-300 {
			x[i] = 0
			continue
		}
		x[i] = sum / mi[i]
	}
	return x
}
