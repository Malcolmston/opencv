package tracking

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// Histogram1D is a one-dimensional intensity histogram over the grayscale range
// [0, 256). It feeds the back-projection pipeline that drives [MeanShift] and
// [CamShift].
type Histogram1D struct {
	// Bins holds one weight per histogram bin; len(Bins) is the bin count.
	Bins []float64
}

// NewHistogram1D allocates a zeroed histogram with the given number of bins,
// which must be between 1 and 256.
func NewHistogram1D(bins int) *Histogram1D {
	if bins < 1 || bins > 256 {
		panic("tracking: NewHistogram1D requires 1..256 bins")
	}
	return &Histogram1D{Bins: make([]float64, bins)}
}

// binOf maps an 8-bit intensity to a bin index.
func (h *Histogram1D) binOf(v uint8) int {
	b := int(v) * len(h.Bins) / 256
	if b >= len(h.Bins) {
		b = len(h.Bins) - 1
	}
	return b
}

// Normalize scales the histogram so its largest bin equals 1. A histogram with
// no mass is left unchanged.
func (h *Histogram1D) Normalize() {
	var m float64
	for _, b := range h.Bins {
		if b > m {
			m = b
		}
	}
	if m <= 0 {
		return
	}
	for i := range h.Bins {
		h.Bins[i] /= m
	}
}

// CalcHistGray builds an intensity histogram of the grayscale of img over the
// pixels inside roi. Passing an empty roi uses the whole image. Multi-channel
// input is converted to grayscale. The number of bins must be 1..256.
func CalcHistGray(img *cv.Mat, roi Rect, bins int) *Histogram1D {
	if img == nil || img.Empty() {
		panic("tracking: CalcHistGray requires a non-empty image")
	}
	g := trackingToGrayF(img)
	h := NewHistogram1D(bins)
	r := roi
	if r.Empty() {
		r = Rect{X: 0, Y: 0, Width: img.Cols, Height: img.Rows}
	}
	r = r.clampTo(img.Rows, img.Cols)
	for y := r.Y; y < r.Bottom(); y++ {
		for x := r.X; x < r.Right(); x++ {
			h.Bins[h.binOf(clampU8(g.at(y, x)))]++
		}
	}
	return h
}

// CalcBackProjection produces a probability image the same size as img in which
// each pixel holds the (normalised) histogram weight of its grayscale value,
// scaled to the 8-bit range [0, 255]. High values mark pixels whose intensity
// is well represented in the model histogram; this is the density image that
// [MeanShift] and [CamShift] climb. Multi-channel input is converted to
// grayscale.
func CalcBackProjection(img *cv.Mat, hist *Histogram1D) *cv.Mat {
	if img == nil || img.Empty() {
		panic("tracking: CalcBackProjection requires a non-empty image")
	}
	if hist == nil || len(hist.Bins) == 0 {
		panic("tracking: CalcBackProjection requires a non-empty histogram")
	}
	// Work on a copy scaled so the peak bin maps to 255.
	var peak float64
	for _, b := range hist.Bins {
		if b > peak {
			peak = b
		}
	}
	g := trackingToGrayF(img)
	out := cv.NewMat(img.Rows, img.Cols, 1)
	for y := 0; y < img.Rows; y++ {
		for x := 0; x < img.Cols; x++ {
			bi := hist.binOf(clampU8(g.at(y, x)))
			var p float64
			if peak > 0 {
				p = hist.Bins[bi] / peak * 255
			}
			out.Data[y*img.Cols+x] = clampU8(p)
		}
	}
	return out
}

// windowMoments accumulates the zeroth and first-order weighted moments of a
// single-channel probability image over window w.
func windowMoments(prob *cv.Mat, w Rect) (m00, m10, m01 float64) {
	r := w.clampTo(prob.Rows, prob.Cols)
	for y := r.Y; y < r.Bottom(); y++ {
		row := y * prob.Cols
		for x := r.X; x < r.Right(); x++ {
			p := float64(prob.Data[row+x])
			m00 += p
			m10 += p * float64(x)
			m01 += p * float64(y)
		}
	}
	return
}

// MeanShift iterates a search window towards the mode (centroid of probability
// mass) of a single-channel probability image, the output of
// [CalcBackProjection]. On each iteration the window is recentred on the weighted
// centroid of the pixels it covers; iteration stops when the centre shift falls
// below crit.Epsilon or crit.MaxCount iterations have run.
//
// It returns the converged window (its size unchanged from the input) and the
// number of iterations performed. prob must be a non-empty single-channel Mat.
// The result is deterministic.
func MeanShift(prob *cv.Mat, window Rect, crit TermCriteria) (result Rect, iterations int) {
	if prob == nil || prob.Empty() || prob.Channels != 1 {
		panic("tracking: MeanShift requires a non-empty single-channel probability image")
	}
	if window.Empty() {
		panic("tracking: MeanShift requires a non-empty window")
	}
	w := window
	maxIter := crit.iterCap(10)
	for iter := 0; iter < maxIter; iter++ {
		iterations = iter + 1
		m00, m10, m01 := windowMoments(prob, w)
		if m00 <= 0 {
			break
		}
		cx := m10 / m00
		cy := m01 / m00
		newX := int(math.Round(cx - float64(w.Width)/2))
		newY := int(math.Round(cy - float64(w.Height)/2))
		// Keep the window inside the image.
		newX = clampInt(newX, 0, prob.Cols-w.Width)
		newY = clampInt(newY, 0, prob.Rows-w.Height)
		shift := math.Hypot(float64(newX-w.X), float64(newY-w.Y))
		w.X = newX
		w.Y = newY
		if crit.Epsilon > 0 && shift <= crit.Epsilon {
			break
		}
	}
	return w, iterations
}

// CamShift is the Continuously Adaptive Mean Shift tracker. It first runs
// [MeanShift] to locate the mode, then adapts the window size to the amount of
// probability mass found and estimates the target orientation from the
// second-order central moments of the probability image, yielding an oriented
// bounding box.
//
// It returns the oriented result, the adapted axis-aligned window and the number
// of mean-shift iterations performed. prob must be a non-empty single-channel
// Mat. The result is deterministic.
func CamShift(prob *cv.Mat, window Rect, crit TermCriteria) (RotatedRect, Rect, int) {
	if prob == nil || prob.Empty() || prob.Channels != 1 {
		panic("tracking: CamShift requires a non-empty single-channel probability image")
	}
	w, iters := MeanShift(prob, window, crit)

	// Adapt window size from the zeroth moment (total mass): s = 2*sqrt(m00/256).
	m00, m10, m01 := windowMoments(prob, w)
	if m00 <= 0 {
		return RotatedRect{Center: w.Center(), Width: float64(w.Width), Height: float64(w.Height)}, w, iters
	}
	cx := m10 / m00
	cy := m01 / m00
	side := 2 * math.Sqrt(m00/256)
	half := int(math.Round(side / 2))
	if half < 1 {
		half = 1
	}
	adapted := Rect{
		X:      clampInt(int(math.Round(cx))-half, 0, prob.Cols-1),
		Y:      clampInt(int(math.Round(cy))-half, 0, prob.Rows-1),
		Width:  2 * half,
		Height: 2 * half,
	}
	adapted = adapted.clampTo(prob.Rows, prob.Cols)

	// Second-order central moments for orientation.
	var mu20, mu02, mu11 float64
	r := adapted
	for y := r.Y; y < r.Bottom(); y++ {
		row := y * prob.Cols
		for x := r.X; x < r.Right(); x++ {
			p := float64(prob.Data[row+x])
			dx := float64(x) - cx
			dy := float64(y) - cy
			mu20 += p * dx * dx
			mu02 += p * dy * dy
			mu11 += p * dx * dy
		}
	}
	mu20 /= m00
	mu02 /= m00
	mu11 /= m00
	angle := 0.5 * math.Atan2(2*mu11, mu20-mu02)
	// Axis lengths from the moment matrix eigenvalues.
	common := math.Sqrt(math.Max(0, (mu20-mu02)*(mu20-mu02)+4*mu11*mu11))
	l1 := (mu20 + mu02 + common) / 2
	l2 := (mu20 + mu02 - common) / 2
	rr := RotatedRect{
		Center: Point2f{X: cx, Y: cy},
		Width:  4 * math.Sqrt(math.Max(0, l1)),
		Height: 4 * math.Sqrt(math.Max(0, l2)),
		Angle:  angle * 180 / math.Pi,
	}
	return rr, adapted, iters
}

// clampInt clamps v into [lo, hi]; if hi < lo it returns lo.
func clampInt(v, lo, hi int) int {
	if hi < lo {
		return lo
	}
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}
