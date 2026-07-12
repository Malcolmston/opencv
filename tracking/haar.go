package tracking

import (
	"math"
	"math/rand"

	cv "github.com/malcolmston/opencv"
)

// integralImage holds the summed-area table of a single-channel image, allowing
// the mean intensity of any axis-aligned rectangle to be read in O(1) — the core
// primitive of the Haar-feature classifiers used by [TrackerMIL] and
// [TrackerBoosting].
type integralImage struct {
	rows, cols int
	sum        []float64 // (rows+1)*(cols+1)
}

// newIntegralImage builds the summed-area table of a single-channel Mat.
func newIntegralImage(gray *cv.Mat) *integralImage {
	r, c := gray.Rows, gray.Cols
	ii := &integralImage{rows: r, cols: c, sum: make([]float64, (r+1)*(c+1))}
	stride := c + 1
	for y := 0; y < r; y++ {
		var rowSum float64
		for x := 0; x < c; x++ {
			rowSum += float64(gray.Data[y*c+x])
			ii.sum[(y+1)*stride+(x+1)] = ii.sum[y*stride+(x+1)] + rowSum
		}
	}
	return ii
}

// rectMean returns the mean intensity of the w×h rectangle at (x, y), clamped to
// the image; an out-of-image rectangle contributes only its in-image part.
func (ii *integralImage) rectMean(x, y, w, h int) float64 {
	x0 := clampInt(x, 0, ii.cols)
	y0 := clampInt(y, 0, ii.rows)
	x1 := clampInt(x+w, 0, ii.cols)
	y1 := clampInt(y+h, 0, ii.rows)
	area := float64((x1 - x0) * (y1 - y0))
	if area <= 0 {
		return 0
	}
	stride := ii.cols + 1
	s := ii.sum[y1*stride+x1] - ii.sum[y0*stride+x1] - ii.sum[y1*stride+x0] + ii.sum[y0*stride+x0]
	return s / area
}

// haarRect is one weighted rectangle of a Haar-like feature, positioned relative
// to the patch's top-left corner.
type haarRect struct {
	x, y, w, h int
	weight     float64
}

// haarFeature is a generalised Haar-like feature: a weighted sum of the mean
// intensities of a few rectangles (as in the MILTrack feature set).
type haarFeature struct {
	rects []haarRect
}

// eval returns the feature response for the patch whose top-left corner is at
// (ox, oy) in the integral image.
func (f haarFeature) eval(ii *integralImage, ox, oy int) float64 {
	var v float64
	for _, r := range f.rects {
		v += r.weight * ii.rectMean(ox+r.x, oy+r.y, r.w, r.h)
	}
	return v
}

// generateHaarPool builds n deterministic generalised-Haar features for a
// patchW×patchH patch using the supplied seeded RNG. Each feature sums two or
// three random rectangles with random signed weights, giving a diverse,
// reproducible feature bank.
func generateHaarPool(rng *rand.Rand, n, patchW, patchH int) []haarFeature {
	feats := make([]haarFeature, n)
	for i := 0; i < n; i++ {
		nr := 2 + rng.Intn(2) // 2 or 3 rectangles
		rects := make([]haarRect, nr)
		for j := 0; j < nr; j++ {
			w := 1 + rng.Intn(maxInt(1, patchW/2))
			h := 1 + rng.Intn(maxInt(1, patchH/2))
			x := rng.Intn(maxInt(1, patchW-w+1))
			y := rng.Intn(maxInt(1, patchH-h+1))
			weight := rng.Float64()*2 - 1
			rects[j] = haarRect{x: x, y: y, w: w, h: h, weight: weight}
		}
		feats[i] = haarFeature{rects: rects}
	}
	return feats
}

// maxInt returns the larger of a and b.
func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

// weakClassifier is an online generative weak learner: a Gaussian model of one
// Haar feature's response under the positive and negative classes. Its output is
// the log-likelihood ratio ln p(v|+) − ln p(v|−).
type weakClassifier struct {
	mu1, sig1 float64
	mu0, sig0 float64
	init1     bool
	init0     bool
}

const logSqrt2Pi = 0.9189385332046727 // 0.5*ln(2π)

// updateClass folds a sample value into the running mean and standard deviation
// of one class (label true for positive), with adaptation rate lr.
func (wc *weakClassifier) updateClass(v float64, positive bool, lr float64) {
	if positive {
		if !wc.init1 {
			wc.mu1, wc.sig1, wc.init1 = v, 1, true
			return
		}
		wc.mu1 = (1-lr)*wc.mu1 + lr*v
		d := v - wc.mu1
		wc.sig1 = math.Sqrt((1-lr)*wc.sig1*wc.sig1 + lr*d*d)
		if wc.sig1 < 1e-3 {
			wc.sig1 = 1e-3
		}
	} else {
		if !wc.init0 {
			wc.mu0, wc.sig0, wc.init0 = v, 1, true
			return
		}
		wc.mu0 = (1-lr)*wc.mu0 + lr*v
		d := v - wc.mu0
		wc.sig0 = math.Sqrt((1-lr)*wc.sig0*wc.sig0 + lr*d*d)
		if wc.sig0 < 1e-3 {
			wc.sig0 = 1e-3
		}
	}
}

// logOdds returns the log-likelihood ratio of value v under the two Gaussians.
func (wc *weakClassifier) logOdds(v float64) float64 {
	l1 := -0.5*((v-wc.mu1)/wc.sig1)*((v-wc.mu1)/wc.sig1) - math.Log(wc.sig1) - logSqrt2Pi
	l0 := -0.5*((v-wc.mu0)/wc.sig0)*((v-wc.mu0)/wc.sig0) - math.Log(wc.sig0) - logSqrt2Pi
	return l1 - l0
}

// sampleOffsets returns integer (dx, dy) offsets on a grid whose distance from
// the origin lies in [rMin, rMax]. When rMin is 0 the origin is included. It is
// used to draw positive (small radius) and negative (annulus) training patches.
func sampleOffsets(rMin, rMax, step int) [][2]int {
	var out [][2]int
	for dy := -rMax; dy <= rMax; dy += step {
		for dx := -rMax; dx <= rMax; dx += step {
			d := math.Hypot(float64(dx), float64(dy))
			if d >= float64(rMin) && d <= float64(rMax) {
				out = append(out, [2]int{dx, dy})
			}
		}
	}
	return out
}
