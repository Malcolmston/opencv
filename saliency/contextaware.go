package saliency

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// StaticSaliencyContextAware implements a context-aware saliency detector after
// Goferman, Zelnik-Manor & Tal, "Context-Aware Saliency Detection" (CVPR 2010).
//
// A region is salient when it is dissimilar in colour to the regions most like
// it, and that dissimilarity is discounted when the similar regions are nearby.
// For every pixel the detector measures a colour dissimilarity to a grid of
// reference locations, weighted down by spatial distance, keeps the K smallest
// such (colour) dissimilarities — the pixel's most similar context — and maps
// their mean d through S = 1 − exp(−d). Pixels whose closest matches are still
// far in colour (a unique object) score high; repetitive background scores low.
//
// The reference set is a fixed sub-sampled grid rather than an exhaustive
// nearest-patch search over multiple scales, which keeps the detector fast and
// deterministic; the qualitative single-scale behaviour matches the original.
//
// Construct one with [NewStaticSaliencyContextAware]. It satisfies
// [StaticSaliency].
type StaticSaliencyContextAware struct {
	// WorkingSize is the side length the image is resampled to before the
	// pairwise comparison (kept small because the cost is quadratic in pixels).
	// The default is 48.
	WorkingSize int
	// K is how many most-similar references contribute to each pixel's score.
	// The default is 32.
	K int
	// PositionWeight controls how strongly spatial proximity discounts colour
	// dissimilarity (larger means distance matters more). The default is 3.
	PositionWeight float64
}

// NewStaticSaliencyContextAware returns a detector with a 48×48 working size and
// K=32 similar references.
func NewStaticSaliencyContextAware() *StaticSaliencyContextAware {
	return &StaticSaliencyContextAware{WorkingSize: 48, K: 32, PositionWeight: 3}
}

// ComputeSaliency returns the context-aware saliency map of img: a
// single-channel [cv.Mat] the same size as img, normalised to [0,255]. It
// panics if img is nil or empty.
func (s *StaticSaliencyContextAware) ComputeSaliency(img *cv.Mat) *cv.Mat {
	l, a, b := labPlanes(img)
	rows, cols := l.rows, l.cols

	ws := s.WorkingSize
	if ws < 8 {
		ws = 48
	}
	wr, wc := ws, ws
	if rows < wr {
		wr = rows
	}
	if cols < wc {
		wc = cols
	}
	sl := resizePlane(l, wr, wc)
	sa := resizePlane(a, wr, wc)
	sb := resizePlane(b, wr, wc)

	k := s.K
	if k < 1 {
		k = 32
	}
	pw := s.PositionWeight
	if pw <= 0 {
		pw = 3
	}
	// Normalise Lab and positions to [0,1] so colour and position combine on a
	// comparable scale.
	const labScale = 1.0 / 255.0
	diag := math.Hypot(float64(wr), float64(wc))

	small := newPlane(wr, wc)
	// Reference grid: sub-sample to keep the pairwise cost bounded.
	stride := 1
	for wr*wc/(stride*stride) > 900 {
		stride++
	}
	for y := 0; y < wr; y++ {
		for x := 0; x < wc; x++ {
			i := y*wc + x
			li, ai, bi := sl.data[i]*labScale, sa.data[i]*labScale, sb.data[i]*labScale
			// Track the K smallest dissimilarities via a small max-heap-free
			// insertion into a fixed slice.
			best := make([]float64, 0, k)
			worst := math.Inf(1)
			for ry := 0; ry < wr; ry += stride {
				for rx := 0; rx < wc; rx += stride {
					if ry == y && rx == x {
						continue
					}
					j := ry*wc + rx
					dl := li - sl.data[j]*labScale
					da := ai - sa.data[j]*labScale
					db := bi - sb.data[j]*labScale
					dcolor := math.Sqrt(dl*dl + da*da + db*db)
					dpos := math.Hypot(float64(y-ry), float64(x-rx)) / diag
					d := dcolor / (1 + pw*dpos)
					if len(best) < k {
						best = append(best, d)
						if d > worst || len(best) == 1 {
							worst = maxSlice(best)
						}
						continue
					}
					if d < worst {
						replaceMax(best, d)
						worst = maxSlice(best)
					}
				}
			}
			var sum float64
			for _, d := range best {
				sum += d
			}
			mean := 0.0
			if len(best) > 0 {
				mean = sum / float64(len(best))
			}
			small.data[i] = 1 - math.Exp(-mean)
		}
	}

	full := resizePlane(small, rows, cols)
	return full.normalizedMat()
}

// maxSlice returns the largest element of a non-empty slice.
func maxSlice(s []float64) float64 {
	m := s[0]
	for _, v := range s[1:] {
		if v > m {
			m = v
		}
	}
	return m
}

// replaceMax overwrites the largest element of s with v (s must be non-empty).
func replaceMax(s []float64, v float64) {
	idx := 0
	for i := 1; i < len(s); i++ {
		if s[i] > s[idx] {
			idx = i
		}
	}
	s[idx] = v
}
