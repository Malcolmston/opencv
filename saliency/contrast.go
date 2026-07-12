package saliency

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// HistogramContrast implements the histogram-based global-contrast salient
// region detector (HC) of Cheng, Mitra, Huang, Torr & Hu, "Global Contrast
// based Salient Region Detection" (CVPR 2011).
//
// Pixel colours are quantised into a small palette. A colour's saliency is the
// sum, over all other palette colours, of that colour's population times its
// Lab-space distance from the query colour:
//
//	S(c) = Σ_j n_j · ‖c − c_j‖
//
// so colours that are both rare and far from the bulk of the image (a distinct
// object against a dominant background) score highest. The per-colour saliency
// is smoothed across nearby palette colours to avoid quantisation artefacts and
// mapped back to the pixels.
//
// Construct one with [NewHistogramContrast]. It satisfies [StaticSaliency].
type HistogramContrast struct {
	// Bins is the number of quantisation levels per channel (Bins³ palette
	// entries). The default is 12.
	Bins int
}

// NewHistogramContrast returns a detector with 12 quantisation levels per
// channel.
func NewHistogramContrast() *HistogramContrast {
	return &HistogramContrast{Bins: 12}
}

// quantizeLab assigns each pixel to a Bins³ Lab palette bin and returns the
// per-pixel bin index, the list of occupied bins with their populations and
// mean Lab colours.
func quantizeLab(img *cv.Mat, bins int) (labelOf []int, pop []float64, cl, ca, cb []float64) {
	l, a, b := labPlanes(img)
	n := l.rows * l.cols
	labelOf = make([]int, n)
	binIndex := make(map[int]int)
	q := func(v float64) int {
		iv := int(v * float64(bins) / 256)
		if iv < 0 {
			iv = 0
		}
		if iv >= bins {
			iv = bins - 1
		}
		return iv
	}
	for i := 0; i < n; i++ {
		key := (q(l.data[i])*bins+q(a.data[i]))*bins + q(b.data[i])
		id, ok := binIndex[key]
		if !ok {
			id = len(pop)
			binIndex[key] = id
			pop = append(pop, 0)
			cl = append(cl, 0)
			ca = append(ca, 0)
			cb = append(cb, 0)
		}
		labelOf[i] = id
		pop[id]++
		cl[id] += l.data[i]
		ca[id] += a.data[i]
		cb[id] += b.data[i]
	}
	for id := range pop {
		cl[id] /= pop[id]
		ca[id] /= pop[id]
		cb[id] /= pop[id]
	}
	return labelOf, pop, cl, ca, cb
}

// ComputeSaliency returns the histogram-contrast saliency map of img: a
// single-channel [cv.Mat] the same size as img, normalised to [0,255]. It
// panics if img is nil or empty.
func (h *HistogramContrast) ComputeSaliency(img *cv.Mat) *cv.Mat {
	bins := h.Bins
	if bins < 2 {
		bins = 12
	}
	labelOf, pop, cl, ca, cb := quantizeLab(img, bins)
	m := len(pop)
	sal := make([]float64, m)
	for i := 0; i < m; i++ {
		var s float64
		for j := 0; j < m; j++ {
			if i == j {
				continue
			}
			dl := cl[i] - cl[j]
			da := ca[i] - ca[j]
			db := cb[i] - cb[j]
			s += pop[j] * math.Sqrt(dl*dl+da*da+db*db)
		}
		sal[i] = s
	}
	smoothColorSaliency(sal, pop, cl, ca, cb)

	out := newPlane(img.Rows, img.Cols)
	for i := range out.data {
		out.data[i] = sal[labelOf[i]]
	}
	return out.normalizedMat()
}

// smoothColorSaliency replaces each colour's saliency with a distance-weighted
// average over its nearest palette colours, damping quantisation noise.
func smoothColorSaliency(sal, pop, cl, ca, cb []float64) {
	m := len(sal)
	if m < 3 {
		return
	}
	kk := m / 4
	if kk < 1 {
		kk = 1
	}
	type nb struct {
		j int
		d float64
	}
	out := make([]float64, m)
	for i := 0; i < m; i++ {
		near := make([]nb, 0, m)
		for j := 0; j < m; j++ {
			dl := cl[i] - cl[j]
			da := ca[i] - ca[j]
			db := cb[i] - cb[j]
			near = append(near, nb{j, math.Sqrt(dl*dl + da*da + db*db)})
		}
		// Partial selection of the kk closest colours.
		for a := 0; a < kk && a < len(near); a++ {
			min := a
			for b := a + 1; b < len(near); b++ {
				if near[b].d < near[min].d {
					min = b
				}
			}
			near[a], near[min] = near[min], near[a]
		}
		var wsum, ssum, dmax float64
		for a := 0; a < kk; a++ {
			if near[a].d > dmax {
				dmax = near[a].d
			}
		}
		if dmax <= 0 {
			out[i] = sal[i]
			continue
		}
		for a := 0; a < kk; a++ {
			w := dmax - near[a].d
			wsum += w
			ssum += w * sal[near[a].j]
		}
		if wsum > 0 {
			out[i] = ssum / wsum
		} else {
			out[i] = sal[i]
		}
	}
	copy(sal, out)
}

// RegionContrast implements the region-based global-contrast salient region
// detector (RC) of Cheng, Mitra, Huang, Torr & Hu, "Global Contrast based
// Salient Region Detection" (CVPR 2011).
//
// The image is divided into regions; each region's saliency is the sum over all
// other regions of a spatial-distance weight times the other region's pixel
// count times the Lab colour distance between the two regions:
//
//	S(r_k) = Σ_{i≠k} exp(−D_s(r_k,r_i)/σ²) · w(r_i) · D_c(r_k,r_i)
//
// The spatial term concentrates contrast contributions from nearby regions, so
// a compact object surrounded by a large uniform background — which contributes
// both high colour distance and high pixel weight — is highlighted while the
// background regions, similar to one another, stay dark.
//
// Regular grid regions stand in for a colour segmentation, keeping the detector
// deterministic; the global-contrast weighting is otherwise faithful to the
// original. Construct one with [NewRegionContrast]. It satisfies
// [StaticSaliency].
type RegionContrast struct {
	// Grid is the number of regions per side (Grid×Grid regions). The default
	// is 12.
	Grid int
	// SpatialSigma is the spatial-distance falloff, as a fraction of the image
	// diagonal. The default is 0.4.
	SpatialSigma float64
}

// NewRegionContrast returns a detector with a 12×12 region grid.
func NewRegionContrast() *RegionContrast {
	return &RegionContrast{Grid: 12, SpatialSigma: 0.4}
}

// ComputeSaliency returns the region-contrast saliency map of img: a
// single-channel [cv.Mat] the same size as img, normalised to [0,255]. It
// panics if img is nil or empty.
func (rc *RegionContrast) ComputeSaliency(img *cv.Mat) *cv.Mat {
	l, a, b := labPlanes(img)
	rows, cols := l.rows, l.cols

	grid := rc.Grid
	if grid < 2 {
		grid = 12
	}
	if grid > rows {
		grid = rows
	}
	if grid > cols {
		grid = cols
	}
	sig := rc.SpatialSigma
	if sig <= 0 {
		sig = 0.4
	}

	n := grid * grid
	cl := make([]float64, n)
	ca := make([]float64, n)
	cb := make([]float64, n)
	cx := make([]float64, n)
	cy := make([]float64, n)
	cnt := make([]float64, n)
	label := make([]int, rows*cols)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			gy := y * grid / rows
			gx := x * grid / cols
			if gy >= grid {
				gy = grid - 1
			}
			if gx >= grid {
				gx = grid - 1
			}
			id := gy*grid + gx
			i := y*cols + x
			label[i] = id
			cl[id] += l.data[i]
			ca[id] += a.data[i]
			cb[id] += b.data[i]
			cx[id] += float64(x)
			cy[id] += float64(y)
			cnt[id]++
		}
	}
	for id := 0; id < n; id++ {
		if cnt[id] > 0 {
			cl[id] /= cnt[id]
			ca[id] /= cnt[id]
			cb[id] /= cnt[id]
			cx[id] /= cnt[id]
			cy[id] /= cnt[id]
		}
	}

	total := float64(rows * cols)
	diag := math.Hypot(float64(rows), float64(cols))
	twoSig2 := 2 * sig * sig
	sal := make([]float64, n)
	for k := 0; k < n; k++ {
		if cnt[k] == 0 {
			continue
		}
		var s float64
		for i := 0; i < n; i++ {
			if i == k || cnt[i] == 0 {
				continue
			}
			dl := cl[k] - cl[i]
			da := ca[k] - ca[i]
			db := cb[k] - cb[i]
			dc := math.Sqrt(dl*dl + da*da + db*db)
			ds := math.Hypot(cx[k]-cx[i], cy[k]-cy[i]) / diag
			w := math.Exp(-ds * ds / twoSig2)
			s += w * (cnt[i] / total) * dc
		}
		sal[k] = s
	}

	out := newPlane(rows, cols)
	for i := range out.data {
		out.data[i] = sal[label[i]]
	}
	out = meanBlur(out, 2)
	return out.normalizedMat()
}
