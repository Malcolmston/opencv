package segmentation

import (
	"fmt"
	"math"

	cv "github.com/malcolmston/opencv"
)

// GrabCut mask codes, matching cv2.GC_* so that mask&1 yields the binary
// foreground segmentation (background and probable-background have their low bit
// clear, foreground and probable-foreground have it set).
const (
	// GcBgd marks a pixel as definite background (never relabelled).
	GcBgd = 0
	// GcFgd marks a pixel as definite foreground (never relabelled).
	GcFgd = 1
	// GcPrBgd marks a pixel as probable background (updated each iteration).
	GcPrBgd = 2
	// GcPrFgd marks a pixel as probable foreground (updated each iteration).
	GcPrFgd = 3
)

// grabCutComponents is the number of Gaussian components per colour model,
// matching OpenCV's default GrabCut mixture size.
const grabCutComponents = 5

// GrabCut segments the three-channel image img into foreground and background
// using an iterative Gaussian-mixture colour model, returning a new
// single-channel mask of [GcBgd], [GcFgd], [GcPrBgd] and [GcPrFgd] codes. Read
// the binary result with mask&1 (1 == foreground).
//
// Initialisation follows cv2.grabCut with GC_INIT_WITH_RECT: pixels outside rect
// become definite background ([GcBgd], fixed for the whole run) and pixels
// inside become probable foreground ([GcPrFgd]). If mask is non-nil and matches
// the image size its channel-0 values are taken as an existing labelling
// (GC_INIT_WITH_MASK); [GcBgd] and [GcFgd] entries are treated as hard
// constraints and the probable labels are refined.
//
// Each of the iters iterations:
//
//  1. Fits a [grabCutComponents]-component Gaussian mixture (diagonal
//     covariance) to the current foreground pixels and another to the
//     background pixels, using deterministic k-means clustering.
//  2. Relabels every non-fixed pixel by iterated conditional modes (ICM): the
//     data term is the negative log-likelihood under each mixture and the
//     smoothness term is a Potts penalty over the 4-neighbourhood.
//
// # Graph-cut approximation
//
// The reference GrabCut minimises its Gibbs energy with a global s-t min-cut.
// This implementation instead approximates that step with ICM, a local
// coordinate-descent minimiser of the same data-plus-smoothness energy. ICM is
// deterministic and dependency-free but only reaches a local optimum, so results
// can differ from a true max-flow solver on ambiguous images; on inputs with a
// clearly separated foreground colour the two agree closely.
//
// It panics if img is not three-channel, if img is empty, if rect is degenerate
// when used for initialisation, or if mask (when supplied) does not match img.
func GrabCut(img *cv.Mat, mask *cv.Mat, rect cv.Rect, iters int) *cv.Mat {
	if img.Empty() {
		panic("segmentation: GrabCut on empty image")
	}
	if img.Channels != 3 {
		panic(fmt.Sprintf("segmentation: GrabCut requires a 3-channel image, got %d channels", img.Channels))
	}
	rows, cols := img.Rows, img.Cols
	n := rows * cols
	if iters < 1 {
		iters = 1
	}

	label := make([]uint8, n) // GC_* codes
	useMask := mask != nil && !mask.Empty()
	if useMask {
		if mask.Rows != rows || mask.Cols != cols {
			panic("segmentation: GrabCut mask must match the image dimensions")
		}
		for i := 0; i < n; i++ {
			v := mask.Data[i*mask.Channels]
			if v > GcPrFgd {
				v = GcPrFgd
			}
			label[i] = v
		}
	} else {
		x0 := clampInt(rect.X, 0, cols)
		y0 := clampInt(rect.Y, 0, rows)
		x1 := clampInt(rect.X+rect.Width, 0, cols)
		y1 := clampInt(rect.Y+rect.Height, 0, rows)
		if x1 <= x0 || y1 <= y0 {
			panic(fmt.Sprintf("segmentation: GrabCut rect %+v is empty within the image", rect))
		}
		for y := y0; y < y1; y++ {
			for x := x0; x < x1; x++ {
				label[y*cols+x] = GcPrFgd
			}
		}
	}

	// Colour samples as float triples for model fitting and scoring.
	colors := make([][3]float64, n)
	for i := 0; i < n; i++ {
		b := i * 3
		colors[i] = [3]float64{
			float64(img.Data[b+0]),
			float64(img.Data[b+1]),
			float64(img.Data[b+2]),
		}
	}

	isFg := func(l uint8) bool { return l == GcFgd || l == GcPrFgd }

	for it := 0; it < iters; it++ {
		var fgIdx, bgIdx []int
		for i := 0; i < n; i++ {
			if isFg(label[i]) {
				fgIdx = append(fgIdx, i)
			} else {
				bgIdx = append(bgIdx, i)
			}
		}
		if len(fgIdx) == 0 || len(bgIdx) == 0 {
			break // Nothing to separate; keep the current labelling.
		}
		fgModel := fitGMM(colors, fgIdx, grabCutComponents)
		bgModel := fitGMM(colors, bgIdx, grabCutComponents)

		// Precompute per-pixel data costs (negative log-likelihood).
		fgCost := make([]float64, n)
		bgCost := make([]float64, n)
		for i := 0; i < n; i++ {
			fgCost[i] = fgModel.negLogLik(colors[i])
			bgCost[i] = bgModel.negLogLik(colors[i])
		}
		icmRefine(label, fgCost, bgCost, rows, cols)
	}

	out := cv.NewMat(rows, cols, 1)
	copy(out.Data, label)
	return out
}

// gamma weights the Potts smoothness term relative to the data term in the ICM
// energy. Larger values favour spatially coherent regions.
const gamma = 8.0

// icmRefine performs iterated conditional modes on the soft-labelled pixels,
// minimising the sum of the data cost and a Potts smoothness penalty over the
// 4-neighbourhood. Fixed GcBgd/GcFgd pixels are never changed.
func icmRefine(label []uint8, fgCost, bgCost []float64, rows, cols int) {
	const sweeps = 4
	for s := 0; s < sweeps; s++ {
		changed := false
		for y := 0; y < rows; y++ {
			for x := 0; x < cols; x++ {
				idx := y*cols + x
				l := label[idx]
				if l == GcBgd || l == GcFgd {
					continue
				}
				// Smoothness cost of labelling this pixel fg vs bg, counting
				// disagreeing 4-neighbours.
				var fgSmooth, bgSmooth float64
				for _, o := range neighbors4 {
					nx, ny := x+o.dx, y+o.dy
					if nx < 0 || nx >= cols || ny < 0 || ny >= rows {
						continue
					}
					nb := label[ny*cols+nx]
					if nb == GcFgd || nb == GcPrFgd {
						bgSmooth += gamma // neighbour is fg, so bg here disagrees
					} else {
						fgSmooth += gamma // neighbour is bg, so fg here disagrees
					}
				}
				fgEnergy := fgCost[idx] + fgSmooth
				bgEnergy := bgCost[idx] + bgSmooth
				var nl uint8 = GcPrBgd
				if fgEnergy < bgEnergy {
					nl = GcPrFgd
				}
				if nl != l {
					label[idx] = nl
					changed = true
				}
			}
		}
		if !changed {
			break
		}
	}
}

// gmm is a diagonal-covariance Gaussian mixture over RGB colour.
type gmm struct {
	weight []float64
	mean   [][3]float64
	varc   [][3]float64 // per-channel variance
}

// minVariance floors each channel variance to keep the Gaussians well-defined
// even for near-constant clusters.
const minVariance = 1.0

// fitGMM clusters the colours at the given indices into k components with
// deterministic k-means and returns the resulting diagonal-covariance mixture.
// k is reduced when there are fewer samples than components.
func fitGMM(colors [][3]float64, idx []int, k int) gmm {
	if k > len(idx) {
		k = len(idx)
	}
	if k < 1 {
		k = 1
	}
	assign, centers := kMeans(colors, idx, k)

	g := gmm{
		weight: make([]float64, k),
		mean:   make([][3]float64, k),
		varc:   make([][3]float64, k),
	}
	counts := make([]int, k)
	for ii, i := range idx {
		c := assign[ii]
		counts[c]++
		for d := 0; d < 3; d++ {
			g.mean[c][d] += colors[i][d]
		}
	}
	for c := 0; c < k; c++ {
		if counts[c] == 0 {
			g.mean[c] = centers[c]
			for d := 0; d < 3; d++ {
				g.varc[c][d] = minVariance
			}
			continue
		}
		for d := 0; d < 3; d++ {
			g.mean[c][d] /= float64(counts[c])
		}
	}
	for ii, i := range idx {
		c := assign[ii]
		for d := 0; d < 3; d++ {
			diff := colors[i][d] - g.mean[c][d]
			g.varc[c][d] += diff * diff
		}
	}
	for c := 0; c < k; c++ {
		if counts[c] == 0 {
			continue
		}
		g.weight[c] = float64(counts[c]) / float64(len(idx))
		for d := 0; d < 3; d++ {
			g.varc[c][d] = g.varc[c][d]/float64(counts[c]) + minVariance
		}
	}
	return g
}

// negLogLik returns the negative log-likelihood of colour under the mixture,
// used as the ICM data term. Empty mixtures score a large constant.
func (g gmm) negLogLik(color [3]float64) float64 {
	if len(g.weight) == 0 {
		return 1e6
	}
	var p float64
	for c := range g.weight {
		if g.weight[c] == 0 {
			continue
		}
		det := 1.0
		exponent := 0.0
		for d := 0; d < 3; d++ {
			v := g.varc[c][d]
			det *= v
			diff := color[d] - g.mean[c][d]
			exponent += diff * diff / v
		}
		// (2*pi)^(3/2) constant omitted: it is identical for both classes and
		// cancels when the fg and bg data terms are compared.
		density := g.weight[c] / math.Sqrt(det) * math.Exp(-0.5*exponent)
		p += density
	}
	if p <= 0 {
		return 1e6
	}
	return -math.Log(p)
}

// kMeans runs deterministic Lloyd's iteration on the colours at idx. Initial
// centres are chosen at evenly spaced positions along idx, so the result depends
// only on the data. It returns each sample's cluster (aligned with idx) and the
// final centres.
func kMeans(colors [][3]float64, idx []int, k int) (assign []int, centers [][3]float64) {
	centers = make([][3]float64, k)
	for c := 0; c < k; c++ {
		pos := 0
		if k > 1 {
			pos = c * (len(idx) - 1) / (k - 1)
		}
		centers[c] = colors[idx[pos]]
	}
	assign = make([]int, len(idx))
	const maxIter = 12
	for iter := 0; iter < maxIter; iter++ {
		changed := false
		for ii, i := range idx {
			best, bestD := 0, math.MaxFloat64
			for c := 0; c < k; c++ {
				d := sqDist3(colors[i], centers[c])
				if d < bestD {
					bestD, best = d, c
				}
			}
			if assign[ii] != best {
				assign[ii] = best
				changed = true
			}
		}
		sum := make([][3]float64, k)
		cnt := make([]int, k)
		for ii, i := range idx {
			c := assign[ii]
			cnt[c]++
			for d := 0; d < 3; d++ {
				sum[c][d] += colors[i][d]
			}
		}
		for c := 0; c < k; c++ {
			if cnt[c] == 0 {
				continue // keep previous centre for an empty cluster
			}
			for d := 0; d < 3; d++ {
				centers[c][d] = sum[c][d] / float64(cnt[c])
			}
		}
		if !changed {
			break
		}
	}
	return assign, centers
}

func sqDist3(a, b [3]float64) float64 {
	var s float64
	for d := 0; d < 3; d++ {
		diff := a[d] - b[d]
		s += diff * diff
	}
	return s
}

func clampInt(v, lo, hi int) int {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}
