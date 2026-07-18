package segment2

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// GrabCut mask states, matching OpenCV's cv::GrabCutClasses. They label a pixel
// as definite or probable foreground/background in [GrabCutWithMask].
const (
	// GCBackground marks a pixel as definite background.
	GCBackground = 0
	// GCForeground marks a pixel as definite foreground.
	GCForeground = 1
	// GCProbablyBackground marks a pixel as probable background.
	GCProbablyBackground = 2
	// GCProbablyForeground marks a pixel as probable foreground.
	GCProbablyForeground = 3
)

// segment2gmmComponent is one diagonal-covariance Gaussian of a mixture.
type segment2gmmComponent struct {
	weight float64
	mean   []float64
	varc   []float64
}

// segment2gmm is a Gaussian mixture over colour vectors with diagonal
// covariance, fitted deterministically by k-means.
type segment2gmm struct {
	comps []segment2gmmComponent
	ch    int
}

// segment2fitGMM fits a k-component diagonal-covariance GMM to the given colour
// samples using deterministic k-means for the hard component assignment.
func segment2fitGMM(samples [][]float64, k, ch int) *segment2gmm {
	g := &segment2gmm{ch: ch}
	if len(samples) == 0 {
		// Degenerate: a single broad component centred at mid-grey.
		mean := make([]float64, ch)
		varc := make([]float64, ch)
		for c := 0; c < ch; c++ {
			mean[c] = 128
			varc[c] = 128 * 128
		}
		g.comps = []segment2gmmComponent{{weight: 1, mean: mean, varc: varc}}
		return g
	}
	if k > len(samples) {
		k = len(samples)
	}
	assign, centers, _ := segment2kmeans(samples, k, 10)
	total := float64(len(samples))
	for ci := 0; ci < k; ci++ {
		mean := centers[ci]
		varc := make([]float64, ch)
		cnt := 0
		for i, a := range assign {
			if a != ci {
				continue
			}
			cnt++
			for c := 0; c < ch; c++ {
				d := samples[i][c] - mean[c]
				varc[c] += d * d
			}
		}
		if cnt == 0 {
			continue
		}
		for c := 0; c < ch; c++ {
			varc[c] = varc[c]/float64(cnt) + 1.0 // regularise to avoid zero variance
		}
		g.comps = append(g.comps, segment2gmmComponent{
			weight: float64(cnt) / total,
			mean:   append([]float64(nil), mean...),
			varc:   varc,
		})
	}
	if len(g.comps) == 0 {
		mean := make([]float64, ch)
		varc := make([]float64, ch)
		for c := 0; c < ch; c++ {
			mean[c] = 128
			varc[c] = 128 * 128
		}
		g.comps = []segment2gmmComponent{{weight: 1, mean: mean, varc: varc}}
	}
	return g
}

// prob returns the mixture probability density at colour x.
func (g *segment2gmm) prob(x []float64) float64 {
	var p float64
	for _, comp := range g.comps {
		d := 1.0
		e := 0.0
		for c := 0; c < g.ch; c++ {
			diff := x[c] - comp.mean[c]
			e += (diff * diff) / (2 * comp.varc[c])
			d *= 2 * math.Pi * comp.varc[c]
		}
		p += comp.weight * math.Exp(-e) / math.Sqrt(d)
	}
	if p < 1e-300 {
		p = 1e-300
	}
	return p
}

// GrabCutWithMask runs GrabCut foreground extraction driven by an initial mask.
// mask is a flat row-major []byte of length img.Rows*img.Cols holding
// [GCBackground], [GCForeground], [GCProbablyBackground] or
// [GCProbablyForeground] states. Each iteration fits foreground/background
// Gaussian mixtures to the current labelling and finds the global minimum cut of
// the GrabCut energy with a real max-flow ([FlowGraph.MaxFlow]), relabelling the
// probable pixels; definite pixels are never changed. It returns the refined
// mask (a new slice). numComponents sets the GMM size per class (5 is typical).
//
// It panics if img is empty, mask has the wrong length, iterations < 1 or
// numComponents < 1.
func GrabCutWithMask(img *cv.Mat, mask []byte, iterations, numComponents int) []byte {
	segment2requireNonEmpty(img, "GrabCutWithMask")
	rows, cols, ch := img.Rows, img.Cols, img.Channels
	n := rows * cols
	if len(mask) != n {
		panic("segment2: GrabCutWithMask mask size mismatch")
	}
	if iterations < 1 {
		panic("segment2: GrabCutWithMask requires iterations >= 1")
	}
	if numComponents < 1 {
		panic("segment2: GrabCutWithMask requires numComponents >= 1")
	}
	out := make([]byte, n)
	copy(out, mask)

	colors := make([][]float64, n)
	for i := 0; i < n; i++ {
		colors[i] = segment2colorAt(img, i%cols, i/cols)
	}

	// beta: 1 / (2 * mean squared colour gradient over 4-neighbours).
	var betaSum float64
	var betaCnt int
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			i := y*cols + x
			if x+1 < cols {
				betaSum += segment2colorDist2(colors[i], colors[i+1])
				betaCnt++
			}
			if y+1 < rows {
				betaSum += segment2colorDist2(colors[i], colors[i+cols])
				betaCnt++
			}
		}
	}
	beta := 0.0
	if betaSum > 0 {
		beta = 1.0 / (2 * betaSum / float64(betaCnt))
	}
	const gamma = 50.0
	lambda := 9*gamma + 1

	source := n
	sink := n + 1

	nlink := func(a, b int) float64 {
		return gamma * math.Exp(-beta*segment2colorDist2(colors[a], colors[b]))
	}

	for it := 0; it < iterations; it++ {
		var fg, bg [][]float64
		for i := 0; i < n; i++ {
			switch out[i] {
			case GCForeground, GCProbablyForeground:
				fg = append(fg, colors[i])
			default:
				bg = append(bg, colors[i])
			}
		}
		fgGMM := segment2fitGMM(fg, numComponents, ch)
		bgGMM := segment2fitGMM(bg, numComponents, ch)

		g := NewFlowGraph(n+2, source, sink)
		for i := 0; i < n; i++ {
			var fromSrc, toSink float64
			switch out[i] {
			case GCBackground:
				fromSrc, toSink = 0, lambda
			case GCForeground:
				fromSrc, toSink = lambda, 0
			default:
				fromSrc = -math.Log(bgGMM.prob(colors[i]))
				toSink = -math.Log(fgGMM.prob(colors[i]))
			}
			if fromSrc > 0 {
				g.AddEdge(source, i, fromSrc, 0)
			}
			if toSink > 0 {
				g.AddEdge(i, sink, toSink, 0)
			}
		}
		for y := 0; y < rows; y++ {
			for x := 0; x < cols; x++ {
				i := y*cols + x
				if x+1 < cols {
					w := nlink(i, i+1)
					g.AddEdge(i, i+1, w, w)
				}
				if y+1 < rows {
					w := nlink(i, i+cols)
					g.AddEdge(i, i+cols, w, w)
				}
			}
		}
		_, reach := g.MaxFlow()
		for i := 0; i < n; i++ {
			if out[i] == GCBackground || out[i] == GCForeground {
				continue
			}
			if reach[i] {
				out[i] = GCProbablyForeground
			} else {
				out[i] = GCProbablyBackground
			}
		}
	}
	return out
}

// GrabCut segments the foreground object contained in rect from img. Pixels
// outside rect are initialised as definite background and pixels inside as
// probable foreground, then [GrabCutWithMask] is run for the given number of
// iterations with five-component GMMs. It returns a single-channel [cv.Mat] mask
// that is 255 on foreground pixels and 0 on background.
//
// It panics if img is empty, rect is empty or degenerate, or iterations < 1.
func GrabCut(img *cv.Mat, rect cv.Rect, iterations int) *cv.Mat {
	segment2requireNonEmpty(img, "GrabCut")
	rows, cols := img.Rows, img.Cols
	if rect.Width <= 0 || rect.Height <= 0 {
		panic("segment2: GrabCut requires a non-empty rect")
	}
	mask := make([]byte, rows*cols)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			if x >= rect.X && x < rect.X+rect.Width && y >= rect.Y && y < rect.Y+rect.Height {
				mask[y*cols+x] = GCProbablyForeground
			} else {
				mask[y*cols+x] = GCBackground
			}
		}
	}
	refined := GrabCutWithMask(img, mask, iterations, 5)
	out := cv.NewMat(rows, cols, 1)
	for i, m := range refined {
		if m == GCForeground || m == GCProbablyForeground {
			out.Data[i] = 255
		}
	}
	return out
}
