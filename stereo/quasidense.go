package stereo

import (
	"container/heap"
	"fmt"
	"math"

	cv "github.com/malcolmston/opencv"
)

// QuasiDenseStereo implements quasi-dense stereo matching by seed-and-grow
// propagation, mirroring cv::stereo::QuasiDenseStereo. A sparse set of reliable
// seed correspondences is found on a regular grid by maximising the normalised
// cross-correlation (ZNCC) over the search range; the seeds are then grown in
// best-first order (highest correlation propagated first) into their
// neighbourhood, admitting a new match only when its disparity stays within
// DisparityGradient of the parent and its ZNCC clears CorrThreshold. Because
// propagation follows the most confident matches, the method spreads dense
// disparities across textured surfaces while refusing to cross depth
// discontinuities.
//
// The zero value is not useful; set at least NumDisparities. Remaining fields
// default when non-positive.
type QuasiDenseStereo struct {
	// MinDisparity is the smallest disparity searched (usually 0).
	MinDisparity int
	// NumDisparities is the width of the search range for seeds. Defaults to 16.
	NumDisparities int
	// CorrWinSize is the odd side of the ZNCC correlation window. Defaults to 5.
	CorrWinSize int
	// CorrThreshold is the minimum ZNCC (in [-1, 1]) for a match to be accepted.
	// Defaults to 0.5.
	CorrThreshold float64
	// MinVariance is the minimum left-window intensity variance; flatter windows
	// are skipped as untextured. Defaults to 25.
	MinVariance float64
	// DisparityGradient is the maximum disparity change allowed between a pixel and
	// its propagated neighbour. Defaults to 1.
	DisparityGradient int
	// SeedStep is the grid spacing at which seeds are sought. Defaults to
	// CorrWinSize.
	SeedStep int
}

// growCandidate is a queued propagation event: assign disparity D to pixel
// (Y, X) with correlation Corr. Seq breaks ties deterministically.
type growCandidate struct {
	Corr float64
	Y, X int
	D    int
	Seq  int
}

// candidateHeap is a max-heap on Corr (ties broken by insertion order) so that
// the most confident correspondence is always propagated next.
type candidateHeap []growCandidate

func (h candidateHeap) Len() int { return len(h) }
func (h candidateHeap) Less(i, j int) bool {
	if h[i].Corr != h[j].Corr {
		return h[i].Corr > h[j].Corr
	}
	return h[i].Seq < h[j].Seq
}
func (h candidateHeap) Swap(i, j int) { h[i], h[j] = h[j], h[i] }
func (h *candidateHeap) Push(x any)   { *h = append(*h, x.(growCandidate)) }
func (h *candidateHeap) Pop() any {
	old := *h
	n := len(old)
	item := old[n-1]
	*h = old[:n-1]
	return item
}

// Process matches left against right and returns a single-channel 8-bit
// quasi-dense disparity map; unmatched pixels hold [InvalidDisparity]. Inputs
// may be single- or three-channel and must share dimensions.
//
// It panics on empty input, a size or channel mismatch, or an even CorrWinSize.
func (q QuasiDenseStereo) Process(left, right *cv.Mat) *cv.Mat {
	minD := q.MinDisparity
	numD := q.NumDisparities
	if numD <= 0 {
		numD = 16
	}
	win := q.CorrWinSize
	if win <= 0 {
		win = 5
	}
	requireOdd(win, "QuasiDenseStereo.CorrWinSize")
	corrThresh := q.CorrThreshold
	if corrThresh <= 0 {
		corrThresh = 0.5
	}
	minVar := q.MinVariance
	if minVar <= 0 {
		minVar = 25
	}
	grad := q.DisparityGradient
	if grad <= 0 {
		grad = 1
	}
	step := q.SeedStep
	if step <= 0 {
		step = win
	}

	rows, cols, gl := toGrayGrid(left)
	rrows, rcols, gr := toGrayGrid(right)
	if rows != rrows || cols != rcols {
		panic(fmt.Sprintf("stereo: QuasiDenseStereo.Process size mismatch left %dx%d right %dx%d",
			rows, cols, rrows, rcols))
	}
	half := win / 2

	assigned := make([]bool, rows*cols)
	out := cv.NewMat(rows, cols, 1)

	h := &candidateHeap{}
	heap.Init(h)
	seq := 0

	// Seed search on a regular grid.
	for y := half; y < rows-half; y += step {
		for x := minD + numD - 1 + half; x < cols-half; x += step {
			if windowVariance(gl, rows, cols, y, x, half) < minVar {
				continue
			}
			bestD, bestCorr := -1, corrThresh
			for idx := 0; idx < numD; idx++ {
				d := minD + idx
				if x-d-half < 0 {
					break
				}
				c := zncc(gl, gr, rows, cols, y, x, x-d, half)
				if c > bestCorr {
					bestCorr, bestD = c, d
				}
			}
			if bestD >= 0 {
				heap.Push(h, growCandidate{Corr: bestCorr, Y: y, X: x, D: bestD, Seq: seq})
				seq++
			}
		}
	}

	// Best-first propagation.
	for h.Len() > 0 {
		cand := heap.Pop(h).(growCandidate)
		p := cand.Y*cols + cand.X
		if assigned[p] {
			continue
		}
		assigned[p] = true
		out.Data[p] = uint8(clampInt(cand.D, 0, 255))

		for ny := cand.Y - 1; ny <= cand.Y+1; ny++ {
			if ny < half || ny >= rows-half {
				continue
			}
			for nx := cand.X - 1; nx <= cand.X+1; nx++ {
				if nx < half || nx >= cols-half {
					continue
				}
				np := ny*cols + nx
				if assigned[np] {
					continue
				}
				if windowVariance(gl, rows, cols, ny, nx, half) < minVar {
					continue
				}
				bestD, bestCorr := -1, corrThresh
				for dd := cand.D - grad; dd <= cand.D+grad; dd++ {
					if dd < minD || dd >= minD+numD || nx-dd-half < 0 {
						continue
					}
					c := zncc(gl, gr, rows, cols, ny, nx, nx-dd, half)
					if c > bestCorr {
						bestCorr, bestD = c, dd
					}
				}
				if bestD >= 0 {
					heap.Push(h, growCandidate{Corr: bestCorr, Y: ny, X: nx, D: bestD, Seq: seq})
					seq++
				}
			}
		}
	}
	return out
}

// windowVariance returns the intensity variance of the window centred at (y, x).
func windowVariance(g []int, rows, cols, y, x, half int) float64 {
	var sum, sumSq float64
	n := 0
	for dy := -half; dy <= half; dy++ {
		yy := clampInt(y+dy, 0, rows-1)
		rowBase := yy * cols
		for dx := -half; dx <= half; dx++ {
			xx := clampInt(x+dx, 0, cols-1)
			v := float64(g[rowBase+xx])
			sum += v
			sumSq += v * v
			n++
		}
	}
	mean := sum / float64(n)
	return sumSq/float64(n) - mean*mean
}

// zncc returns the zero-mean normalised cross-correlation in [-1, 1] between the
// left window centred at (y, xL) and the right window centred at (y, xR). A
// window with zero variance on either side returns -1.
func zncc(gl, gr []int, rows, cols, y, xL, xR, half int) float64 {
	var sumL, sumR float64
	n := 0
	for dy := -half; dy <= half; dy++ {
		yy := clampInt(y+dy, 0, rows-1)
		rowBase := yy * cols
		for dx := -half; dx <= half; dx++ {
			lx := clampInt(xL+dx, 0, cols-1)
			rx := clampInt(xR+dx, 0, cols-1)
			sumL += float64(gl[rowBase+lx])
			sumR += float64(gr[rowBase+rx])
			n++
		}
	}
	meanL := sumL / float64(n)
	meanR := sumR / float64(n)
	var num, varL, varR float64
	for dy := -half; dy <= half; dy++ {
		yy := clampInt(y+dy, 0, rows-1)
		rowBase := yy * cols
		for dx := -half; dx <= half; dx++ {
			lx := clampInt(xL+dx, 0, cols-1)
			rx := clampInt(xR+dx, 0, cols-1)
			a := float64(gl[rowBase+lx]) - meanL
			b := float64(gr[rowBase+rx]) - meanR
			num += a * b
			varL += a * a
			varR += b * b
		}
	}
	if varL <= 0 || varR <= 0 {
		return -1
	}
	return num / math.Sqrt(varL*varR)
}
