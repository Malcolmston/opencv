package cudastereo

import (
	"fmt"
	"math"

	cv "github.com/malcolmston/opencv"
)

// Default parameters for [StereoBeliefPropagation], matching OpenCV's CUDA
// defaults.
const (
	bpDefaultIters       = 5
	bpDefaultLevels      = 5
	bpDefaultMaxData     = 10.0
	bpDefaultDataWeight  = 0.07
	bpDefaultMaxDisc     = 1.7
	bpDefaultSingleJump  = 1.0
	bpDefaultNumDisparit = 64
)

// StereoBeliefPropagation is a CPU-backed mirror of
// cv::cuda::StereoBeliefPropagation and a genuine hierarchical loopy
// belief-propagation stereo matcher (Felzenszwalb–Huttenlocher).
//
// For a rectified pair it forms a truncated-linear data cost
// D_p(d) = DataWeight·min(|I_L(p) − I_R(p−d)|, MaxDataTerm), then minimises the
// energy
//
//	E(d) = Σ_p D_p(d_p) + Σ_{(p,q)} V(d_p, d_q),  V(a,b) = min(DiscSingleJump·|a−b|, MaxDiscTerm)
//
// by passing messages on the four-connected pixel grid. Each message update is
// the O(NumDisparities) lower-envelope (distance-transform) computation, so a
// full sweep is linear in the disparity range. Messages are iterated NumIters
// times per pyramid level, coarse-to-fine over NumLevels levels, with the coarse
// solution initialising the finer one — the standard way to make loopy BP
// converge quickly and escape weak local minima.
//
// Unlike the block matchers, belief propagation assigns a disparity to every
// pixel (there is no [github.com/malcolmston/opencv/stereo.InvalidDisparity]
// marker in the output).
//
// Build one with [CreateStereoBeliefPropagation].
type StereoBeliefPropagation struct {
	// NumDisparities is the width of the disparity search range in pixels.
	NumDisparities int
	// NumIters is the number of message-passing iterations run at each pyramid
	// level. Defaults to 5 when non-positive.
	NumIters int
	// NumLevels is the number of coarse-to-fine pyramid levels. Defaults to 5 when
	// non-positive; a value of 1 runs single-scale loopy BP.
	NumLevels int
	// MaxDataTerm truncates the per-pixel data cost. Defaults to 10 when
	// non-positive.
	MaxDataTerm float64
	// DataWeight scales the truncated data cost relative to the smoothness prior.
	// Defaults to 0.07 when non-positive.
	DataWeight float64
	// MaxDiscTerm caps the smoothness penalty between neighbours (the
	// discontinuity truncation). Defaults to 1.7 when non-positive.
	MaxDiscTerm float64
	// DiscSingleJump is the smoothness penalty slope per unit disparity change.
	// Defaults to 1 when non-positive.
	DiscSingleJump float64
}

// CreateStereoBeliefPropagation constructs a matcher, mirroring
// cv::cuda::createStereoBeliefPropagation(ndisp, iters, levels). Non-positive
// arguments fall back to the OpenCV CUDA defaults (64, 5, 5); the energy-term
// weights are always seeded with the OpenCV defaults.
func CreateStereoBeliefPropagation(ndisp, iters, levels int) *StereoBeliefPropagation {
	if ndisp <= 0 {
		ndisp = bpDefaultNumDisparit
	}
	if iters <= 0 {
		iters = bpDefaultIters
	}
	if levels <= 0 {
		levels = bpDefaultLevels
	}
	return &StereoBeliefPropagation{
		NumDisparities: ndisp,
		NumIters:       iters,
		NumLevels:      levels,
		MaxDataTerm:    bpDefaultMaxData,
		DataWeight:     bpDefaultDataWeight,
		MaxDiscTerm:    bpDefaultMaxDisc,
		DiscSingleJump: bpDefaultSingleJump,
	}
}

// EstimateRecommendedParams fills in disparity range, iteration count and level
// count appropriate for an image of the given size, mirroring
// cv::cuda::StereoBeliefPropagation::estimateRecommendedParams. It also stores
// the results into the receiver's NumDisparities/NumIters/NumLevels fields for
// convenience and returns them.
func (bp *StereoBeliefPropagation) EstimateRecommendedParams(width, height int) (ndisp, iters, levels int) {
	ndisp = width / 4
	if ndisp&1 != 0 {
		ndisp++
	}
	if ndisp < 1 {
		ndisp = 1
	}
	mm := width
	if height > mm {
		mm = height
	}
	iters = mm/100 + 2
	levels = int(math.Log(float64(mm))) + 1
	if levels < 1 {
		levels = 1
	}
	bp.NumDisparities, bp.NumIters, bp.NumLevels = ndisp, iters, levels
	return ndisp, iters, levels
}

func (bp *StereoBeliefPropagation) resolved() (ndisp, iters, levels int, maxData, dataWeight, maxDisc, jump float64) {
	ndisp = bp.NumDisparities
	if ndisp <= 0 {
		ndisp = bpDefaultNumDisparit
	}
	iters = bp.NumIters
	if iters <= 0 {
		iters = bpDefaultIters
	}
	levels = bp.NumLevels
	if levels <= 0 {
		levels = bpDefaultLevels
	}
	maxData = bp.MaxDataTerm
	if maxData <= 0 {
		maxData = bpDefaultMaxData
	}
	dataWeight = bp.DataWeight
	if dataWeight <= 0 {
		dataWeight = bpDefaultDataWeight
	}
	maxDisc = bp.MaxDiscTerm
	if maxDisc <= 0 {
		maxDisc = bpDefaultMaxDisc
	}
	jump = bp.DiscSingleJump
	if jump <= 0 {
		jump = bpDefaultSingleJump
	}
	return ndisp, iters, levels, maxData, dataWeight, maxDisc, jump
}

// Compute matches left against right and returns a single-channel 8-bit
// disparity map as a [GpuMat], mirroring
// cv::cuda::StereoBeliefPropagation::compute. The stream argument is accepted for
// API compatibility and may be nil.
//
// It panics if either input is empty, the inputs differ in size, or an input has
// an unsupported channel count.
func (bp *StereoBeliefPropagation) Compute(left, right *GpuMat, stream *Stream) *GpuMat {
	_ = stream
	ndisp, iters, levels, maxData, dataWeight, maxDisc, jump := bp.resolved()

	rows, cols, gl := grayGrid(matOf(left, "left"))
	rrows, rcols, gr := grayGrid(matOf(right, "right"))
	if rows != rrows || cols != rcols {
		panic(fmt.Sprintf("cudastereo: StereoBeliefPropagation.Compute size mismatch left %dx%d right %dx%d", rows, cols, rrows, rcols))
	}

	disp := bpSolve(gl, gr, rows, cols, ndisp, iters, levels, maxData, dataWeight, maxDisc, jump)

	out := cv.NewMat(rows, cols, 1)
	for i, d := range disp {
		out.Data[i] = uint8(clampInt(d, 0, 255))
	}
	return &GpuMat{mat: out}
}

// bpDataCost builds the full-resolution truncated-linear data cost volume,
// laid out as cost[(y*cols+x)*ndisp + d].
func bpDataCost(gl, gr []int, rows, cols, ndisp int, dataWeight, maxData float64) []float64 {
	cost := make([]float64, rows*cols*ndisp)
	for y := 0; y < rows; y++ {
		row := y * cols
		for x := 0; x < cols; x++ {
			base := (row + x) * ndisp
			il := gl[row+x]
			for d := 0; d < ndisp; d++ {
				xr := x - d
				if xr < 0 {
					xr = 0
				}
				diff := il - gr[row+xr]
				if diff < 0 {
					diff = -diff
				}
				c := float64(diff)
				if c > maxData {
					c = maxData
				}
				cost[base+d] = dataWeight * c
			}
		}
	}
	return cost
}

// bpDownsampleData sum-pools a data-cost volume by 2×2 blocks, the coarsening
// step of the hierarchical BP pyramid. Costs of the four child pixels are summed
// so the coarse cost keeps the same disparity indexing.
func bpDownsampleData(prev []float64, pr, pc, nr, nc, ndisp int) []float64 {
	cur := make([]float64, nr*nc*ndisp)
	for y := 0; y < pr; y++ {
		cy := y / 2
		for x := 0; x < pc; x++ {
			cx := x / 2
			src := (y*pc + x) * ndisp
			dst := (cy*nc + cx) * ndisp
			for d := 0; d < ndisp; d++ {
				cur[dst+d] += prev[src+d]
			}
		}
	}
	return cur
}

// bpUpsampleMessages expands a coarse message array to the finer grid by
// replicating each parent message to its (up to four) children.
func bpUpsampleMessages(coarse []float64, cr, cc, fr, fc, ndisp int) []float64 {
	fine := make([]float64, fr*fc*ndisp)
	for y := 0; y < fr; y++ {
		py := y / 2
		if py >= cr {
			py = cr - 1
		}
		for x := 0; x < fc; x++ {
			px := x / 2
			if px >= cc {
				px = cc - 1
			}
			copy(fine[(y*fc+x)*ndisp:(y*fc+x)*ndisp+ndisp], coarse[(py*cc+px)*ndisp:(py*cc+px)*ndisp+ndisp])
		}
	}
	return fine
}

// bpSolve runs hierarchical loopy belief propagation and returns the per-pixel
// integer disparity in row-major order.
func bpSolve(gl, gr []int, rows, cols, ndisp, iters, levels int, maxData, dataWeight, maxDisc, jump float64) []int {
	data0 := bpDataCost(gl, gr, rows, cols, ndisp, dataWeight, maxData)

	// Build the data-cost pyramid, coarsest last.
	dataPyr := [][]float64{data0}
	dims := [][2]int{{rows, cols}}
	for l := 1; l < levels; l++ {
		pr, pc := dims[l-1][0], dims[l-1][1]
		if pr <= 1 && pc <= 1 {
			break
		}
		nr, nc := (pr+1)/2, (pc+1)/2
		dataPyr = append(dataPyr, bpDownsampleData(dataPyr[l-1], pr, pc, nr, nc, ndisp))
		dims = append(dims, [2]int{nr, nc})
	}

	var msgU, msgD, msgL, msgR []float64
	for l := len(dataPyr) - 1; l >= 0; l-- {
		r, c := dims[l][0], dims[l][1]
		n := r * c
		if l == len(dataPyr)-1 {
			msgU = make([]float64, n*ndisp)
			msgD = make([]float64, n*ndisp)
			msgL = make([]float64, n*ndisp)
			msgR = make([]float64, n*ndisp)
		} else {
			cr, cc := dims[l+1][0], dims[l+1][1]
			msgU = bpUpsampleMessages(msgU, cr, cc, r, c, ndisp)
			msgD = bpUpsampleMessages(msgD, cr, cc, r, c, ndisp)
			msgL = bpUpsampleMessages(msgL, cr, cc, r, c, ndisp)
			msgR = bpUpsampleMessages(msgR, cr, cc, r, c, ndisp)
		}
		bpRun(dataPyr[l], msgU, msgD, msgL, msgR, r, c, ndisp, iters, jump, maxDisc)
	}

	return bpDecode(data0, msgU, msgD, msgL, msgR, rows, cols, ndisp)
}

// bpLowerEnvelope performs the truncated-linear message update in place: the
// L1 distance transform with slope jump, followed by truncation at min+maxDisc
// and mean-centring to keep the accumulated messages bounded.
func bpLowerEnvelope(h []float64, jump, maxDisc float64) {
	n := len(h)
	for d := 1; d < n; d++ {
		if h[d] > h[d-1]+jump {
			h[d] = h[d-1] + jump
		}
	}
	for d := n - 2; d >= 0; d-- {
		if h[d] > h[d+1]+jump {
			h[d] = h[d+1] + jump
		}
	}
	mn := h[0]
	for _, v := range h {
		if v < mn {
			mn = v
		}
	}
	capv := mn + maxDisc
	var sum float64
	for d := range h {
		if h[d] > capv {
			h[d] = capv
		}
		sum += h[d]
	}
	mean := sum / float64(n)
	for d := range h {
		h[d] -= mean
	}
}

// bpRun iterates message passing on one pyramid level, double-buffering so every
// message in an iteration is computed from the previous iteration's values.
func bpRun(data, msgU, msgD, msgL, msgR []float64, rows, cols, ndisp, iters int, jump, maxDisc float64) {
	n := rows * cols
	newU := make([]float64, n*ndisp)
	newD := make([]float64, n*ndisp)
	newL := make([]float64, n*ndisp)
	newR := make([]float64, n*ndisp)
	h := make([]float64, ndisp)

	for it := 0; it < iters; it++ {
		for y := 0; y < rows; y++ {
			for x := 0; x < cols; x++ {
				p := y*cols + x
				base := p * ndisp
				up, dn, lf, rt := p-cols, p+cols, p-1, p+1
				hasUp, hasDn, hasLf, hasRt := y > 0, y < rows-1, x > 0, x < cols-1

				// Message p -> right (exclude incoming from right).
				if hasRt {
					for d := 0; d < ndisp; d++ {
						v := data[base+d]
						if hasUp {
							v += msgD[up*ndisp+d]
						}
						if hasDn {
							v += msgU[dn*ndisp+d]
						}
						if hasLf {
							v += msgR[lf*ndisp+d]
						}
						h[d] = v
					}
					bpLowerEnvelope(h, jump, maxDisc)
					copy(newR[base:base+ndisp], h)
				}
				// Message p -> left (exclude incoming from left).
				if hasLf {
					for d := 0; d < ndisp; d++ {
						v := data[base+d]
						if hasUp {
							v += msgD[up*ndisp+d]
						}
						if hasDn {
							v += msgU[dn*ndisp+d]
						}
						if hasRt {
							v += msgL[rt*ndisp+d]
						}
						h[d] = v
					}
					bpLowerEnvelope(h, jump, maxDisc)
					copy(newL[base:base+ndisp], h)
				}
				// Message p -> up (exclude incoming from up).
				if hasUp {
					for d := 0; d < ndisp; d++ {
						v := data[base+d]
						if hasDn {
							v += msgU[dn*ndisp+d]
						}
						if hasLf {
							v += msgR[lf*ndisp+d]
						}
						if hasRt {
							v += msgL[rt*ndisp+d]
						}
						h[d] = v
					}
					bpLowerEnvelope(h, jump, maxDisc)
					copy(newU[base:base+ndisp], h)
				}
				// Message p -> down (exclude incoming from down).
				if hasDn {
					for d := 0; d < ndisp; d++ {
						v := data[base+d]
						if hasUp {
							v += msgD[up*ndisp+d]
						}
						if hasLf {
							v += msgR[lf*ndisp+d]
						}
						if hasRt {
							v += msgL[rt*ndisp+d]
						}
						h[d] = v
					}
					bpLowerEnvelope(h, jump, maxDisc)
					copy(newD[base:base+ndisp], h)
				}
			}
		}
		copy(msgU, newU)
		copy(msgD, newD)
		copy(msgL, newL)
		copy(msgR, newR)
	}
}

// bpDecode selects, for every pixel, the disparity minimising the data cost plus
// the four incoming messages.
func bpDecode(data, msgU, msgD, msgL, msgR []float64, rows, cols, ndisp int) []int {
	out := make([]int, rows*cols)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			p := y*cols + x
			base := p * ndisp
			up, dn, lf, rt := p-cols, p+cols, p-1, p+1
			hasUp, hasDn, hasLf, hasRt := y > 0, y < rows-1, x > 0, x < cols-1
			bestD, bestC := 0, math.Inf(1)
			for d := 0; d < ndisp; d++ {
				v := data[base+d]
				if hasUp {
					v += msgD[up*ndisp+d]
				}
				if hasDn {
					v += msgU[dn*ndisp+d]
				}
				if hasLf {
					v += msgR[lf*ndisp+d]
				}
				if hasRt {
					v += msgL[rt*ndisp+d]
				}
				if v < bestC {
					bestC, bestD = v, d
				}
			}
			out[p] = bestD
		}
	}
	return out
}
