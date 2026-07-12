package cudastereo

import (
	"fmt"
	"math"
	"sort"

	cv "github.com/malcolmston/opencv"
)

// Default parameters for [StereoConstantSpaceBP], matching OpenCV's CUDA
// defaults.
const (
	csbpDefaultIters      = 8
	csbpDefaultLevels     = 4
	csbpDefaultNrPlane    = 4
	csbpDefaultNumDisp    = 128
	csbpDefaultMaxData    = 30.0
	csbpDefaultDataWeight = 0.03
	csbpDefaultMaxDisc    = 1.94
	csbpDefaultSingleJump = 1.0
)

// StereoConstantSpaceBP is a CPU-backed mirror of
// cv::cuda::StereoConstantSpaceBP and a genuine constant-space belief
// propagation stereo matcher.
//
// Where full belief propagation ([StereoBeliefPropagation]) stores messages over
// the whole disparity range at every pixel, constant-space BP keeps only the
// NrPlane cheapest disparity hypotheses ("planes") per pixel. The message
// storage is therefore O(rows·cols·NrPlane), independent of NumDisparities — the
// constant-space property that lets the CUDA version scale to large disparity
// ranges. Messages are passed between neighbouring pixels' plane sets: because
// two neighbours may hold different candidate disparities, each message update is
// an explicit O(NrPlane²) minimisation over the truncated-linear smoothness
// prior V(a,b) = min(DiscSingleJump·|a−b|, MaxDiscTerm).
//
// Like the other belief-propagation matcher, it assigns a disparity to every
// pixel. Build one with [CreateStereoConstantSpaceBP].
type StereoConstantSpaceBP struct {
	// NumDisparities is the width of the disparity search range in pixels.
	NumDisparities int
	// NumIters is the number of message-passing iterations. Defaults to 8 when
	// non-positive.
	NumIters int
	// NumLevels is retained for API compatibility with the CUDA class and reported
	// by [StereoConstantSpaceBP.EstimateRecommendedParams]; this implementation
	// runs the constant-space message passing at full resolution. Defaults to 4.
	NumLevels int
	// NrPlane is the number of disparity hypotheses kept per pixel. Defaults to 4
	// when non-positive.
	NrPlane int
	// MaxDataTerm truncates the per-pixel data cost. Defaults to 30 when
	// non-positive.
	MaxDataTerm float64
	// DataWeight scales the truncated data cost. Defaults to 0.03 when
	// non-positive.
	DataWeight float64
	// MaxDiscTerm caps the smoothness penalty between neighbours. Defaults to 1.94
	// when non-positive.
	MaxDiscTerm float64
	// DiscSingleJump is the smoothness penalty slope per unit disparity change.
	// Defaults to 1 when non-positive.
	DiscSingleJump float64
}

// CreateStereoConstantSpaceBP constructs a matcher, mirroring
// cv::cuda::createStereoConstantSpaceBP(ndisp, iters, levels, nr_plane).
// Non-positive arguments fall back to the OpenCV CUDA defaults (128, 8, 4, 4).
func CreateStereoConstantSpaceBP(ndisp, iters, levels, nrPlane int) *StereoConstantSpaceBP {
	if ndisp <= 0 {
		ndisp = csbpDefaultNumDisp
	}
	if iters <= 0 {
		iters = csbpDefaultIters
	}
	if levels <= 0 {
		levels = csbpDefaultLevels
	}
	if nrPlane <= 0 {
		nrPlane = csbpDefaultNrPlane
	}
	return &StereoConstantSpaceBP{
		NumDisparities: ndisp,
		NumIters:       iters,
		NumLevels:      levels,
		NrPlane:        nrPlane,
		MaxDataTerm:    csbpDefaultMaxData,
		DataWeight:     csbpDefaultDataWeight,
		MaxDiscTerm:    csbpDefaultMaxDisc,
		DiscSingleJump: csbpDefaultSingleJump,
	}
}

// EstimateRecommendedParams fills in disparity range, iterations, levels and
// plane count appropriate for an image of the given size, mirroring
// cv::cuda::StereoConstantSpaceBP::estimateRecommendedParams. It also stores the
// results into the receiver and returns them.
func (csbp *StereoConstantSpaceBP) EstimateRecommendedParams(width, height int) (ndisp, iters, levels, nrPlane int) {
	ndisp = int(float64(width) / 3.14)
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
	if mm > 1200 {
		iters = mm/100 - 4
	} else {
		iters = mm/100 + 4
	}
	if iters < 1 {
		iters = 1
	}
	levels = int(math.Log(float64(mm)))*2 - 2
	if levels < 1 {
		levels = 1
	}
	nrPlane = int(float64(ndisp) / math.Pow(2.0, float64(levels+1)))
	if nrPlane < 1 {
		nrPlane = 1
	}
	csbp.NumDisparities, csbp.NumIters, csbp.NumLevels, csbp.NrPlane = ndisp, iters, levels, nrPlane
	return ndisp, iters, levels, nrPlane
}

func (csbp *StereoConstantSpaceBP) resolved() (ndisp, iters, nrPlane int, maxData, dataWeight, maxDisc, jump float64) {
	ndisp = csbp.NumDisparities
	if ndisp <= 0 {
		ndisp = csbpDefaultNumDisp
	}
	iters = csbp.NumIters
	if iters <= 0 {
		iters = csbpDefaultIters
	}
	nrPlane = csbp.NrPlane
	if nrPlane <= 0 {
		nrPlane = csbpDefaultNrPlane
	}
	if nrPlane > ndisp {
		nrPlane = ndisp
	}
	maxData = csbp.MaxDataTerm
	if maxData <= 0 {
		maxData = csbpDefaultMaxData
	}
	dataWeight = csbp.DataWeight
	if dataWeight <= 0 {
		dataWeight = csbpDefaultDataWeight
	}
	maxDisc = csbp.MaxDiscTerm
	if maxDisc <= 0 {
		maxDisc = csbpDefaultMaxDisc
	}
	jump = csbp.DiscSingleJump
	if jump <= 0 {
		jump = csbpDefaultSingleJump
	}
	return ndisp, iters, nrPlane, maxData, dataWeight, maxDisc, jump
}

// Compute matches left against right and returns a single-channel 8-bit
// disparity map as a [GpuMat], mirroring
// cv::cuda::StereoConstantSpaceBP::compute. The stream argument is accepted for
// API compatibility and may be nil.
//
// It panics if either input is empty, the inputs differ in size, or an input has
// an unsupported channel count.
func (csbp *StereoConstantSpaceBP) Compute(left, right *GpuMat, stream *Stream) *GpuMat {
	_ = stream
	ndisp, iters, nrPlane, maxData, dataWeight, maxDisc, jump := csbp.resolved()

	rows, cols, gl := grayGrid(matOf(left, "left"))
	rrows, rcols, gr := grayGrid(matOf(right, "right"))
	if rows != rrows || cols != rcols {
		panic(fmt.Sprintf("cudastereo: StereoConstantSpaceBP.Compute size mismatch left %dx%d right %dx%d", rows, cols, rrows, rcols))
	}

	disp := csbpSolve(gl, gr, rows, cols, ndisp, iters, nrPlane, maxData, dataWeight, maxDisc, jump)

	out := cv.NewMat(rows, cols, 1)
	for i, d := range disp {
		out.Data[i] = uint8(clampInt(d, 0, 255))
	}
	return &GpuMat{mat: out}
}

// csbpSolve runs constant-space belief propagation and returns per-pixel integer
// disparities in row-major order.
func csbpSolve(gl, gr []int, rows, cols, ndisp, iters, nrPlane int, maxData, dataWeight, maxDisc, jump float64) []int {
	n := rows * cols

	// Per-pixel candidate planes (disparities) and their data cost, chosen as the
	// nrPlane cheapest disparities. This is the constant-space selection: message
	// storage below is O(n*nrPlane), independent of ndisp.
	planeDisp := make([]int, n*nrPlane)
	planeData := make([]float64, n*nrPlane)
	full := make([]float64, ndisp)
	order := make([]int, ndisp)
	for y := 0; y < rows; y++ {
		row := y * cols
		for x := 0; x < cols; x++ {
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
				full[d] = dataWeight * c
				order[d] = d
			}
			sort.SliceStable(order, func(a, b int) bool { return full[order[a]] < full[order[b]] })
			pbase := (row + x) * nrPlane
			for k := 0; k < nrPlane; k++ {
				d := order[k]
				planeDisp[pbase+k] = d
				planeData[pbase+k] = full[d]
			}
		}
	}

	// Messages, each indexed by the receiver pixel's plane set.
	msgU := make([]float64, n*nrPlane)
	msgD := make([]float64, n*nrPlane)
	msgL := make([]float64, n*nrPlane)
	msgR := make([]float64, n*nrPlane)
	newU := make([]float64, n*nrPlane)
	newD := make([]float64, n*nrPlane)
	newL := make([]float64, n*nrPlane)
	newR := make([]float64, n*nrPlane)

	smooth := func(a, b int) float64 {
		v := jump * float64(absInt(a-b))
		if v > maxDisc {
			v = maxDisc
		}
		return v
	}

	// message computes, for receiver q with plane disparities qDisp, the message
	// sent from sender pixel sBase using its planes, excluding the incoming
	// message that comes back from q's direction.
	message := func(dst []float64, sBase int, qDisp []int, incoming [][]float64, sPlaneDisp []int, sPlaneData []float64) {
		for j := 0; j < nrPlane; j++ {
			best := math.Inf(1)
			for i := 0; i < nrPlane; i++ {
				v := sPlaneData[i] + smooth(sPlaneDisp[i], qDisp[j])
				for _, in := range incoming {
					v += in[i]
				}
				if v < best {
					best = v
				}
			}
			dst[j] = best
		}
		// Mean-centre to keep messages bounded.
		var sum float64
		for j := 0; j < nrPlane; j++ {
			sum += dst[j]
		}
		mean := sum / float64(nrPlane)
		for j := 0; j < nrPlane; j++ {
			dst[j] -= mean
		}
	}

	slice := func(m []int, p int) []int { return m[p*nrPlane : p*nrPlane+nrPlane] }
	fslice := func(m []float64, p int) []float64 { return m[p*nrPlane : p*nrPlane+nrPlane] }

	for it := 0; it < iters; it++ {
		for y := 0; y < rows; y++ {
			for x := 0; x < cols; x++ {
				p := y*cols + x
				up, dn, lf, rt := p-cols, p+cols, p-1, p+1
				hasUp, hasDn, hasLf, hasRt := y > 0, y < rows-1, x > 0, x < cols-1
				sDisp := slice(planeDisp, p)
				sData := fslice(planeData, p)

				// Incoming messages to p, each indexed by p's own planes.
				inUp := zeroIfMissing(hasUp, msgD, up, nrPlane) // from up neighbour (sends down)
				inDn := zeroIfMissing(hasDn, msgU, dn, nrPlane) // from down neighbour (sends up)
				inLf := zeroIfMissing(hasLf, msgR, lf, nrPlane) // from left neighbour (sends right)
				inRt := zeroIfMissing(hasRt, msgL, rt, nrPlane) // from right neighbour (sends left)

				if hasRt {
					message(fslice(newR, p), p, slice(planeDisp, rt), [][]float64{inUp, inDn, inLf}, sDisp, sData)
				}
				if hasLf {
					message(fslice(newL, p), p, slice(planeDisp, lf), [][]float64{inUp, inDn, inRt}, sDisp, sData)
				}
				if hasUp {
					message(fslice(newU, p), p, slice(planeDisp, up), [][]float64{inDn, inLf, inRt}, sDisp, sData)
				}
				if hasDn {
					message(fslice(newD, p), p, slice(planeDisp, dn), [][]float64{inUp, inLf, inRt}, sDisp, sData)
				}
			}
		}
		copy(msgU, newU)
		copy(msgD, newD)
		copy(msgL, newL)
		copy(msgR, newR)
	}

	// Decode: pick the plane minimising data cost plus incoming messages.
	out := make([]int, n)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			p := y*cols + x
			up, dn, lf, rt := p-cols, p+cols, p-1, p+1
			hasUp, hasDn, hasLf, hasRt := y > 0, y < rows-1, x > 0, x < cols-1
			pbase := p * nrPlane
			bestK, bestC := 0, math.Inf(1)
			for k := 0; k < nrPlane; k++ {
				v := planeData[pbase+k]
				if hasUp {
					v += msgD[up*nrPlane+k]
				}
				if hasDn {
					v += msgU[dn*nrPlane+k]
				}
				if hasLf {
					v += msgR[lf*nrPlane+k]
				}
				if hasRt {
					v += msgL[rt*nrPlane+k]
				}
				if v < bestC {
					bestC, bestK = v, k
				}
			}
			out[p] = planeDisp[pbase+bestK]
		}
	}
	return out
}

// zeroIfMissing returns the nrPlane-length message slice for pixel p from array
// m when present is true, otherwise a fresh zero slice (no contribution).
func zeroIfMissing(present bool, m []float64, p, nrPlane int) []float64 {
	if !present {
		return make([]float64, nrPlane)
	}
	return m[p*nrPlane : p*nrPlane+nrPlane]
}
