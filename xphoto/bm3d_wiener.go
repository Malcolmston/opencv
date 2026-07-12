package xphoto

import (
	"sort"

	cv "github.com/malcolmston/opencv"
)

// Bm3dDenoisingTwoStep denoises src with the full two-step BM3D pipeline,
// porting cv::xphoto::bm3dDenoising's default BM3D_STEPALL mode. It first runs
// the hard-threshold basic estimate ([Bm3dDenoising]) and then refines it with
// an empirical-Wiener second stage ([Bm3dDenoisingStep2]) that uses the basic
// estimate as an oracle. The two-step result is stronger than the basic estimate
// alone: the Wiener stage recovers detail the hard threshold removed while
// further suppressing noise in flat regions. h is the assumed noise standard
// deviation (sigma). src may be single- or three-channel; each channel is
// processed independently. The input is not modified.
func Bm3dDenoisingTwoStep(src *cv.Mat, h float64) *cv.Mat {
	requireNonEmpty(src, "Bm3dDenoisingTwoStep")
	if h <= 0 {
		h = 1
	}
	basic := Bm3dDenoising(src, h)
	return Bm3dDenoisingStep2(src, basic, h)
}

// Bm3dDenoisingStep2 runs only the empirical-Wiener second stage of BM3D
// (BM3D_STEP2), porting the refinement half of cv::xphoto::bm3dDenoising. noisy
// is the original noisy image and basic is a first-stage (basic) estimate such
// as the output of [Bm3dDenoising]; the two must have identical dimensions and
// channel counts. Block matching is performed on the cleaner basic estimate, and
// for each group the transform coefficients of the noisy stack are shrunk by
// empirical-Wiener gains W = E^2/(E^2 + sigma^2) derived from the basic estimate
// (E is the basic-estimate coefficient). The shrunk stacks are inverse
// transformed and aggregated with weights 1/sum(W^2). h is the assumed noise
// sigma. Neither input is modified.
func Bm3dDenoisingStep2(noisy, basic *cv.Mat, h float64) *cv.Mat {
	requireNonEmpty(noisy, "Bm3dDenoisingStep2")
	requireNonEmpty(basic, "Bm3dDenoisingStep2")
	requireSameSize(noisy, basic, "Bm3dDenoisingStep2")
	if noisy.Channels != basic.Channels {
		requireChannels(basic, noisy.Channels, "Bm3dDenoisingStep2")
	}
	if h <= 0 {
		h = 1
	}
	dst := cv.NewMat(noisy.Rows, noisy.Cols, noisy.Channels)
	for c := 0; c < noisy.Channels; c++ {
		bm3dWienerChannel(noisy, basic, dst, c, h)
	}
	return dst
}

// bm3dWienerChannel runs the Wiener stage on channel c.
func bm3dWienerChannel(noisy, basic, dst *cv.Mat, c int, sigma float64) {
	rows, cols := noisy.Rows, noisy.Cols
	B := bm3dBlock
	if rows < B || cols < B {
		for y := 0; y < rows; y++ {
			for x := 0; x < cols; x++ {
				dst.Set(y, x, c, basic.At(y, x, c))
			}
		}
		return
	}
	nPlane := make([]float64, rows*cols)
	bPlane := make([]float64, rows*cols)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			nPlane[y*cols+x] = float64(noisy.At(y, x, c))
			bPlane[y*cols+x] = float64(basic.At(y, x, c))
		}
	}
	numAcc := make([]float64, rows*cols)
	denAcc := make([]float64, rows*cols)
	matchTau := bm3dMatchFactor*sigma*sigma + bm3dMatchExtra
	cos := dctBasis(B)

	step := func(ry, rx int) {
		processReferenceWiener(nPlane, bPlane, rows, cols, ry, rx, B, cos, sigma, matchTau, numAcc, denAcc)
	}
	for ry := 0; ry <= rows-B; ry += bm3dStep {
		for rx := 0; rx <= cols-B; rx += bm3dStep {
			step(ry, rx)
		}
	}
	if (rows-B)%bm3dStep != 0 {
		for rx := 0; rx <= cols-B; rx += bm3dStep {
			step(rows-B, rx)
		}
	}
	if (cols-B)%bm3dStep != 0 {
		for ry := 0; ry <= rows-B; ry += bm3dStep {
			step(ry, cols-B)
		}
	}

	for i := 0; i < rows*cols; i++ {
		v := bPlane[i]
		if denAcc[i] > 0 {
			v = numAcc[i] / denAcc[i]
		}
		dst.Set(i/cols, i%cols, c, clampU8(v))
	}
}

// processReferenceWiener performs block matching on the basic estimate and
// empirical-Wiener filtering of the noisy stack for one reference block.
func processReferenceWiener(nPlane, bPlane []float64, rows, cols, ry, rx, B int, cos [][]float64, sigma, matchTau float64, numAcc, denAcc []float64) {
	type cand struct {
		y, x int
		dist float64
	}
	var cands []cand
	for sy := ry - bm3dSearchRad; sy <= ry+bm3dSearchRad; sy++ {
		if sy < 0 || sy > rows-B {
			continue
		}
		for sx := rx - bm3dSearchRad; sx <= rx+bm3dSearchRad; sx++ {
			if sx < 0 || sx > cols-B {
				continue
			}
			// Match on the cleaner basic estimate.
			d := blockDist(bPlane, cols, ry, rx, sy, sx, B)
			if sy == ry && sx == rx {
				cands = append(cands, cand{sy, sx, -1})
			} else if d <= matchTau {
				cands = append(cands, cand{sy, sx, d})
			}
		}
	}
	sort.SliceStable(cands, func(i, j int) bool {
		if cands[i].dist != cands[j].dist {
			return cands[i].dist < cands[j].dist
		}
		if cands[i].y != cands[j].y {
			return cands[i].y < cands[j].y
		}
		return cands[i].x < cands[j].x
	})
	if len(cands) > bm3dMaxGroup {
		cands = cands[:bm3dMaxGroup]
	}
	g := len(cands)

	// 2D-DCT of every block for both the noisy and basic (oracle) stacks.
	nGroup := make([][]float64, g)
	bGroup := make([][]float64, g)
	for b := 0; b < g; b++ {
		nBlk := make([]float64, B*B)
		bBlk := make([]float64, B*B)
		for i := 0; i < B; i++ {
			for j := 0; j < B; j++ {
				idx := (cands[b].y+i)*cols + (cands[b].x + j)
				nBlk[i*B+j] = nPlane[idx]
				bBlk[i*B+j] = bPlane[idx]
			}
		}
		nGroup[b] = dct2d(nBlk, B, cos)
		bGroup[b] = dct2d(bBlk, B, cos)
	}

	// 1D-DCT across the group, empirical-Wiener shrinkage, inverse 1D-DCT.
	sigma2 := sigma * sigma
	nCol := make([]float64, g)
	bCol := make([]float64, g)
	var wsum float64
	for pos := 0; pos < B*B; pos++ {
		for b := 0; b < g; b++ {
			nCol[b] = nGroup[b][pos]
			bCol[b] = bGroup[b][pos]
		}
		nCoef := dct1d(nCol, g)
		bCoef := dct1d(bCol, g)
		for w := 0; w < g; w++ {
			e2 := bCoef[w] * bCoef[w]
			gain := e2 / (e2 + sigma2)
			nCoef[w] *= gain
			wsum += gain * gain
		}
		inv := idct1d(nCoef, g)
		for b := 0; b < g; b++ {
			nGroup[b][pos] = inv[b]
		}
	}
	weight := 1.0
	if wsum > 0 {
		weight = 1.0 / wsum
	}

	for b := 0; b < g; b++ {
		blk := idct2d(nGroup[b], B, cos)
		by, bx := cands[b].y, cands[b].x
		for i := 0; i < B; i++ {
			for j := 0; j < B; j++ {
				idx := (by+i)*cols + (bx + j)
				numAcc[idx] += weight * blk[i*B+j]
				denAcc[idx] += weight
			}
		}
	}
}
