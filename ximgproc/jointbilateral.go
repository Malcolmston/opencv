package ximgproc

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// JointBilateralFilter applies a cross/joint bilateral filter: the spatial and
// range weights are taken from the guidance image joint, but the weighted
// average is formed over the samples of src. It returns a new Mat shaped like
// src.
//
// For every output pixel p,
//
//	out(p) = (1/W) · Σ_q Gs(‖p−q‖)·Gr(‖joint_p − joint_q‖)·src(q),
//
// where Gs is a spatial Gaussian of standard deviation sigmaSpace and Gr a range
// Gaussian of standard deviation sigmaColor evaluated on the guidance
// differences (summed over the guide's channels). Because the edge structure
// comes from joint, this transfers the guide's edges onto src — the basis of
// flash/no-flash denoising, depth up-sampling and the rolling-guidance filter.
//
// d is the diameter of the pixel neighbourhood; when d ≤ 0 it is derived from
// sigmaSpace as 2·round(1.5·sigmaSpace)+1. joint and src must share width and
// height; joint may have any channel count. It panics on a size mismatch. When
// joint == src this reduces to an ordinary bilateral filter. The filter is
// deterministic.
func JointBilateralFilter(joint, src *cv.Mat, d int, sigmaColor, sigmaSpace float64) *cv.Mat {
	if joint.Rows != src.Rows || joint.Cols != src.Cols {
		panic("ximgproc: JointBilateralFilter joint and src must share dimensions")
	}
	return jointBilateralCore(joint, src, d, sigmaColor, sigmaSpace)
}

// jointBilateralCore is the shared cross-bilateral engine. When sigmaSpace or
// sigmaColor is non-positive it is clamped to a small value, mirroring the root
// package's BilateralFilter.
func jointBilateralCore(joint, src *cv.Mat, d int, sigmaColor, sigmaSpace float64) *cv.Mat {
	if sigmaColor <= 0 {
		sigmaColor = 1
	}
	if sigmaSpace <= 0 {
		sigmaSpace = 1
	}
	if d <= 0 {
		d = 2*int(math.Round(1.5*sigmaSpace)) + 1
	}
	if d%2 == 0 {
		d++
	}
	radius := d / 2
	rows, cols := src.Rows, src.Cols
	gch := joint.Channels
	sch := src.Channels

	// Spatial weights.
	side := 2*radius + 1
	spatial := make([]float64, side*side)
	gs2 := 2 * sigmaSpace * sigmaSpace
	idx := 0
	for dy := -radius; dy <= radius; dy++ {
		for dx := -radius; dx <= radius; dx++ {
			spatial[idx] = math.Exp(-float64(dx*dx+dy*dy) / gs2)
			idx++
		}
	}
	// Range weights as a lookup over the summed absolute guidance difference,
	// quantised to unit steps up to the maximum possible (255·channels).
	gc2 := 2 * sigmaColor * sigmaColor
	maxDiff := 255*gch + 1
	rangeLUT := make([]float64, maxDiff)
	for k := 0; k < maxDiff; k++ {
		rangeLUT[k] = math.Exp(-float64(k*k) / gc2)
	}

	dst := cv.NewMat(rows, cols, sch)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			gi := (y*cols + x) * gch
			sums := make([]float64, sch)
			var wsum float64
			si := 0
			for dy := -radius; dy <= radius; dy++ {
				yy := reflect(y+dy, rows)
				for dx := -radius; dx <= radius; dx++ {
					xx := reflect(x+dx, cols)
					// Guidance difference (L1 over guide channels).
					gj := (yy*cols + xx) * gch
					diff := 0
					for c := 0; c < gch; c++ {
						diff += absInt(int(joint.Data[gi+c]) - int(joint.Data[gj+c]))
					}
					w := spatial[si] * rangeLUT[diff]
					sj := (yy*cols + xx) * sch
					for c := 0; c < sch; c++ {
						sums[c] += w * float64(src.Data[sj+c])
					}
					wsum += w
					si++
				}
			}
			oi := (y*cols + x) * sch
			for c := 0; c < sch; c++ {
				dst.Data[oi+c] = clampU8(sums[c] / wsum)
			}
		}
	}
	return dst
}

// absInt returns the absolute value of an int.
func absInt(v int) int {
	if v < 0 {
		return -v
	}
	return v
}

// RollingGuidanceFilter applies the rolling-guidance filter of Zhang, Shen, Xu
// and Jia ("Rolling Guidance Filter", 2014) to src and returns a new Mat of the
// same shape. It removes small-scale structures (texture, noise) while
// progressively recovering and sharpening the large-scale edges.
//
// The filter first erases fine detail with a Gaussian of standard deviation
// sigmaSpace, then repeatedly re-estimates the guidance by joint-bilateral
// filtering the ORIGINAL src using the current smoothed image as the guide:
//
//	J^{t+1} = JointBilateral(guide = J^t, src = src),   J^1 = Gaussian(src).
//
// Each rolling iteration lets genuine edges (which survive in J^t) re-assert
// themselves while structures smaller than sigmaSpace stay suppressed. d,
// sigmaColor and sigmaSpace have the same meaning as in [JointBilateralFilter];
// iters is the number of rolling iterations (typically 3–5, and must be ≥ 1).
// It panics if iters < 1. The filter is deterministic.
func RollingGuidanceFilter(src *cv.Mat, d int, sigmaColor, sigmaSpace float64, iters int) *cv.Mat {
	if iters < 1 {
		panic("ximgproc: RollingGuidanceFilter requires iters >= 1")
	}
	rows, cols := src.Rows, src.Cols

	// J^1 = Gaussian(src): small structures removed.
	planes := planesFromMat(src)
	for c := range planes {
		planes[c] = gaussianBlurFloat(planes[c], rows, cols, sigmaSpace)
	}
	guide := matFromPlanes(planes, rows, cols)

	for it := 0; it < iters; it++ {
		guide = jointBilateralCore(guide, src, d, sigmaColor, sigmaSpace)
	}
	return guide
}
