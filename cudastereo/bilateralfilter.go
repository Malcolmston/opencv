package cudastereo

import (
	"fmt"
	"math"

	cv "github.com/malcolmston/opencv"
)

// DisparityBilateralFilter is a CPU-backed mirror of
// cv::cuda::DisparityBilateralFilter: an edge-preserving refinement of a
// disparity map guided by the reference image. It replaces each disparity with a
// bilateral-weighted average of its neighbours, where the weight combines a
// spatial Gaussian, a range Gaussian on the guidance image (so smoothing does not
// cross intensity edges), and a disparity-continuity gate that ignores
// neighbours whose disparity differs by more than MaxDiscThreshold. The filter is
// applied Iters times.
//
// Build one with [CreateDisparityBilateralFilter].
type DisparityBilateralFilter struct {
	// NumDisparities is the disparity range the filter was configured for; it
	// scales the default MaxDiscThreshold.
	NumDisparities int
	// Radius is the half-width of the square filter window in pixels. Defaults to 3
	// when non-positive.
	Radius int
	// Iters is the number of filtering passes. Defaults to 1 when non-positive.
	Iters int
	// EdgeThreshold is the guidance-image intensity difference above which
	// neighbours are strongly attenuated (feeds the range Gaussian sigma).
	// Defaults to 10 when non-positive.
	EdgeThreshold float64
	// MaxDiscThreshold is the maximum disparity difference, in pixels, a neighbour
	// may have to still contribute; it preserves depth discontinuities. Defaults to
	// max(1, NumDisparities/10) when non-positive.
	MaxDiscThreshold float64
	// SigmaRange controls the guidance range Gaussian. Defaults to 10 when
	// non-positive.
	SigmaRange float64
}

// CreateDisparityBilateralFilter constructs a filter, mirroring
// cv::cuda::createDisparityBilateralFilter(ndisp, radius, iters). Non-positive
// arguments fall back to the OpenCV CUDA defaults (64, 3, 1).
func CreateDisparityBilateralFilter(ndisp, radius, iters int) *DisparityBilateralFilter {
	if ndisp <= 0 {
		ndisp = 64
	}
	if radius <= 0 {
		radius = 3
	}
	if iters <= 0 {
		iters = 1
	}
	return &DisparityBilateralFilter{
		NumDisparities:   ndisp,
		Radius:           radius,
		Iters:            iters,
		EdgeThreshold:    10,
		MaxDiscThreshold: math.Max(1, float64(ndisp)/10),
		SigmaRange:       10,
	}
}

// Apply refines disparity using image as the guidance reference and returns a new
// single-channel 8-bit disparity map as a [GpuMat], mirroring
// cv::cuda::DisparityBilateralFilter::apply. disparity must be single-channel;
// image may be single- or three-channel (converted to gray) and must match
// disparity in size. Pixels holding
// [github.com/malcolmston/opencv/stereo.InvalidDisparity] (0) are preserved. The
// stream argument is accepted for API compatibility and may be nil.
//
// It panics on empty inputs, a multi-channel disparity map, or a size mismatch.
func (f *DisparityBilateralFilter) Apply(disparity, image *GpuMat, stream *Stream) *GpuMat {
	_ = stream
	disp := matOf(disparity, "disparity")
	if disp.Channels != 1 {
		panic("cudastereo: DisparityBilateralFilter.Apply requires a single-channel disparity map")
	}
	rows, cols, guide := grayGrid(matOf(image, "image"))
	if rows != disp.Rows || cols != disp.Cols {
		panic(fmt.Sprintf("cudastereo: DisparityBilateralFilter.Apply size mismatch disparity %dx%d image %dx%d",
			disp.Rows, disp.Cols, rows, cols))
	}

	radius := f.Radius
	if radius <= 0 {
		radius = 3
	}
	iters := f.Iters
	if iters <= 0 {
		iters = 1
	}
	edge := f.EdgeThreshold
	if edge <= 0 {
		edge = 10
	}
	maxDisc := f.MaxDiscThreshold
	if maxDisc <= 0 {
		maxDisc = math.Max(1, float64(f.NumDisparities)/10)
	}
	sigmaRange := f.SigmaRange
	if sigmaRange <= 0 {
		sigmaRange = 10
	}
	sigmaSpace := float64(radius)
	if sigmaSpace <= 0 {
		sigmaSpace = 1
	}

	cur := make([]float64, rows*cols)
	for i := range cur {
		cur[i] = float64(disp.Data[i])
	}

	for it := 0; it < iters; it++ {
		next := make([]float64, rows*cols)
		for y := 0; y < rows; y++ {
			for x := 0; x < cols; x++ {
				p := y*cols + x
				center := cur[p]
				if disp.Data[p] == 0 { // preserve InvalidDisparity holes
					next[p] = center
					continue
				}
				gc := float64(guide[p])
				var wSum, dSum float64
				for dy := -radius; dy <= radius; dy++ {
					ny := y + dy
					if ny < 0 || ny >= rows {
						continue
					}
					for dx := -radius; dx <= radius; dx++ {
						nx := x + dx
						if nx < 0 || nx >= cols {
							continue
						}
						q := ny*cols + nx
						if disp.Data[q] == 0 {
							continue
						}
						nd := cur[q]
						if math.Abs(nd-center) > maxDisc {
							continue // preserve depth discontinuity
						}
						gdiff := float64(guide[q]) - gc
						sdist := float64(dx*dx + dy*dy)
						w := math.Exp(-sdist/(2*sigmaSpace*sigmaSpace)) *
							math.Exp(-(gdiff*gdiff)/(2*sigmaRange*sigmaRange)) *
							math.Exp(-math.Abs(gdiff)/edge)
						wSum += w
						dSum += w * nd
					}
				}
				if wSum > 0 {
					next[p] = dSum / wSum
				} else {
					next[p] = center
				}
			}
		}
		cur = next
	}

	out := cv.NewMat(rows, cols, 1)
	for i := range cur {
		out.Data[i] = uint8(clampInt(int(math.Round(cur[i])), 0, 255))
	}
	return &GpuMat{mat: out}
}
