package cudaarithm

import (
	"fmt"
	"math"

	cv "github.com/malcolmston/opencv"
)

// Split separates a multi-channel GpuMat into one single-channel GpuMat per
// channel, in channel order, mirroring cv::cuda::split. It delegates to
// [cv.Mat.Split].
func Split(src *GpuMat, _ ...*Stream) []*GpuMat {
	requireNonEmpty(src, "Split")
	planes := src.mat.Split()
	out := make([]*GpuMat, len(planes))
	for i, p := range planes {
		out[i] = wrap(p)
	}
	return out
}

// Merge interleaves single-channel GpuMats into one multi-channel GpuMat,
// mirroring cv::cuda::merge. Every plane must be single-channel and share a
// shape. It delegates to [cv.Merge].
func Merge(planes []*GpuMat, _ ...*Stream) *GpuMat {
	if len(planes) == 0 {
		panic("cudaarithm: Merge requires at least one plane")
	}
	mats := make([]*cv.Mat, len(planes))
	for i, p := range planes {
		requireChannels(p, 1, "Merge")
		mats[i] = p.mat
	}
	return wrap(cv.Merge(mats))
}

// Transpose swaps rows and columns, returning a new GpuMat of shape Cols×Rows,
// mirroring cv::cuda::transpose. It delegates to [cv.Transpose].
func Transpose(src *GpuMat, _ ...*Stream) *GpuMat {
	requireNonEmpty(src, "Transpose")
	return wrap(cv.Transpose(src.mat))
}

// Flip mirrors src along the axis chosen by code, mirroring cv::cuda::flip. It
// delegates to [cv.Flip]; code is a [cv.FlipCode].
func Flip(src *GpuMat, code cv.FlipCode, _ ...*Stream) *GpuMat {
	requireNonEmpty(src, "Flip")
	return wrap(cv.Flip(src.mat, code))
}

// LUT remaps every sample through a 256-entry lookup table, mirroring
// cv::cuda::LUT (the OpenCV CUDA LookUpTable). table must have exactly 256
// entries; the same table applies to every channel. It panics otherwise.
func LUT(src *GpuMat, table []uint8, _ ...*Stream) *GpuMat {
	requireNonEmpty(src, "LUT")
	if len(table) != 256 {
		panic(fmt.Sprintf("cudaarithm: LUT table must have 256 entries, got %d", len(table)))
	}
	out := cv.NewMat(src.mat.Rows, src.mat.Cols, src.mat.Channels)
	for i, s := range src.mat.Data {
		out.Data[i] = table[s]
	}
	return wrap(out)
}

// NormType selects the norm used by [Norm] and [Normalize].
type NormType int

const (
	// NormInf is the L-infinity norm: the maximum absolute sample value.
	NormInf NormType = iota
	// NormL1 is the L1 norm: the sum of absolute sample values.
	NormL1
	// NormL2 is the L2 (Euclidean) norm: sqrt of the sum of squared samples.
	NormL2
	// NormMinMax rescales samples so the range maps to [alpha, beta]. It is only
	// valid for [Normalize].
	NormMinMax
)

// Normalize rescales src according to normType and returns a new GpuMat,
// mirroring cv::cuda::normalize. For [NormMinMax] the sample range is mapped
// linearly onto [alpha, beta] (delegating to [cv.Normalize]). For [NormL1],
// [NormL2] and [NormInf] every sample is scaled by alpha divided by the
// corresponding norm of src, so the result has that norm equal to alpha; beta is
// unused. Results are rounded and saturated into [0,255].
func Normalize(src *GpuMat, alpha, beta float64, normType NormType, _ ...*Stream) *GpuMat {
	requireNonEmpty(src, "Normalize")
	if normType == NormMinMax {
		return wrap(cv.Normalize(src.mat, alpha, beta))
	}
	n := normValue(src.mat.Data, normType)
	out := cv.NewMat(src.mat.Rows, src.mat.Cols, src.mat.Channels)
	if n == 0 {
		return wrap(out)
	}
	scale := alpha / n
	for i, s := range src.mat.Data {
		out.Data[i] = roundToUint8(float64(s) * scale)
	}
	return wrap(out)
}

// Norm returns the norm of src selected by normType, mirroring cv::cuda::norm.
// [NormMinMax] is not a valid measurement norm and panics.
func Norm(src *GpuMat, normType NormType, _ ...*Stream) float64 {
	requireNonEmpty(src, "Norm")
	if normType == NormMinMax {
		panic("cudaarithm: Norm does not support NormMinMax")
	}
	return normValue(src.mat.Data, normType)
}

// normValue computes the requested norm over a flat sample slice.
func normValue(data []uint8, normType NormType) float64 {
	switch normType {
	case NormInf:
		var m float64
		for _, s := range data {
			if float64(s) > m {
				m = float64(s)
			}
		}
		return m
	case NormL1:
		var sum float64
		for _, s := range data {
			sum += float64(s)
		}
		return sum
	case NormL2:
		var sum float64
		for _, s := range data {
			sum += float64(s) * float64(s)
		}
		return math.Sqrt(sum)
	default:
		panic(fmt.Sprintf("cudaarithm: unknown norm type %d", normType))
	}
}
