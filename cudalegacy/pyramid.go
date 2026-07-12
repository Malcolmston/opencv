package cudalegacy

import (
	cv "github.com/malcolmston/opencv"
)

// ImagePyramid is a CPU-backed mirror of OpenCV's cv::cuda::ImagePyramid. It
// precomputes a Gaussian image pyramid — each level is the previous level
// smoothed and halved with [cv.PyrDown] — and lets callers read back a layer at
// an arbitrary requested size, resampling from the nearest stored level with
// [cv.Resize].
//
// Construct one with [NewImagePyramid]; the zero value is not usable.
type ImagePyramid struct {
	levels []*cv.Mat
}

// NewImagePyramid builds a pyramid of up to numLayers levels from img, mirroring
// the ImagePyramid constructor. Level 0 is a copy of img and each subsequent
// level halves the previous one with [cv.PyrDown]. Construction stops early if a
// level would shrink below 1×1. numLayers <= 0 is treated as 1. It panics on a
// nil or empty frame. The stream is a no-op, accepted for API compatibility.
func NewImagePyramid(img *GpuMat, numLayers int, stream *Stream) *ImagePyramid {
	_ = stream
	src := requireMat(img, "NewImagePyramid")
	if numLayers <= 0 {
		numLayers = 1
	}
	levels := make([]*cv.Mat, 0, numLayers)
	levels = append(levels, src.Clone())
	for len(levels) < numLayers {
		prev := levels[len(levels)-1]
		if prev.Rows < 2 || prev.Cols < 2 {
			break
		}
		levels = append(levels, cv.PyrDown(prev))
	}
	return &ImagePyramid{levels: levels}
}

// NumLayers returns the number of stored pyramid levels.
func (p *ImagePyramid) NumLayers() int { return len(p.levels) }

// GetLayer returns a fresh [GpuMat] holding the pyramid content resampled to
// outRows×outCols, mirroring cv::cuda::ImagePyramid::getLayer. The layer whose
// stored size is closest to the request (by total pixel count) is chosen as the
// source and resized with bilinear interpolation. It panics on non-positive
// dimensions. The stream is a no-op.
func (p *ImagePyramid) GetLayer(outRows, outCols int, stream *Stream) *GpuMat {
	_ = stream
	if outRows <= 0 || outCols <= 0 {
		panic("cudalegacy: ImagePyramid.GetLayer requires positive dimensions")
	}
	want := outRows * outCols
	best := p.levels[0]
	bestDiff := -1
	for _, lv := range p.levels {
		diff := lv.Rows*lv.Cols - want
		if diff < 0 {
			diff = -diff
		}
		if bestDiff < 0 || diff < bestDiff {
			bestDiff = diff
			best = lv
		}
	}
	if best.Rows == outRows && best.Cols == outCols {
		return GpuMatFromMat(best.Clone())
	}
	return GpuMatFromMat(cv.Resize(best, outCols, outRows, cv.InterLinear))
}

// Layer returns a copy of stored pyramid level i (level 0 is the original), as a
// convenience beyond the OpenCV surface. It panics if i is out of range.
func (p *ImagePyramid) Layer(i int) *GpuMat {
	if i < 0 || i >= len(p.levels) {
		panic("cudalegacy: ImagePyramid.Layer index out of range")
	}
	return GpuMatFromMat(p.levels[i].Clone())
}
