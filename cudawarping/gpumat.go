package cudawarping

import (
	cv "github.com/malcolmston/opencv"
)

// GpuMat is a CPU-backed stand-in for OpenCV's cv::cuda::GpuMat. In a real CUDA
// build a GpuMat holds pixels in device (GPU) memory and warping kernels run on
// the GPU; here the pixels live in ordinary host memory inside a wrapped
// [cv.Mat] and every operation is computed on the CPU. The type exists so that
// code written against the cudawarping API surface compiles and produces the
// same results as the GPU module, at CPU speed and with no cgo or GPU
// dependency.
//
// The zero value is an empty GpuMat. Construct instances with [NewGpuMat] or by
// [Upload]-ing a [cv.Mat].
type GpuMat struct {
	// mat is the host-memory backing store. It is nil for an empty GpuMat.
	mat *cv.Mat
}

// NewGpuMat allocates a zero-filled GpuMat with the given dimensions, mirroring
// the cv::cuda::GpuMat(rows, cols, type) constructor. It panics if any
// dimension is not positive.
func NewGpuMat(rows, cols, channels int) *GpuMat {
	return &GpuMat{mat: cv.NewMat(rows, cols, channels)}
}

// Upload copies a host [cv.Mat] into a new GpuMat, mirroring
// cv::cuda::GpuMat::upload. In this CPU port the copy is a deep clone of the
// host matrix, so later edits to src do not affect the GpuMat. It panics if src
// is nil or empty.
func Upload(src *cv.Mat) *GpuMat {
	if src == nil || src.Empty() {
		panic("cudawarping: Upload requires a non-empty source Mat")
	}
	return &GpuMat{mat: src.Clone()}
}

// Upload replaces the contents of g with a deep copy of the host matrix src,
// mirroring cv::cuda::GpuMat::upload on an existing GpuMat. It panics if src is
// nil or empty.
func (g *GpuMat) Upload(src *cv.Mat) {
	if src == nil || src.Empty() {
		panic("cudawarping: Upload requires a non-empty source Mat")
	}
	g.mat = src.Clone()
}

// Download copies the GpuMat back to a host [cv.Mat], mirroring
// cv::cuda::GpuMat::download. The returned Mat is an independent deep copy. It
// panics if g is empty.
func (g *GpuMat) Download() *cv.Mat {
	if g.Empty() {
		panic("cudawarping: Download on an empty GpuMat")
	}
	return g.mat.Clone()
}

// Empty reports whether the GpuMat holds no samples, mirroring
// cv::cuda::GpuMat::empty.
func (g *GpuMat) Empty() bool {
	return g == nil || g.mat == nil || g.mat.Empty()
}

// Size returns the GpuMat dimensions as (rows, cols), matching [cv.Mat.Size].
// It returns (0, 0) for an empty GpuMat.
func (g *GpuMat) Size() (rows, cols int) {
	if g.Empty() {
		return 0, 0
	}
	return g.mat.Rows, g.mat.Cols
}

// Channels returns the number of samples per pixel, or 0 for an empty GpuMat.
func (g *GpuMat) Channels() int {
	if g.Empty() {
		return 0
	}
	return g.mat.Channels
}

// Clone returns a deep copy of the GpuMat with its own backing storage,
// mirroring cv::cuda::GpuMat::clone.
func (g *GpuMat) Clone() *GpuMat {
	if g.Empty() {
		return &GpuMat{}
	}
	return &GpuMat{mat: g.mat.Clone()}
}

// Release frees the GpuMat's backing store, mirroring
// cv::cuda::GpuMat::release. After Release the GpuMat is empty and may be
// re-used via [GpuMat.Upload].
func (g *GpuMat) Release() {
	g.mat = nil
}

// host returns the backing Mat, panicking with the given operation name if the
// GpuMat is empty. It is the common precondition check for every warp method.
func (g *GpuMat) host(op string) *cv.Mat {
	if g.Empty() {
		panic("cudawarping: " + op + " on an empty GpuMat")
	}
	return g.mat
}

// Stream is a CPU-backed stand-in for cv::cuda::Stream, the handle that
// schedules asynchronous work on a CUDA stream. Because every operation in this
// port runs synchronously on the CPU, a Stream carries no state and every
// method is a no-op; the type exists only so that code passing a stream through
// the API compiles unchanged. A nil *Stream is valid everywhere and denotes the
// default (null) stream.
type Stream struct{}

// NewStream returns a new no-op [Stream].
func NewStream() *Stream { return &Stream{} }

// WaitForCompletion returns immediately: in this CPU port all work has already
// finished synchronously by the time a Stream would be waited on. It mirrors
// cv::cuda::Stream::waitForCompletion.
func (s *Stream) WaitForCompletion() {}

// QueryIfComplete always reports true, since work is never outstanding in the
// CPU port. It mirrors cv::cuda::Stream::queryIfComplete.
func (s *Stream) QueryIfComplete() bool { return true }
