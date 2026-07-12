package cudafeatures2d

import (
	cv "github.com/malcolmston/opencv"
)

// GpuMat is a CPU-backed stand-in for cv::cuda::GpuMat. It wraps a host *cv.Mat
// and provides the upload/download surface of the GPU type without any device
// memory. All operations execute synchronously on the host.
//
// The zero value is an empty matrix; construct a populated one with [NewGpuMat]
// or by calling [GpuMat.Upload] on a fresh value.
type GpuMat struct {
	mat *cv.Mat
}

// NewGpuMat returns a GpuMat holding a deep copy of the host matrix m, mirroring
// a host-to-device upload. Passing a nil or empty m yields an empty GpuMat.
func NewGpuMat(m *cv.Mat) *GpuMat {
	g := &GpuMat{}
	if m != nil && !m.Empty() {
		g.mat = m.Clone()
	}
	return g
}

// Upload copies the host matrix m into the GpuMat, replacing any current
// contents. It mirrors cv::cuda::GpuMat::upload. A nil or empty m clears the
// GpuMat.
func (g *GpuMat) Upload(m *cv.Mat) {
	if m == nil || m.Empty() {
		g.mat = nil
		return
	}
	g.mat = m.Clone()
}

// Download returns a deep copy of the GpuMat's contents as a host *cv.Mat,
// mirroring cv::cuda::GpuMat::download. It returns nil for an empty GpuMat.
func (g *GpuMat) Download() *cv.Mat {
	if g == nil || g.mat == nil {
		return nil
	}
	return g.mat.Clone()
}

// Empty reports whether the GpuMat holds no samples.
func (g *GpuMat) Empty() bool {
	return g == nil || g.mat == nil || g.mat.Empty()
}

// Rows returns the number of rows, or 0 when empty.
func (g *GpuMat) Rows() int {
	if g.Empty() {
		return 0
	}
	return g.mat.Rows
}

// Cols returns the number of columns, or 0 when empty.
func (g *GpuMat) Cols() int {
	if g.Empty() {
		return 0
	}
	return g.mat.Cols
}

// Channels returns the number of channels, or 0 when empty.
func (g *GpuMat) Channels() int {
	if g.Empty() {
		return 0
	}
	return g.mat.Channels
}

// Size returns the matrix dimensions as (rows, cols), both 0 when empty.
func (g *GpuMat) Size() (rows, cols int) {
	if g.Empty() {
		return 0, 0
	}
	return g.mat.Rows, g.mat.Cols
}

// Clone returns an independent deep copy of the GpuMat.
func (g *GpuMat) Clone() *GpuMat {
	if g.Empty() {
		return &GpuMat{}
	}
	return &GpuMat{mat: g.mat.Clone()}
}

// host returns the wrapped host matrix without copying, or nil when empty. It is
// used internally to feed the delegated CPU implementations.
func (g *GpuMat) host() *cv.Mat {
	if g == nil {
		return nil
	}
	return g.mat
}

// Stream is a CPU-backed no-op stand-in for cv::cuda::Stream. Because this
// package runs synchronously on the host, a Stream carries no state and every
// "asynchronous" call completes before it returns.
type Stream struct{}

// NewStream returns a new (inert) Stream.
func NewStream() *Stream {
	return &Stream{}
}

// WaitForCompletion returns immediately: all work in this package is already
// synchronous, so there is nothing to wait for. It mirrors
// cv::cuda::Stream::waitForCompletion.
func (s *Stream) WaitForCompletion() {}
