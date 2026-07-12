package cudaoptflow

import (
	cv "github.com/malcolmston/opencv"
)

// GpuMat is a CPU-backed stand-in for cv::cuda::GpuMat. It wraps a single
// *cv.Mat living in ordinary host memory; despite the name there is no device
// buffer and no host/device transfer cost. It exists so that optical-flow code
// ported from OpenCV's cuda bindings keeps the same shape.
//
// The zero value is not usable; construct one with [NewGpuMat] or
// [GpuMatFromMat], or receive one from [EncodeFlow]/[FlowToGpuMat].
type GpuMat struct {
	mat *cv.Mat
}

// NewGpuMat allocates a zero-filled GpuMat with the given dimensions, mirroring
// GpuMat(rows, cols, type). It panics on non-positive dimensions.
func NewGpuMat(rows, cols, channels int) *GpuMat {
	return &GpuMat{mat: cv.NewMat(rows, cols, channels)}
}

// GpuMatFromMat uploads a host image into a new GpuMat, mirroring
// GpuMat::upload / the GpuMat(Mat) constructor. The source is cloned so later
// mutation of m does not affect the returned GpuMat. It panics if m is nil or
// empty.
func GpuMatFromMat(m *cv.Mat) *GpuMat {
	if m == nil || m.Empty() {
		panic("cudaoptflow: GpuMatFromMat requires a non-empty Mat")
	}
	return &GpuMat{mat: m.Clone()}
}

// Upload copies a host image into the receiver, mirroring GpuMat::upload. The
// source is cloned. It panics if m is nil or empty.
func (g *GpuMat) Upload(m *cv.Mat) {
	if m == nil || m.Empty() {
		panic("cudaoptflow: Upload requires a non-empty Mat")
	}
	g.mat = m.Clone()
}

// Download returns a host copy of the contents, mirroring GpuMat::download. The
// returned Mat is an independent clone. It panics on an empty GpuMat.
func (g *GpuMat) Download() *cv.Mat {
	if g.Empty() {
		panic("cudaoptflow: Download on an empty GpuMat")
	}
	return g.mat.Clone()
}

// Mat returns the underlying *cv.Mat without copying. Because a GpuMat is
// CPU-backed the buffer is directly accessible; mutating it mutates the GpuMat.
// It returns nil for an empty GpuMat.
func (g *GpuMat) Mat() *cv.Mat {
	return g.mat
}

// Empty reports whether the GpuMat holds no samples.
func (g *GpuMat) Empty() bool {
	return g == nil || g.mat == nil || g.mat.Empty()
}

// Size returns the number of rows and columns, mirroring GpuMat::size. An empty
// GpuMat reports (0, 0).
func (g *GpuMat) Size() (rows, cols int) {
	if g.Empty() {
		return 0, 0
	}
	return g.mat.Rows, g.mat.Cols
}

// Channels returns the number of samples per pixel. An empty GpuMat reports 0.
func (g *GpuMat) Channels() int {
	if g.Empty() {
		return 0
	}
	return g.mat.Channels
}

// Clone returns a deep copy of the GpuMat, mirroring GpuMat::clone.
func (g *GpuMat) Clone() *GpuMat {
	if g.Empty() {
		return &GpuMat{}
	}
	return &GpuMat{mat: g.mat.Clone()}
}

// Release frees the underlying buffer, mirroring GpuMat::release. After Release
// the GpuMat reports Empty.
func (g *GpuMat) Release() {
	g.mat = nil
}

// Stream is a no-op stand-in for cv::cuda::Stream. Every operation in this
// package runs synchronously on the CPU, so a Stream carries no state and its
// methods do nothing; the type exists only so ported call sites can keep passing
// a stream argument. A nil *Stream is a valid "default stream".
type Stream struct{}

// NewStream returns a new no-op Stream, mirroring the cv::cuda::Stream
// constructor.
func NewStream() *Stream {
	return &Stream{}
}

// WaitForCompletion returns immediately. There is nothing asynchronous to wait
// for; the method exists to mirror cv::cuda::Stream::waitForCompletion.
func (s *Stream) WaitForCompletion() {}

// QueryIfComplete always reports true: because work is synchronous, any Stream
// is always idle. It mirrors cv::cuda::Stream::queryIfComplete.
func (s *Stream) QueryIfComplete() bool { return true }
