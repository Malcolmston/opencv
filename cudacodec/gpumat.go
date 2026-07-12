package cudacodec

import (
	"image"

	cv "github.com/malcolmston/opencv"
)

// GpuMat mirrors OpenCV's cv::cuda::GpuMat, the device-resident matrix that the
// real cudacodec API exchanges with NVDEC/NVENC. In this pure-Go port there is
// no device memory: a GpuMat simply wraps a host-resident [cv.Mat]. The wrapper
// exists so that method signatures match OpenCV's GpuMat-based calls
// ([VideoReader.NextFrame], [VideoWriter.Write]); [GpuMat.Upload] and
// [GpuMat.Download] are plain host copies rather than PCIe transfers.
//
// The zero value is an empty GpuMat ready to receive a frame; construct
// instances with [NewGpuMat] or [NewGpuMatFromMat].
type GpuMat struct {
	mat *cv.Mat
}

// NewGpuMat returns an empty GpuMat. It holds no frame until something is
// uploaded into it, so [GpuMat.Empty] reports true.
func NewGpuMat() *GpuMat {
	return &GpuMat{}
}

// NewGpuMatFromMat returns a GpuMat backed by a deep copy of m, so later
// mutations of m do not affect the GpuMat. A nil or empty m yields an empty
// GpuMat.
func NewGpuMatFromMat(m *cv.Mat) *GpuMat {
	g := &GpuMat{}
	if !m.Empty() {
		g.mat = m.Clone()
	}
	return g
}

// Upload copies a host [cv.Mat] into the GpuMat, mirroring
// cv::cuda::GpuMat::upload. In OpenCV this is a host-to-device transfer; here it
// is a deep copy, so the GpuMat is independent of m afterwards. The variadic
// stream argument is accepted for API compatibility and ignored — see [Stream].
// Uploading a nil or empty Mat clears the GpuMat.
func (g *GpuMat) Upload(m *cv.Mat, _ ...*Stream) {
	if m.Empty() {
		g.mat = nil
		return
	}
	g.mat = m.Clone()
}

// Download copies the GpuMat back to a host [cv.Mat], mirroring
// cv::cuda::GpuMat::download. In OpenCV this is a device-to-host transfer; here
// it is a deep copy, so the returned Mat is independent of the GpuMat. An empty
// GpuMat downloads as nil. The variadic stream argument is ignored.
func (g *GpuMat) Download(_ ...*Stream) *cv.Mat {
	if g.Empty() {
		return nil
	}
	return g.mat.Clone()
}

// Mat returns the GpuMat's underlying host matrix without copying, or nil when
// empty. Callers that mutate the result also mutate the GpuMat; use
// [GpuMat.Download] for an independent copy.
func (g *GpuMat) Mat() *cv.Mat {
	if g == nil {
		return nil
	}
	return g.mat
}

// Clone returns an independent deep copy of the GpuMat.
func (g *GpuMat) Clone() *GpuMat {
	return NewGpuMatFromMat(g.Mat())
}

// Empty reports whether the GpuMat currently holds no frame.
func (g *GpuMat) Empty() bool {
	return g == nil || g.mat.Empty()
}

// Size returns the frame size as an image.Point in OpenCV's (width, height)
// order: X is the number of columns and Y the number of rows. An empty GpuMat
// reports the zero Point.
func (g *GpuMat) Size() image.Point {
	if g.Empty() {
		return image.Point{}
	}
	return image.Pt(g.mat.Cols, g.mat.Rows)
}

// Rows returns the frame height in pixels, or 0 when empty.
func (g *GpuMat) Rows() int {
	if g.Empty() {
		return 0
	}
	return g.mat.Rows
}

// Cols returns the frame width in pixels, or 0 when empty.
func (g *GpuMat) Cols() int {
	if g.Empty() {
		return 0
	}
	return g.mat.Cols
}

// Channels returns the number of samples per pixel, or 0 when empty.
func (g *GpuMat) Channels() int {
	if g.Empty() {
		return 0
	}
	return g.mat.Channels
}

// Release frees the GpuMat's frame, mirroring cv::cuda::GpuMat::release. After
// Release the GpuMat is empty and can be reused as an [Upload] target.
func (g *GpuMat) Release() {
	if g != nil {
		g.mat = nil
	}
}

// Stream mirrors OpenCV's cv::cuda::Stream, the handle that queues asynchronous
// GPU work. This port performs everything synchronously on the CPU, so a Stream
// carries no state and every operation on it completes instantly. It exists only
// so that calls taking a stream argument compile unchanged.
type Stream struct{}

// NewStream returns a no-op [Stream]. The returned value can be passed to any
// method that accepts a stream; all such work has already completed by the time
// the call returns.
func NewStream() *Stream {
	return &Stream{}
}

// WaitForCompletion blocks until all queued work on the stream finishes,
// mirroring cv::cuda::Stream::waitForCompletion. Because this port is
// synchronous, there is never any outstanding work and the call returns
// immediately.
func (s *Stream) WaitForCompletion() {}

// QueryIfComplete reports whether all queued work has finished, mirroring
// cv::cuda::Stream::queryIfComplete. It always returns true here, as work
// completes synchronously.
func (s *Stream) QueryIfComplete() bool {
	return true
}
