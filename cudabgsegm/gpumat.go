package cudabgsegm

import (
	cv "github.com/malcolmston/opencv"
)

// GpuMat is a CPU-backed stand-in for OpenCV's cv::cuda::GpuMat. In real OpenCV
// a GpuMat holds pixels in device (GPU) memory; here it simply wraps a host-side
// *[cv.Mat]. It exists so that code written against the CUDA API — which passes
// frames and masks as GpuMat values and moves them between host and device with
// Upload/Download — compiles and runs unchanged against this pure-Go port.
//
// The zero value holds a nil Mat and is reported [GpuMat.Empty]. Construct usable
// instances with [NewGpuMat] or [GpuMatFromMat].
type GpuMat struct {
	// Mat is the underlying host matrix. It may be nil for an empty GpuMat.
	Mat *cv.Mat
}

// NewGpuMat returns an empty GpuMat with no backing matrix. It mirrors the
// default-constructed cv::cuda::GpuMat.
func NewGpuMat() *GpuMat {
	return &GpuMat{}
}

// GpuMatFromMat wraps an existing host matrix as a GpuMat without copying it.
// The returned GpuMat shares storage with m; use [GpuMat.Clone] for an
// independent copy. A nil m yields an empty GpuMat.
func GpuMatFromMat(m *cv.Mat) *GpuMat {
	return &GpuMat{Mat: m}
}

// Upload copies host matrix m into this GpuMat, standing in for
// cv::cuda::GpuMat::upload. Because there is no device memory, the wrapped Mat is
// replaced with a deep copy of m (or nil when m is nil). The optional stream is
// accepted for API compatibility and ignored.
func (g *GpuMat) Upload(m *cv.Mat, stream *Stream) {
	_ = stream
	if m == nil {
		g.Mat = nil
		return
	}
	g.Mat = m.Clone()
}

// Download returns a host copy of the wrapped matrix, standing in for
// cv::cuda::GpuMat::download. It returns nil for an empty GpuMat. The optional
// stream is accepted for API compatibility and ignored.
func (g *GpuMat) Download(stream *Stream) *cv.Mat {
	_ = stream
	if g.Mat == nil {
		return nil
	}
	return g.Mat.Clone()
}

// Empty reports whether the GpuMat has no samples, mirroring
// cv::cuda::GpuMat::empty.
func (g *GpuMat) Empty() bool {
	return g == nil || g.Mat.Empty()
}

// Size returns the dimensions of the wrapped matrix as (rows, cols), or (0, 0)
// when the GpuMat is empty.
func (g *GpuMat) Size() (rows, cols int) {
	if g.Empty() {
		return 0, 0
	}
	return g.Mat.Rows, g.Mat.Cols
}

// Channels returns the channel count of the wrapped matrix, or 0 when empty.
func (g *GpuMat) Channels() int {
	if g.Empty() {
		return 0
	}
	return g.Mat.Channels
}

// Clone returns a deep copy of the GpuMat with its own backing matrix, mirroring
// cv::cuda::GpuMat::clone. Cloning an empty GpuMat yields another empty GpuMat.
func (g *GpuMat) Clone() *GpuMat {
	if g.Empty() {
		return &GpuMat{}
	}
	return &GpuMat{Mat: g.Mat.Clone()}
}

// requireFrame extracts the host matrix from a frame GpuMat, panicking with a
// package-tagged message when the frame is nil or empty. It centralises the
// precondition shared by every Apply.
func requireFrame(frame *GpuMat) *cv.Mat {
	if frame == nil || frame.Mat == nil || frame.Mat.Empty() {
		panic("cudabgsegm: Apply given a nil or empty GpuMat frame")
	}
	return frame.Mat
}
