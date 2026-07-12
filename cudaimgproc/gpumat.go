package cudaimgproc

import cv "github.com/malcolmston/opencv"

// Stream is a CPU no-op stand-in for OpenCV's cuda::Stream. In the native CUDA
// module a Stream queues asynchronous kernel launches; here nothing is
// scheduled and every operation runs synchronously on the CPU. It exists so
// that call sites written against the cuda API — which pass a Stream as the
// final argument — compile and run unchanged. Its methods are all no-ops.
type Stream struct {
	// valid distinguishes a real (non-null) stream from the null stream. The
	// zero Stream is the null stream.
	valid bool
}

// NewStream returns a ready, non-null [Stream]. Because all work is synchronous
// the returned stream never needs waiting on.
func NewStream() Stream { return Stream{valid: true} }

// WaitForCompletion returns immediately: CPU work has already finished by the
// time any function that accepted this stream returned.
func (s Stream) WaitForCompletion() {}

// QueryIfComplete always reports true, since there is never pending work.
func (s Stream) QueryIfComplete() bool { return true }

// Empty reports whether this is the null stream. The zero Stream is null.
func (s Stream) Empty() bool { return !s.valid }

// firstStream returns the first stream from a variadic argument list, or the
// null stream when none was supplied. It is used only to accept and ignore the
// trailing Stream argument that mirrors the CUDA calling convention.
func firstStream(streams []Stream) Stream {
	if len(streams) > 0 {
		return streams[0]
	}
	return Stream{}
}

// GpuMat is a CPU-backed analogue of OpenCV's cuda::GpuMat. It wraps a
// host-resident [cv.Mat]; there is no device memory involved. GpuMat values
// share their underlying Mat when copied (like OpenCV's reference-counted
// GpuMat), so passing a GpuMat by value to the functions in this package does
// not copy pixel data.
//
// Construct an empty GpuMat with [NewGpuMat] (or the zero value) and fill it
// with [GpuMat.Upload]; retrieve results with [GpuMat.Download].
type GpuMat struct {
	// mat is the host image standing in for device memory. A nil mat means the
	// GpuMat is empty.
	mat *cv.Mat
}

// NewGpuMat returns an empty [GpuMat]. Upload data into it with
// [GpuMat.Upload].
func NewGpuMat() GpuMat { return GpuMat{} }

// NewGpuMatWithSize returns a zero-filled [GpuMat] of the given dimensions,
// mirroring the sized cuda::GpuMat constructor. It panics on non-positive
// dimensions.
func NewGpuMatWithSize(rows, cols, channels int) GpuMat {
	return GpuMat{mat: cv.NewMat(rows, cols, channels)}
}

// wrap builds a GpuMat around an existing Mat without copying. A nil Mat yields
// an empty GpuMat.
func wrap(m *cv.Mat) GpuMat { return GpuMat{mat: m} }

// Upload copies a host [cv.Mat] into this GpuMat, replacing any previous
// contents. In OpenCV this is the host-to-device transfer; here it stores a
// clone so later host-side mutation of src does not alias the GpuMat. The
// trailing Stream argument is accepted and ignored. It panics if src is nil or
// empty.
func (g *GpuMat) Upload(src *cv.Mat, streams ...Stream) {
	_ = firstStream(streams)
	if src == nil || src.Empty() {
		panic("cudaimgproc: Upload of nil or empty Mat")
	}
	g.mat = src.Clone()
}

// Download copies this GpuMat back to a fresh host [cv.Mat], mirroring the
// device-to-host transfer. The trailing Stream argument is accepted and
// ignored. It panics if the GpuMat is empty.
func (g GpuMat) Download(streams ...Stream) *cv.Mat {
	_ = firstStream(streams)
	if g.Empty() {
		panic("cudaimgproc: Download of empty GpuMat")
	}
	return g.mat.Clone()
}

// Empty reports whether the GpuMat holds no image.
func (g GpuMat) Empty() bool { return g.mat == nil || g.mat.Empty() }

// Size returns the matrix dimensions as (rows, cols). It panics if the GpuMat
// is empty.
func (g GpuMat) Size() (rows, cols int) {
	if g.Empty() {
		panic("cudaimgproc: Size of empty GpuMat")
	}
	return g.mat.Rows, g.mat.Cols
}

// Channels returns the number of channels of the wrapped image, or 0 when
// empty.
func (g GpuMat) Channels() int {
	if g.Empty() {
		return 0
	}
	return g.mat.Channels
}

// Clone returns a deep copy of the GpuMat with its own backing storage.
func (g GpuMat) Clone() GpuMat {
	if g.Empty() {
		return GpuMat{}
	}
	return GpuMat{mat: g.mat.Clone()}
}

// Release frees the GpuMat's storage, leaving it empty. After Release the
// GpuMat reports [GpuMat.Empty] as true.
func (g *GpuMat) Release() { g.mat = nil }

// requireHost returns the wrapped Mat, panicking with the given operation name
// when the GpuMat is empty.
func (g GpuMat) requireHost(op string) *cv.Mat {
	if g.Empty() {
		panic("cudaimgproc: " + op + " on empty GpuMat")
	}
	return g.mat
}
