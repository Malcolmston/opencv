package cudaarithm

import (
	"fmt"

	cv "github.com/malcolmston/opencv"
)

// MatType encodes an OpenCV matrix type. Because the root package's [cv.Mat]
// only stores 8-bit unsigned samples, every [GpuMat] has depth CV_8U and its
// type is fully determined by the channel count. The constants below name the
// common cases; [MakeType] builds one for an arbitrary channel count.
type MatType int

// Depth CV_8U is 0 in OpenCV; a type with n channels is n-1 shifted left by the
// depth-bit width (3). These constants match OpenCV's CV_8UC1..CV_8UC4 values.
const (
	// CV_8UC1 is a single-channel 8-bit unsigned matrix (value 0).
	CV_8UC1 MatType = 0
	// CV_8UC2 is a two-channel 8-bit unsigned matrix (value 8).
	CV_8UC2 MatType = 8
	// CV_8UC3 is a three-channel 8-bit unsigned matrix (value 16).
	CV_8UC3 MatType = 16
	// CV_8UC4 is a four-channel 8-bit unsigned matrix (value 24).
	CV_8UC4 MatType = 24
)

// MakeType returns the CV_8U MatType for the given channel count, mirroring
// OpenCV's CV_8UC(n). It panics if channels is not positive.
func MakeType(channels int) MatType {
	if channels <= 0 {
		panic(fmt.Sprintf("cudaarithm: MakeType requires positive channels, got %d", channels))
	}
	return MatType((channels - 1) << 3)
}

// Channels returns the number of channels encoded by the type.
func (t MatType) Channels() int {
	return int(t)>>3 + 1
}

// String renders the type as OpenCV's CV_8UC<n> spelling.
func (t MatType) String() string {
	return fmt.Sprintf("CV_8UC%d", t.Channels())
}

// GpuMat is a CPU-backed stand-in for OpenCV's cv::cuda::GpuMat. It wraps a
// host-resident [cv.Mat]; there is no device memory and no GPU. The wrapped Mat
// is unexported so that callers move data through [GpuMat.Upload] and
// [GpuMat.Download], preserving OpenCV's host/device copy discipline, but it is
// reachable read-only through [GpuMat.Mat] for interop with the root package.
type GpuMat struct {
	mat *cv.Mat
}

// NewGpuMat returns a GpuMat holding a copy of src, mimicking an upload of a
// host matrix to the device. Because there is no device memory the "upload" is
// a deep copy: mutating src afterwards does not affect the GpuMat. A nil src
// yields an empty GpuMat.
func NewGpuMat(src *cv.Mat) *GpuMat {
	g := &GpuMat{}
	if src != nil && !src.Empty() {
		g.mat = src.Clone()
	}
	return g
}

// Upload copies src into the GpuMat, replacing any previous contents. It is the
// analogue of cv::cuda::GpuMat::upload. As there is no device memory the copy is
// a deep clone; it is a no-op transfer in the sense that no bus is crossed, but
// it does duplicate the samples so the GpuMat never aliases src.
func (g *GpuMat) Upload(src *cv.Mat) {
	if src == nil || src.Empty() {
		g.mat = nil
		return
	}
	g.mat = src.Clone()
}

// Download returns a copy of the GpuMat's contents as a host [cv.Mat], the
// analogue of cv::cuda::GpuMat::download. The returned Mat is an independent
// deep copy; it is nil for an empty GpuMat. No actual device-to-host transfer
// occurs because the data already lives in host memory.
func (g *GpuMat) Download() *cv.Mat {
	if g.Empty() {
		return nil
	}
	return g.mat.Clone()
}

// Mat returns the wrapped [cv.Mat] without copying, for read-only interop with
// the root package. Callers must not mutate it; use [GpuMat.Download] for an
// owned copy. It is nil for an empty GpuMat.
func (g *GpuMat) Mat() *cv.Mat {
	return g.mat
}

// Empty reports whether the GpuMat holds no samples.
func (g *GpuMat) Empty() bool {
	return g == nil || g.mat == nil || g.mat.Empty()
}

// Size returns the matrix dimensions as (rows, cols), or (0, 0) when empty.
func (g *GpuMat) Size() (rows, cols int) {
	if g.Empty() {
		return 0, 0
	}
	return g.mat.Rows, g.mat.Cols
}

// Channels returns the number of channels, or 0 when empty.
func (g *GpuMat) Channels() int {
	if g.Empty() {
		return 0
	}
	return g.mat.Channels
}

// Type returns the [MatType] of the matrix. Every GpuMat has CV_8U depth, so
// the type reflects only the channel count. An empty GpuMat reports CV_8UC1.
func (g *GpuMat) Type() MatType {
	if g.Empty() {
		return CV_8UC1
	}
	return MakeType(g.mat.Channels)
}

// Clone returns a deep copy of the GpuMat with its own backing storage.
func (g *GpuMat) Clone() *GpuMat {
	if g.Empty() {
		return &GpuMat{}
	}
	return &GpuMat{mat: g.mat.Clone()}
}

// Release frees the GpuMat's contents, leaving it empty. It mirrors
// cv::cuda::GpuMat::release; here it simply drops the reference so the garbage
// collector can reclaim the samples.
func (g *GpuMat) Release() {
	g.mat = nil
}

// Stream is a CPU-backed stand-in for cv::cuda::Stream. Every operation in this
// package runs synchronously, so a Stream carries no state and enqueues no
// work. It exists solely so that code written against the CUDA API — which
// threads an optional stream through every call — ports without change.
type Stream struct{}

// NewStream returns a ready-to-use no-op Stream.
func NewStream() *Stream {
	return &Stream{}
}

// WaitForCompletion returns immediately. Because operations complete
// synchronously before they return, there is never any outstanding work to wait
// for. It is provided for API compatibility with cv::cuda::Stream.
func (s *Stream) WaitForCompletion() {}

// requireSameShape panics unless a and b have identical dimensions and channel
// counts.
func requireSameShape(a, b *GpuMat, name string) {
	requireNonEmpty(a, name)
	requireNonEmpty(b, name)
	am, bm := a.mat, b.mat
	if am.Rows != bm.Rows || am.Cols != bm.Cols || am.Channels != bm.Channels {
		panic(fmt.Sprintf("cudaarithm: %s shape mismatch %dx%dx%d vs %dx%dx%d",
			name, am.Rows, am.Cols, am.Channels, bm.Rows, bm.Cols, bm.Channels))
	}
}

// requireNonEmpty panics if g is empty.
func requireNonEmpty(g *GpuMat, name string) {
	if g.Empty() {
		panic(fmt.Sprintf("cudaarithm: %s given an empty GpuMat", name))
	}
}

// requireChannels panics unless g has exactly want channels.
func requireChannels(g *GpuMat, want int, name string) {
	requireNonEmpty(g, name)
	if g.mat.Channels != want {
		panic(fmt.Sprintf("cudaarithm: %s requires %d channel(s), got %d", name, want, g.mat.Channels))
	}
}

// clampToUint8 rounds toward zero (the caller adds any rounding bias) and clamps
// into [0,255], mirroring the root package's helper so results agree at the
// boundary.
func clampToUint8(v float64) uint8 {
	if v <= 0 {
		return 0
	}
	if v >= 255 {
		return 255
	}
	return uint8(v)
}

// roundToUint8 rounds v to the nearest integer with the root package's +0.5 bias
// and saturates into [0,255].
func roundToUint8(v float64) uint8 {
	return clampToUint8(v + 0.5)
}
