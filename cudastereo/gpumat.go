package cudastereo

import (
	"fmt"

	cv "github.com/malcolmston/opencv"
)

// GpuMat is a CPU-backed stand-in for cv::cuda::GpuMat: the device-resident
// matrix type of OpenCV's CUDA modules. Because this port has no GPU, a GpuMat
// simply wraps a host [github.com/malcolmston/opencv.Mat]; "uploading" and
// "downloading" are ordinary deep copies. The zero value is an empty matrix
// (see [GpuMat.Empty]); build instances with [NewGpuMat], [NewGpuMatFromMat] or
// [GpuMat.Upload].
type GpuMat struct {
	// mat is the host-side backing matrix, or nil when the GpuMat is empty.
	mat *cv.Mat
}

// NewGpuMat allocates a zero-filled device matrix of the given shape, mirroring
// the cv::cuda::GpuMat(rows, cols, type) constructor. It panics if any dimension
// is not positive.
func NewGpuMat(rows, cols, channels int) *GpuMat {
	return &GpuMat{mat: cv.NewMat(rows, cols, channels)}
}

// NewGpuMatFromMat returns a device matrix holding an independent copy of the
// host matrix m, the equivalent of default-constructing a GpuMat and calling
// upload. It panics if m is nil or empty.
func NewGpuMatFromMat(m *cv.Mat) *GpuMat {
	g := &GpuMat{}
	g.Upload(m)
	return g
}

// Upload copies the host matrix host into this device matrix, replacing any
// previous contents, like cv::cuda::GpuMat::upload. The copy is deep, so later
// edits to host do not affect the GpuMat. It panics if host is nil or empty.
func (g *GpuMat) Upload(host *cv.Mat) {
	if host == nil || host.Empty() {
		panic("cudastereo: GpuMat.Upload given a nil or empty Mat")
	}
	g.mat = host.Clone()
}

// Download returns an independent host copy of the device matrix, like
// cv::cuda::GpuMat::download. It panics if the GpuMat is empty.
func (g *GpuMat) Download() *cv.Mat {
	if g.Empty() {
		panic("cudastereo: GpuMat.Download on an empty GpuMat")
	}
	return g.mat.Clone()
}

// Mat returns the underlying host matrix without copying. The result aliases the
// GpuMat's storage; callers that intend to mutate it should [GpuMat.Download]
// instead. It returns nil when the GpuMat is empty.
func (g *GpuMat) Mat() *cv.Mat {
	return g.mat
}

// Empty reports whether the device matrix holds no samples.
func (g *GpuMat) Empty() bool {
	return g == nil || g.mat == nil || g.mat.Empty()
}

// Rows returns the number of rows (height), or 0 when empty.
func (g *GpuMat) Rows() int {
	if g.Empty() {
		return 0
	}
	return g.mat.Rows
}

// Cols returns the number of columns (width), or 0 when empty.
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

// Size returns the device matrix dimensions as (rows, cols).
func (g *GpuMat) Size() (rows, cols int) {
	return g.Rows(), g.Cols()
}

// Clone returns a deep copy of the device matrix with its own backing storage.
// Cloning an empty GpuMat yields another empty GpuMat.
func (g *GpuMat) Clone() *GpuMat {
	if g.Empty() {
		return &GpuMat{}
	}
	return &GpuMat{mat: g.mat.Clone()}
}

// Release frees the device matrix, leaving it empty, like
// cv::cuda::GpuMat::release. After Release the GpuMat may be reused via
// [GpuMat.Upload].
func (g *GpuMat) Release() {
	g.mat = nil
}

// Stream is a CPU-backed placeholder for cv::cuda::Stream. In a CUDA build a
// stream sequences asynchronous device work; here every operation runs
// synchronously on the CPU, so a Stream carries no state and exists only to keep
// call signatures API-compatible. A nil *Stream is always acceptable wherever a
// stream is expected.
type Stream struct{}

// NewStream returns a ready-to-use stream. It never fails.
func NewStream() *Stream {
	return &Stream{}
}

// WaitForCompletion blocks until all work queued on the stream has finished,
// like cv::cuda::Stream::waitForCompletion. Because operations here are already
// synchronous it returns immediately. It is safe to call on a nil *Stream.
func (s *Stream) WaitForCompletion() {}

// matOf returns the host matrix backing a required, non-empty GpuMat argument,
// panicking with a descriptive message otherwise.
func matOf(g *GpuMat, name string) *cv.Mat {
	if g == nil || g.Empty() {
		panic(fmt.Sprintf("cudastereo: %s must be a non-empty GpuMat", name))
	}
	return g.mat
}

// grayGrid extracts a single-channel intensity grid from m as a flat []int in
// row-major order, converting three-channel input with the root package's
// RGB->Gray. It panics on empty or unsupported input.
func grayGrid(m *cv.Mat) (rows, cols int, g []int) {
	if m == nil || m.Empty() {
		panic("cudastereo: nil or empty input Mat")
	}
	src := m
	switch m.Channels {
	case 1:
		// already grayscale
	case 3:
		src = cv.CvtColor(m, cv.ColorRGB2Gray)
	default:
		panic(fmt.Sprintf("cudastereo: input must be 1- or 3-channel, got %d", m.Channels))
	}
	rows, cols = src.Rows, src.Cols
	g = make([]int, rows*cols)
	for i := range g {
		g[i] = int(src.Data[i])
	}
	return rows, cols, g
}

// clampInt clamps v to [lo, hi].
func clampInt(v, lo, hi int) int {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

// absInt returns the absolute value of v.
func absInt(v int) int {
	if v < 0 {
		return -v
	}
	return v
}
