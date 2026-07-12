package cudafilters

import (
	"image"

	cv "github.com/malcolmston/opencv"
)

// GpuMat is a CPU-backed stand-in for OpenCV's cv::cuda::GpuMat. In real
// cudafilters a GpuMat references memory on the GPU; here it simply wraps an
// ordinary [cv.Mat] held in host memory. It exists so that code written against
// the CUDA filter API — which passes GpuMat values into [Filter.Apply] — compiles
// and runs unchanged.
//
// The wrapped matrix is unexported so that the only ways to move data across the
// (simulated) host⇆device boundary are [GpuMat.Upload] and [GpuMat.Download],
// mirroring the discipline the real API enforces. Both perform a deep copy, so a
// GpuMat never aliases the caller's [cv.Mat].
type GpuMat struct {
	mat *cv.Mat
}

// NewGpuMat returns an empty [GpuMat] holding no matrix. Populate it with
// [GpuMat.Upload], or use [GpuMatFromMat] to wrap an existing [cv.Mat] in one
// step.
func NewGpuMat() *GpuMat {
	return &GpuMat{}
}

// GpuMatFromMat returns a [GpuMat] whose contents are a deep copy of m. It is a
// convenience for NewGpuMat followed by Upload. A nil m yields an empty GpuMat.
func GpuMatFromMat(m *cv.Mat) *GpuMat {
	g := &GpuMat{}
	g.Upload(m)
	return g
}

// Upload copies m into the GpuMat, modelling a host-to-device transfer. The copy
// is deep, so later mutations of m do not affect the GpuMat. Uploading a nil or
// empty Mat leaves the GpuMat empty.
func (g *GpuMat) Upload(m *cv.Mat) {
	if m == nil || m.Empty() {
		g.mat = nil
		return
	}
	g.mat = m.Clone()
}

// Download returns a deep copy of the GpuMat's contents as a [cv.Mat], modelling
// a device-to-host transfer. It returns nil when the GpuMat is empty. The result
// is independent of the GpuMat.
func (g *GpuMat) Download() *cv.Mat {
	if g.Empty() {
		return nil
	}
	return g.mat.Clone()
}

// Empty reports whether the GpuMat holds no usable matrix.
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

// Channels returns the number of samples per pixel, or 0 when empty.
func (g *GpuMat) Channels() int {
	if g.Empty() {
		return 0
	}
	return g.mat.Channels
}

// Clone returns a deep copy of the GpuMat with its own backing storage.
func (g *GpuMat) Clone() *GpuMat {
	if g.Empty() {
		return &GpuMat{}
	}
	return &GpuMat{mat: g.mat.Clone()}
}

// Release frees the GpuMat's contents, leaving it empty. It models cv::cuda::
// GpuMat::release; after the call [GpuMat.Empty] reports true.
func (g *GpuMat) Release() {
	g.mat = nil
}

// Stream is a no-op stand-in for OpenCV's cv::cuda::Stream. On a GPU a Stream
// orders asynchronous work; here every operation is synchronous, so a Stream
// carries no state and its methods do nothing. It is accepted by [Filter.Apply]
// purely for source compatibility.
type Stream struct{}

// NewStream returns a ready-to-use [Stream]. It never fails.
func NewStream() *Stream {
	return &Stream{}
}

// WaitForCompletion returns immediately: all work in this package is synchronous,
// so there is never anything to wait for.
func (s *Stream) WaitForCompletion() {}

// Release does nothing; a [Stream] owns no resources.
func (s *Stream) Release() {}

// AnchorCenter is the anchor sentinel selecting the kernel centre, equal to
// image.Pt(-1, -1). It matches OpenCV's default anchor. The underlying engine
// only supports centred anchors.
var AnchorCenter = image.Pt(-1, -1)

// isCenterAnchor reports whether anchor requests the (default) kernel centre.
func isCenterAnchor(anchor image.Point) bool {
	return anchor.X < 0 && anchor.Y < 0
}
