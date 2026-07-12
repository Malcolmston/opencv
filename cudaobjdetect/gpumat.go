package cudaobjdetect

import (
	cv "github.com/malcolmston/opencv"
)

// GpuMat is a CPU-backed stand-in for OpenCV's cv::cuda::GpuMat. It has the same
// role in this package's API — the container that detector inputs and outputs
// travel in — but there is no GPU: it simply wraps a host-resident [cv.Mat].
// Upload and Download therefore copy between host buffers rather than crossing a
// PCIe bus, and every operation runs on the CPU.
//
// A GpuMat may additionally carry a decoded detection result (see
// [CascadeClassifier.DetectMultiScale] and [CascadeClassifier.Convert]). OpenCV
// returns detections as a packed CV_32SC4 GpuMat; because the root [cv.Mat] is
// 8-bit only, the integer rectangles are carried out-of-band in an unexported
// field instead of being bit-packed into the 8-bit buffer.
type GpuMat struct {
	mat     *cv.Mat
	objects []cv.Rect // non-nil only for detection-result mats
}

// NewGpuMat allocates a zero-filled GpuMat of the given dimensions, mirroring
// the GpuMat(rows, cols, type) constructor. It panics if any dimension is not
// positive.
func NewGpuMat(rows, cols, channels int) *GpuMat {
	return &GpuMat{mat: cv.NewMat(rows, cols, channels)}
}

// NewGpuMatFromMat returns a GpuMat that shares m's backing storage (no copy),
// the analogue of wrapping an existing device allocation. Use [GpuMat.Clone] or
// [GpuMat.Upload] if independent storage is wanted. It panics if m is nil.
func NewGpuMatFromMat(m *cv.Mat) *GpuMat {
	if m == nil {
		panic("cudaobjdetect: NewGpuMatFromMat given a nil Mat")
	}
	return &GpuMat{mat: m}
}

// Upload copies host data from m into the GpuMat, the analogue of
// cv::cuda::GpuMat::upload. The GpuMat takes an independent deep copy so later
// edits to m are not reflected. It panics if m is nil.
func (g *GpuMat) Upload(m *cv.Mat) {
	if m == nil {
		panic("cudaobjdetect: Upload given a nil Mat")
	}
	g.mat = m.Clone()
	g.objects = nil
}

// Download returns a host [cv.Mat] holding an independent copy of the GpuMat's
// pixels, the analogue of cv::cuda::GpuMat::download. It panics if the GpuMat
// holds no image (for example a pure detection-result mat).
func (g *GpuMat) Download() *cv.Mat {
	if g.mat == nil {
		panic("cudaobjdetect: Download on a GpuMat with no image data")
	}
	return g.mat.Clone()
}

// Mat returns the GpuMat's underlying host [cv.Mat] without copying. It may be
// nil for a detection-result mat produced by [CascadeClassifier.DetectMultiScale].
func (g *GpuMat) Mat() *cv.Mat { return g.mat }

// Empty reports whether the GpuMat holds no image samples.
func (g *GpuMat) Empty() bool { return g == nil || g.mat == nil || g.mat.Empty() }

// Size returns the image dimensions as (rows, cols). It returns (0, 0) for a
// GpuMat with no image data.
func (g *GpuMat) Size() (rows, cols int) {
	if g.mat == nil {
		return 0, 0
	}
	return g.mat.Rows, g.mat.Cols
}

// Channels returns the number of samples per pixel, or 0 for a GpuMat with no
// image data.
func (g *GpuMat) Channels() int {
	if g.mat == nil {
		return 0
	}
	return g.mat.Channels
}

// Clone returns a deep copy of the GpuMat, including any carried detection
// rectangles, with its own backing storage.
func (g *GpuMat) Clone() *GpuMat {
	out := &GpuMat{}
	if g.mat != nil {
		out.mat = g.mat.Clone()
	}
	if g.objects != nil {
		out.objects = make([]cv.Rect, len(g.objects))
		copy(out.objects, g.objects)
	}
	return out
}

// Stream is a CPU-backed no-op analogue of cv::cuda::Stream. OpenCV uses a
// Stream to queue asynchronous work on a device; here every operation is
// synchronous, so a Stream carries no state and its methods do nothing. It is
// accepted throughout the package purely for API compatibility — passing nil is
// always valid and equivalent to the null stream.
type Stream struct{}

// NewStream returns a new no-op Stream.
func NewStream() *Stream { return &Stream{} }

// WaitForCompletion returns immediately: all work in this package is executed
// synchronously, so there is never anything outstanding to wait for.
func (s *Stream) WaitForCompletion() {}

// QueryIfComplete always reports true, since work is never deferred.
func (s *Stream) QueryIfComplete() bool { return true }
