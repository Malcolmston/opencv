package cudacore

import (
	"fmt"

	cv "github.com/malcolmston/opencv"
)

// BorderType selects how [GpuMat.CopyMakeBorder] fabricates samples outside the
// source rectangle. The values match OpenCV's cv::BorderTypes.
type BorderType int

const (
	// BorderConstant pads with a fixed scalar value: iiiiii|abcdefgh|iiiiii.
	BorderConstant BorderType = 0
	// BorderReplicate repeats the edge sample: aaaaaa|abcdefgh|hhhhhh.
	BorderReplicate BorderType = 1
	// BorderReflect mirrors including the edge: fedcba|abcdefgh|hgfedc.
	BorderReflect BorderType = 2
	// BorderWrap tiles the image: cdefgh|abcdefgh|abcdef.
	BorderWrap BorderType = 3
	// BorderReflect101 mirrors excluding the edge: gfedcb|abcdefgh|gfedcb.
	BorderReflect101 BorderType = 4
)

// borderInterpolate maps an out-of-range coordinate p on an axis of the given
// length to the source index dictated by bt, mirroring cv::borderInterpolate.
// For BorderConstant it returns -1, signalling the caller to use the fill value.
func borderInterpolate(p, length int, bt BorderType) int {
	if p >= 0 && p < length {
		return p
	}
	switch bt {
	case BorderReplicate:
		if p < 0 {
			return 0
		}
		return length - 1
	case BorderReflect, BorderReflect101:
		delta := 0
		if bt == BorderReflect101 {
			delta = 1
		}
		if length == 1 {
			return 0
		}
		for p < 0 || p >= length {
			if p < 0 {
				p = -p - 1 + delta
			} else {
				p = 2*length - p - 1 - delta
			}
		}
		return p
	case BorderWrap:
		p %= length
		if p < 0 {
			p += length
		}
		return p
	default: // BorderConstant
		return -1
	}
}

// CopyMakeBorder returns a new GpuMat that is the receiver surrounded by top,
// bottom, left and right extra rows/columns, the analogue of
// cv::cuda::copyMakeBorder. The fabricated samples follow borderType; for
// [BorderConstant] the per-channel components of value fill the border (rounded
// and saturated like [GpuMat.SetTo]). It panics on an empty GpuMat or negative
// border sizes.
func (g *GpuMat) CopyMakeBorder(top, bottom, left, right int, borderType BorderType, value cv.Scalar) *GpuMat {
	requireNonEmpty(g, "CopyMakeBorder")
	if top < 0 || bottom < 0 || left < 0 || right < 0 {
		panic(fmt.Sprintf("cudacore: CopyMakeBorder needs non-negative borders, got t=%d b=%d l=%d r=%d", top, bottom, left, right))
	}
	m := g.mat
	ch := m.Channels
	dstRows := m.Rows + top + bottom
	dstCols := m.Cols + left + right
	out := cv.NewMat(dstRows, dstCols, ch)

	fill := make([]uint8, ch)
	for c := 0; c < ch; c++ {
		fill[c] = roundToUint8(value[c])
	}

	for dy := 0; dy < dstRows; dy++ {
		sy := borderInterpolate(dy-top, m.Rows, borderType)
		for dx := 0; dx < dstCols; dx++ {
			sx := borderInterpolate(dx-left, m.Cols, borderType)
			di := (dy*dstCols + dx) * ch
			if sy < 0 || sx < 0 { // BorderConstant
				copy(out.Data[di:di+ch], fill)
				continue
			}
			si := (sy*m.Cols + sx) * ch
			copy(out.Data[di:di+ch], m.Data[si:si+ch])
		}
	}
	return &GpuMat{mat: out}
}

// EnsureSizeIsEnough guarantees that m has at least the requested rows, cols and
// type, reallocating it only when its current layout does not already match, the
// analogue of cv::cuda::ensureSizeIsEnough. When a fresh buffer is needed it is
// zero-filled. It panics if m is nil or the size is not positive.
func EnsureSizeIsEnough(rows, cols int, typ MatType, m *GpuMat) {
	if m == nil {
		panic("cudacore: EnsureSizeIsEnough given a nil GpuMat")
	}
	if rows <= 0 || cols <= 0 {
		panic(fmt.Sprintf("cudacore: EnsureSizeIsEnough requires positive size, got rows=%d cols=%d", rows, cols))
	}
	if !m.Empty() && m.mat.Rows == rows && m.mat.Cols == cols && m.mat.Channels == typ.Channels() {
		return
	}
	m.mat = cv.NewMat(rows, cols, typ.Channels())
}

// PtrStepSz is a lightweight, read-only descriptor of a [GpuMat]'s raw storage,
// the analogue of cv::cuda::PtrStepSz. Data shares the owning GpuMat's backing
// slice (it is a pointer-like handle, not a copy). Step is the number of samples
// between the starts of consecutive rows, which equals Cols*Channels because
// this CPU-backed container never pads rows.
type PtrStepSz struct {
	// Data is the shared, read-only sample buffer in row-major, channel-
	// interleaved order.
	Data []uint8
	// Rows is the number of rows.
	Rows int
	// Cols is the number of columns.
	Cols int
	// Channels is the number of samples per pixel.
	Channels int
	// Step is the samples-per-row stride (Cols*Channels).
	Step int
}

// HostMem is a CPU-backed stand-in for cv::cuda::HostMem, OpenCV's page-locked
// (pinned) host buffer used to accelerate host/device transfers. Here there is
// no device, so it is simply an owned host [cv.Mat] with no pinning and no
// transfer benefit; it exists so that code allocating pinned staging buffers
// ports without change.
type HostMem struct {
	mat *cv.Mat
}

// NewHostMem allocates a zero-filled HostMem of the given size and type. It
// panics if rows or cols is not positive.
func NewHostMem(rows, cols int, typ MatType) *HostMem {
	if rows <= 0 || cols <= 0 {
		panic(fmt.Sprintf("cudacore: NewHostMem requires positive size, got rows=%d cols=%d", rows, cols))
	}
	return &HostMem{mat: cv.NewMat(rows, cols, typ.Channels())}
}

// CreateMatHeader returns the underlying [cv.Mat] without copying, the analogue
// of cv::cuda::HostMem::createMatHeader. Because host and device memory are the
// same here, this is also the buffer a GpuMat would upload from. It is nil for
// an empty HostMem.
func (h *HostMem) CreateMatHeader() *cv.Mat {
	if h == nil {
		return nil
	}
	return h.mat
}

// Empty reports whether the HostMem holds no samples.
func (h *HostMem) Empty() bool {
	return h == nil || h.mat == nil || h.mat.Empty()
}

// Size returns the dimensions as (rows, cols), or (0, 0) when empty.
func (h *HostMem) Size() (rows, cols int) {
	if h.Empty() {
		return 0, 0
	}
	return h.mat.Rows, h.mat.Cols
}

// GpuMatND is a CPU-backed stand-in for cv::cuda::GpuMatND, an N-dimensional
// device array. The root package only models 2-D CV_8U matrices, so this type
// records the requested shape and channel count for API compatibility but is a
// documented no-op container: it stores no samples and performs no computation.
type GpuMatND struct {
	sizes    []int
	channels int
}

// NewGpuMatND returns a GpuMatND describing an array of the given per-dimension
// sizes and type. The shape is copied. It panics if any size is not positive.
func NewGpuMatND(sizes []int, typ MatType) *GpuMatND {
	dims := make([]int, len(sizes))
	for i, s := range sizes {
		if s <= 0 {
			panic(fmt.Sprintf("cudacore: NewGpuMatND size[%d]=%d must be positive", i, s))
		}
		dims[i] = s
	}
	return &GpuMatND{sizes: dims, channels: typ.Channels()}
}

// Size returns a copy of the per-dimension sizes.
func (g *GpuMatND) Size() []int {
	out := make([]int, len(g.sizes))
	copy(out, g.sizes)
	return out
}

// Channels returns the channel count recorded at construction.
func (g *GpuMatND) Channels() int {
	return g.channels
}

// Dims returns the number of dimensions.
func (g *GpuMatND) Dims() int {
	return len(g.sizes)
}

// Empty reports whether the array has no dimensions.
func (g *GpuMatND) Empty() bool {
	return g == nil || len(g.sizes) == 0
}

// Release clears the recorded shape, leaving the GpuMatND empty, the analogue of
// cv::cuda::GpuMatND::release.
func (g *GpuMatND) Release() {
	g.sizes = nil
	g.channels = 0
}

// BufferPool is a CPU-backed stand-in for cv::cuda::BufferPool, OpenCV's
// per-stream scratch allocator. With no device memory to recycle it is a
// documented no-op: [BufferPool.GetBuffer] simply allocates a fresh [GpuMat]
// each call rather than leasing from a pool.
type BufferPool struct{}

// NewBufferPool returns a BufferPool associated with the given stream. The
// stream is accepted for API compatibility and otherwise ignored.
func NewBufferPool(_ *Stream) *BufferPool {
	return &BufferPool{}
}

// GetBuffer returns a freshly allocated, zero-filled [GpuMat] of the requested
// size and type, the analogue of cv::cuda::BufferPool::getBuffer. Nothing is
// pooled or reused. It panics if rows or cols is not positive.
func (p *BufferPool) GetBuffer(rows, cols int, typ MatType) *GpuMat {
	if rows <= 0 || cols <= 0 {
		panic(fmt.Sprintf("cudacore: GetBuffer requires positive size, got rows=%d cols=%d", rows, cols))
	}
	return &GpuMat{mat: cv.NewMat(rows, cols, typ.Channels())}
}

// SetBufferPoolUsage mirrors cv::cuda::setBufferPoolUsage. Because there is no
// device buffer pool to enable, it is a documented no-op that accepts and
// discards its argument for source compatibility.
func SetBufferPoolUsage(_ bool) {}
