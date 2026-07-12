package cudacore

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
		panic(fmt.Sprintf("cudacore: MakeType requires positive channels, got %d", channels))
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
// a deep copy: mutating src afterwards does not affect the GpuMat. A nil or
// empty src yields an empty GpuMat.
func NewGpuMat(src *cv.Mat) *GpuMat {
	g := &GpuMat{}
	if src != nil && !src.Empty() {
		g.mat = src.Clone()
	}
	return g
}

// Upload copies src into the GpuMat, replacing any previous contents. It is the
// analogue of cv::cuda::GpuMat::upload. As there is no device memory the copy is
// a deep clone; no bus is crossed, but the samples are duplicated so the GpuMat
// never aliases src. A nil or empty src leaves the GpuMat empty.
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

// Create allocates fresh zero-filled storage of the given size and type,
// replacing any previous contents. It mirrors cv::cuda::GpuMat::create. It
// panics if rows or cols is not positive.
func (g *GpuMat) Create(rows, cols int, typ MatType) {
	if rows <= 0 || cols <= 0 {
		panic(fmt.Sprintf("cudacore: Create requires positive size, got rows=%d cols=%d", rows, cols))
	}
	g.mat = cv.NewMat(rows, cols, typ.Channels())
}

// Release frees the GpuMat's contents, leaving it empty. It mirrors
// cv::cuda::GpuMat::release; here it simply drops the reference so the garbage
// collector can reclaim the samples.
func (g *GpuMat) Release() {
	g.mat = nil
}

// Clone returns a deep copy of the GpuMat with its own backing storage,
// mirroring cv::cuda::GpuMat::clone.
func (g *GpuMat) Clone() *GpuMat {
	if g.Empty() {
		return &GpuMat{}
	}
	return &GpuMat{mat: g.mat.Clone()}
}

// CopyTo copies the entire matrix into dst, reallocating dst to match, the
// analogue of cv::cuda::GpuMat::copyTo. dst ends up an independent deep copy; it
// does not alias the receiver. It panics if dst is nil.
func (g *GpuMat) CopyTo(dst *GpuMat) {
	if dst == nil {
		panic("cudacore: CopyTo given a nil destination")
	}
	if g.Empty() {
		dst.mat = nil
		return
	}
	dst.mat = g.mat.Clone()
}

// SetTo fills every pixel with the scalar value, one component per channel,
// rounding and saturating each component into [0,255]. Components beyond the
// channel count are ignored; missing components default to zero. It returns the
// receiver for chaining and mirrors cv::cuda::GpuMat::setTo. It panics on an
// empty GpuMat.
func (g *GpuMat) SetTo(value cv.Scalar) *GpuMat {
	requireNonEmpty(g, "SetTo")
	ch := g.mat.Channels
	fill := make([]uint8, ch)
	for c := 0; c < ch; c++ {
		fill[c] = roundToUint8(value[c])
	}
	for i := 0; i < len(g.mat.Data); i++ {
		g.mat.Data[i] = fill[i%ch]
	}
	return g
}

// ConvertTo returns a new GpuMat whose every sample is saturate(round(src*alpha
// + beta)), the analogue of cv::cuda::GpuMat::convertTo restricted to the CV_8U
// depth this package supports. Unlike [cv.ConvertScaleAbs] the result is not
// taken in absolute value: negative results clamp to 0. It panics on an empty
// GpuMat.
func (g *GpuMat) ConvertTo(alpha, beta float64) *GpuMat {
	requireNonEmpty(g, "ConvertTo")
	out := cv.NewMat(g.mat.Rows, g.mat.Cols, g.mat.Channels)
	for i, v := range g.mat.Data {
		out.Data[i] = roundToUint8(float64(v)*alpha + beta)
	}
	return &GpuMat{mat: out}
}

// Reshape returns a new GpuMat that reinterprets the same samples with a
// different channel count and/or row count, the analogue of
// cv::cuda::GpuMat::reshape. A cn of 0 keeps the current channel count; a rows
// of 0 keeps the current row count. The total number of samples must divide
// evenly into the requested layout, otherwise it panics. Unlike OpenCV, which
// returns a header sharing memory, the result owns an independent copy.
func (g *GpuMat) Reshape(cn, rows int) *GpuMat {
	requireNonEmpty(g, "Reshape")
	total := len(g.mat.Data)
	newCh := cn
	if newCh == 0 {
		newCh = g.mat.Channels
	}
	if newCh <= 0 {
		panic(fmt.Sprintf("cudacore: Reshape invalid channel count %d", cn))
	}
	newRows := rows
	if newRows == 0 {
		newRows = g.mat.Rows
	}
	if total%newCh != 0 {
		panic(fmt.Sprintf("cudacore: Reshape %d samples not divisible by %d channels", total, newCh))
	}
	pixels := total / newCh
	if newRows <= 0 || pixels%newRows != 0 {
		panic(fmt.Sprintf("cudacore: Reshape cannot form %d rows from %d pixels", newRows, pixels))
	}
	newCols := pixels / newRows
	data := make([]uint8, total)
	copy(data, g.mat.Data)
	return &GpuMat{mat: &cv.Mat{Rows: newRows, Cols: newCols, Channels: newCh, Data: data}}
}

// Row returns a new single-row GpuMat holding a deep copy of row y, the analogue
// of cv::cuda::GpuMat::row. It panics if y is out of range.
func (g *GpuMat) Row(y int) *GpuMat {
	return g.RowRange(y, y+1)
}

// Col returns a new single-column GpuMat holding a deep copy of column x, the
// analogue of cv::cuda::GpuMat::col. It panics if x is out of range.
func (g *GpuMat) Col(x int) *GpuMat {
	return g.ColRange(x, x+1)
}

// RowRange returns a new GpuMat holding a deep copy of rows [start, end), the
// analogue of cv::cuda::GpuMat::rowRange. Unlike OpenCV's header-sharing range,
// the result owns independent storage. It panics if the range is empty or out
// of bounds.
func (g *GpuMat) RowRange(start, end int) *GpuMat {
	requireNonEmpty(g, "RowRange")
	m := g.mat
	if start < 0 || end > m.Rows || start >= end {
		panic(fmt.Sprintf("cudacore: RowRange [%d,%d) out of bounds for %d rows", start, end, m.Rows))
	}
	stride := m.Cols * m.Channels
	rows := end - start
	data := make([]uint8, rows*stride)
	copy(data, m.Data[start*stride:end*stride])
	return &GpuMat{mat: &cv.Mat{Rows: rows, Cols: m.Cols, Channels: m.Channels, Data: data}}
}

// ColRange returns a new GpuMat holding a deep copy of columns [start, end), the
// analogue of cv::cuda::GpuMat::colRange. Unlike OpenCV's header-sharing range,
// the result owns independent storage. It panics if the range is empty or out
// of bounds.
func (g *GpuMat) ColRange(start, end int) *GpuMat {
	requireNonEmpty(g, "ColRange")
	m := g.mat
	if start < 0 || end > m.Cols || start >= end {
		panic(fmt.Sprintf("cudacore: ColRange [%d,%d) out of bounds for %d cols", start, end, m.Cols))
	}
	cols := end - start
	ch := m.Channels
	out := cv.NewMat(m.Rows, cols, ch)
	srcStride := m.Cols * ch
	dstStride := cols * ch
	for y := 0; y < m.Rows; y++ {
		src := m.Data[y*srcStride+start*ch:]
		copy(out.Data[y*dstStride:y*dstStride+dstStride], src[:dstStride])
	}
	return &GpuMat{mat: out}
}

// Size returns the matrix dimensions as (rows, cols), or (0, 0) when empty.
func (g *GpuMat) Size() (rows, cols int) {
	if g.Empty() {
		return 0, 0
	}
	return g.mat.Rows, g.mat.Cols
}

// Type returns the [MatType] of the matrix. Every GpuMat has CV_8U depth, so the
// type reflects only the channel count. An empty GpuMat reports CV_8UC1.
func (g *GpuMat) Type() MatType {
	if g.Empty() {
		return CV_8UC1
	}
	return MakeType(g.mat.Channels)
}

// Channels returns the number of channels, or 0 when empty.
func (g *GpuMat) Channels() int {
	if g.Empty() {
		return 0
	}
	return g.mat.Channels
}

// Empty reports whether the GpuMat holds no samples.
func (g *GpuMat) Empty() bool {
	return g == nil || g.mat == nil || g.mat.Empty()
}

// IsContinuous reports whether the samples occupy a single contiguous block with
// no inter-row padding, the analogue of cv::cuda::GpuMat::isContinuous. Because
// this CPU-backed container never pads rows, a non-empty GpuMat is always
// continuous; an empty one reports false.
func (g *GpuMat) IsContinuous() bool {
	return !g.Empty()
}

// SwapChannels reorders the channels in place according to order, the analogue
// of cv::cuda::swapChannels. order must contain exactly Channels entries, each a
// valid source-channel index in [0, Channels); the new channel c takes its
// samples from old channel order[c]. It panics on an empty GpuMat or an invalid
// permutation.
func (g *GpuMat) SwapChannels(order []int) {
	requireNonEmpty(g, "SwapChannels")
	ch := g.mat.Channels
	if len(order) != ch {
		panic(fmt.Sprintf("cudacore: SwapChannels needs %d indices, got %d", ch, len(order)))
	}
	for _, s := range order {
		if s < 0 || s >= ch {
			panic(fmt.Sprintf("cudacore: SwapChannels index %d out of range [0,%d)", s, ch))
		}
	}
	data := g.mat.Data
	px := make([]uint8, ch)
	for i := 0; i < len(data); i += ch {
		copy(px, data[i:i+ch])
		for c := 0; c < ch; c++ {
			data[i+c] = px[order[c]]
		}
	}
}

// PtrStepSz returns a lightweight, read-only view of the matrix's raw storage,
// the analogue of the cv::cuda::PtrStepSz descriptor OpenCV kernels receive.
// The returned view shares the receiver's backing slice: it is a pointer-like
// handle, not a copy, so callers must treat Data as read-only. It panics on an
// empty GpuMat.
func (g *GpuMat) PtrStepSz() PtrStepSz {
	requireNonEmpty(g, "PtrStepSz")
	m := g.mat
	return PtrStepSz{
		Data:     m.Data,
		Rows:     m.Rows,
		Cols:     m.Cols,
		Channels: m.Channels,
		Step:     m.Cols * m.Channels,
	}
}

// requireNonEmpty panics if g is empty.
func requireNonEmpty(g *GpuMat, name string) {
	if g.Empty() {
		panic(fmt.Sprintf("cudacore: %s given an empty GpuMat", name))
	}
}

// clampToUint8 clamps v into [0,255] (the caller supplies any rounding bias),
// mirroring the root package's helper so results agree at the boundary.
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
