# cudacore — CPU-backed, API-compatible mirror of the core cuda submodule of OpenCV (the cv::cuda

[![Go Reference](https://pkg.go.dev/badge/github.com/malcolmston/opencv/cudacore.svg)](https://pkg.go.dev/github.com/malcolmston/opencv/cudacore)

Package cudacore is a CPU-backed, API-compatible mirror of the core cuda
submodule of OpenCV (the cv::cuda names declared in core/cuda.hpp): device
management and the `GpuMat` device-matrix container. It is implemented
entirely in pure Go on top of the root module github.com/malcolmston/opencv
(imported as cv) and the Go standard library. There is no cgo, no CUDA runtime
and no GPU.

- Device queries report a single CPU pseudo-device.
`GetCudaEnabledDeviceCount` returns 0 — exactly what stock OpenCV built
without CUDA returns — because no CUDA-capable device exists. The
`DeviceInfo` type still describes a pseudo-device (its name is derived from
the Go runtime, its "multiprocessor" count from the CPU count) so that
diagnostic code has something coherent to print, but `DeviceInfo.IsCompatible`
is false and `DeviceSupports` always reports false: nothing here is a real
CUDA device. - `GpuMat` is a thin wrapper around a *cv.Mat. "Uploading" and
"downloading" (`GpuMat.Upload`, `GpuMat.Download`, `NewGpuMat`) copy the
matrix rather than moving it across a PCIe bus, because host and device memory
are the same memory here. The copy semantics are preserved (the
stored/returned Mat does not alias the caller's), which keeps behaviour
predictable, but there is no acceleration. - `Stream` and `Event` are no-ops.
Every container operation runs synchronously before it returns, so
`Stream.WaitForCompletion`, `Event.Record`, `Event.WaitForCompletion`,
`StreamWaitEvent` and `ElapsedTime` have no asynchronous work to coordinate.
ElapsedTime always reports zero. - `GpuMatND`, `BufferPool` and
`SetBufferPoolUsage` are documented no-op stand-ins that carry no device
state.

## Install

```sh
go get github.com/malcolmston/opencv@v0.8.0
```

```go
import "github.com/malcolmston/opencv/cudacore"
```

## Usage

This is the package's own `Example`, so it compiles and its output is
asserted on every `go test ./cudacore/`.

```go
fmt.Println("cuda devices:", cudacore.GetCudaEnabledDeviceCount())

	src := cv.NewMat(1, 3, 1)
	copy(src.Data, []uint8{10, 20, 30})

	var g cudacore.GpuMat
	g.Upload(src)
	out := g.Download()

	fmt.Println("round-trip:", ints(out.Data))
```

```
cuda devices: 0
round-trip: [10 20 30]
```

## Exported surface

### Functions

| Function | What it does |
| --- | --- |
| `func DeviceSupports(_ FeatureSet) bool` | DeviceSupports reports whether the current device provides the given feature set, the analogue of cv::cuda::deviceSupports. |
| `func ElapsedTime(_, _ *Event) float64` | ElapsedTime returns the elapsed time in milliseconds between two recorded Events, the analogue of cv::cuda::Event::elapsedTime. |
| `func EnsureSizeIsEnough(rows, cols int, typ MatType, m *GpuMat)` | EnsureSizeIsEnough guarantees that m has at least the requested rows, cols and type, reallocating it only when its current layout does not already… |
| `func GetCudaEnabledDeviceCount() int` | GetCudaEnabledDeviceCount reports the number of CUDA-capable devices, the analogue of cv::cuda::getCudaEnabledDeviceCount. |
| `func GetDevice() int` | GetDevice returns the index of the currently selected pseudo-device (0 by default), the analogue of cv::cuda::getDevice. |
| `func PrintCudaDeviceInfo(device int)` | PrintCudaDeviceInfo writes a multi-line description of the pseudo-device to standard output, the analogue of cv::cuda::printCudaDeviceInfo. |
| `func PrintShortCudaDeviceInfo(device int)` | PrintShortCudaDeviceInfo writes a one-line description of the pseudo-device to standard output, the analogue of cv::cuda::printShortCudaDeviceInfo. |
| `func ResetDevice()` | ResetDevice mirrors cv::cuda::resetDevice, which frees all device resources of the current context. |
| `func SetBufferPoolUsage(_ bool)` | SetBufferPoolUsage mirrors cv::cuda::setBufferPoolUsage. |
| `func SetDevice(device int)` | SetDevice records device as the selected pseudo-device index, the analogue of cv::cuda::setDevice. |
| `func StreamWaitEvent(_ *Stream, _ *Event)` | StreamWaitEvent makes stream wait for event before proceeding, the analogue of cv::cuda::Stream::waitEvent. |

### Types

| Type | What it is |
| --- | --- |
| `BorderType` | BorderType selects how `GpuMat.CopyMakeBorder` fabricates samples outside the source rectangle. |
| `BufferPool` | BufferPool is a CPU-backed stand-in for cv::cuda::BufferPool, OpenCV's per-stream scratch allocator. |
| `DeviceInfo` | DeviceInfo describes a pseudo-device, mirroring cv::cuda::DeviceInfo. |
| `Event` | Event is a CPU-backed stand-in for cv::cuda::Event. |
| `FeatureSet` | FeatureSet names a CUDA compute-capability feature set, mirroring cv::cuda::FeatureSet. |
| `GpuMat` | GpuMat is a CPU-backed stand-in for OpenCV's cv::cuda::GpuMat. |
| `GpuMatND` | GpuMatND is a CPU-backed stand-in for cv::cuda::GpuMatND, an N-dimensional device array. |
| `HostMem` | HostMem is a CPU-backed stand-in for cv::cuda::HostMem, OpenCV's page-locked (pinned) host buffer used to accelerate host/device transfers. |
| `MatType` | MatType encodes an OpenCV matrix type. |
| `PtrStepSz` | PtrStepSz is a lightweight, read-only descriptor of a `GpuMat`'s raw storage, the analogue of cv::cuda::PtrStepSz. |
| `Stream` | Stream is a CPU-backed stand-in for cv::cuda::Stream. |

<details>
<summary><code>BufferPool</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewBufferPool(_ *Stream) *BufferPool` | NewBufferPool returns a BufferPool associated with the given stream. |
| `func (p *BufferPool) GetBuffer(rows, cols int, typ MatType) *GpuMat` | GetBuffer returns a freshly allocated, zero-filled `GpuMat` of the requested size and type, the analogue of cv::cuda::BufferPool::getBuffer. |

</details>

<details>
<summary><code>DeviceInfo</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewDeviceInfo(device int) *DeviceInfo` | NewDeviceInfo returns a DeviceInfo for the given pseudo-device index, the analogue of constructing cv::cuda::DeviceInfo(device). |
| `func (d *DeviceInfo) DeviceID() int` | DeviceID returns the pseudo-device index this DeviceInfo describes. |
| `func (d *DeviceInfo) FreeMemory() uint64` | FreeMemory returns the free device memory in bytes, the analogue of cv::cuda::DeviceInfo::freeMemory. |
| `func (d *DeviceInfo) IsCompatible() bool` | IsCompatible reports whether the device's binaries are compatible with the current OpenCV build, the analogue of cv::cuda::DeviceInfo::isCompatible. |
| `func (d *DeviceInfo) MajorVersion() int` | MajorVersion returns the CUDA compute-capability major version. |
| `func (d *DeviceInfo) MinorVersion() int` | MinorVersion returns the CUDA compute-capability minor version. |
| `func (d *DeviceInfo) MultiProcessorCount() int` | MultiProcessorCount returns the number of streaming multiprocessors, the analogue of cv::cuda::DeviceInfo::multiProcessorCount. |
| `func (d *DeviceInfo) Name() string` | Name returns a human-readable name for the pseudo-device. |
| `func (d *DeviceInfo) Query() *DeviceInfo` | Query refreshes the DeviceInfo from the (pseudo) device, the analogue of cv::cuda::DeviceInfo::queryMemory-style refresh. |
| `func (d *DeviceInfo) TotalMemory() uint64` | TotalMemory returns the total device memory in bytes, the analogue of cv::cuda::DeviceInfo::totalMemory. |

</details>

<details>
<summary><code>Event</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewEvent() *Event` | NewEvent returns a ready-to-use no-op Event. |
| `func (e *Event) QueryIfComplete() bool` | QueryIfComplete reports whether the Event has been reached, the analogue of cv::cuda::Event::queryIfComplete. |
| `func (e *Event) Record(_ *Stream)` | Record marks the Event in the given stream, the analogue of cv::cuda::Event::record. |
| `func (e *Event) WaitForCompletion()` | WaitForCompletion returns immediately, the analogue of cv::cuda::Event::waitForCompletion. |

</details>

<details>
<summary><code>GpuMat</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewGpuMat(src *cv.Mat) *GpuMat` | NewGpuMat returns a GpuMat holding a copy of src, mimicking an upload of a host matrix to the device. |
| `func (g *GpuMat) Channels() int` | Channels returns the number of channels, or 0 when empty. |
| `func (g *GpuMat) Clone() *GpuMat` | Clone returns a deep copy of the GpuMat with its own backing storage, mirroring cv::cuda::GpuMat::clone. |
| `func (g *GpuMat) Col(x int) *GpuMat` | Col returns a new single-column GpuMat holding a deep copy of column x, the analogue of cv::cuda::GpuMat::col. |
| `func (g *GpuMat) ColRange(start, end int) *GpuMat` | ColRange returns a new GpuMat holding a deep copy of columns [start, end), the analogue of cv::cuda::GpuMat::colRange. |
| `func (g *GpuMat) ConvertTo(alpha, beta float64) *GpuMat` | ConvertTo returns a new GpuMat whose every sample is saturate(round(src*alpha + beta)), the analogue of cv::cuda::GpuMat::convertTo restricted to the… |
| `func (g *GpuMat) CopyMakeBorder(top, bottom, left, right int, borderType BorderType, value cv.Scalar) *GpuMat` | CopyMakeBorder returns a new GpuMat that is the receiver surrounded by top, bottom, left and right extra rows/columns, the analogue of… |
| `func (g *GpuMat) CopyTo(dst *GpuMat)` | CopyTo copies the entire matrix into dst, reallocating dst to match, the analogue of cv::cuda::GpuMat::copyTo. |
| `func (g *GpuMat) Create(rows, cols int, typ MatType)` | Create allocates fresh zero-filled storage of the given size and type, replacing any previous contents. |
| `func (g *GpuMat) Download() *cv.Mat` | Download returns a copy of the GpuMat's contents as a host `cv.Mat`, the analogue of cv::cuda::GpuMat::download. |
| `func (g *GpuMat) Empty() bool` | Empty reports whether the GpuMat holds no samples. |
| `func (g *GpuMat) IsContinuous() bool` | IsContinuous reports whether the samples occupy a single contiguous block with no inter-row padding, the analogue of cv::cuda::GpuMat::isContinuous. |
| `func (g *GpuMat) Mat() *cv.Mat` | Mat returns the wrapped `cv.Mat` without copying, for read-only interop with the root package. |
| `func (g *GpuMat) PtrStepSz() PtrStepSz` | PtrStepSz returns a lightweight, read-only view of the matrix's raw storage, the analogue of the cv::cuda::PtrStepSz descriptor OpenCV kernels… |
| `func (g *GpuMat) Release()` | Release frees the GpuMat's contents, leaving it empty. |
| `func (g *GpuMat) Reshape(cn, rows int) *GpuMat` | Reshape returns a new GpuMat that reinterprets the same samples with a different channel count and/or row count, the analogue of… |
| `func (g *GpuMat) Row(y int) *GpuMat` | Row returns a new single-row GpuMat holding a deep copy of row y, the analogue of cv::cuda::GpuMat::row. |
| `func (g *GpuMat) RowRange(start, end int) *GpuMat` | RowRange returns a new GpuMat holding a deep copy of rows [start, end), the analogue of cv::cuda::GpuMat::rowRange. |
| `func (g *GpuMat) SetTo(value cv.Scalar) *GpuMat` | SetTo fills every pixel with the scalar value, one component per channel, rounding and saturating each component into [0,255]. |
| `func (g *GpuMat) Size() (rows, cols int)` | Size returns the matrix dimensions as (rows, cols), or (0, 0) when empty. |
| `func (g *GpuMat) SwapChannels(order []int)` | SwapChannels reorders the channels in place according to order, the analogue of cv::cuda::swapChannels. |
| `func (g *GpuMat) Type() MatType` | Type returns the `MatType` of the matrix. |
| `func (g *GpuMat) Upload(src *cv.Mat)` | Upload copies src into the GpuMat, replacing any previous contents. |

</details>

<details>
<summary><code>GpuMatND</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewGpuMatND(sizes []int, typ MatType) *GpuMatND` | NewGpuMatND returns a GpuMatND describing an array of the given per-dimension sizes and type. |
| `func (g *GpuMatND) Channels() int` | Channels returns the channel count recorded at construction. |
| `func (g *GpuMatND) Dims() int` | Dims returns the number of dimensions. |
| `func (g *GpuMatND) Empty() bool` | Empty reports whether the array has no dimensions. |
| `func (g *GpuMatND) Release()` | Release clears the recorded shape, leaving the GpuMatND empty, the analogue of cv::cuda::GpuMatND::release. |
| `func (g *GpuMatND) Size() []int` | Size returns a copy of the per-dimension sizes. |

</details>

<details>
<summary><code>HostMem</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewHostMem(rows, cols int, typ MatType) *HostMem` | NewHostMem allocates a zero-filled HostMem of the given size and type. |
| `func (h *HostMem) CreateMatHeader() *cv.Mat` | CreateMatHeader returns the underlying `cv.Mat` without copying, the analogue of cv::cuda::HostMem::createMatHeader. |
| `func (h *HostMem) Empty() bool` | Empty reports whether the HostMem holds no samples. |
| `func (h *HostMem) Size() (rows, cols int)` | Size returns the dimensions as (rows, cols), or (0, 0) when empty. |

</details>

<details>
<summary><code>MatType</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func MakeType(channels int) MatType` | MakeType returns the CV_8U MatType for the given channel count, mirroring OpenCV's CV_8UC(n). |
| `func (t MatType) Channels() int` | Channels returns the number of channels encoded by the type. |
| `func (t MatType) String() string` | String renders the type as OpenCV's CV_8UC<n> spelling. |

</details>

<details>
<summary><code>Stream</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewStream() *Stream` | NewStream returns a ready-to-use no-op Stream. |
| `func (s *Stream) QueryIfComplete() bool` | QueryIfComplete reports whether all work on the stream has finished, the analogue of cv::cuda::Stream::queryIfComplete. |
| `func (s *Stream) WaitForCompletion()` | WaitForCompletion returns immediately, the analogue of cv::cuda::Stream::waitForCompletion. |

</details>

Full signatures, doc comments and every runnable example are on
[pkg.go.dev](https://pkg.go.dev/github.com/malcolmston/opencv/cudacore).

## License

MIT, as part of [`github.com/malcolmston/opencv`](..).
An independent re-implementation, not affiliated with or endorsed by the
original project.
