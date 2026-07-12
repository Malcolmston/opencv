// Package cudacore is a CPU-backed, API-compatible mirror of the core cuda
// submodule of OpenCV (the cv::cuda names declared in core/cuda.hpp): device
// management and the [GpuMat] device-matrix container. It is implemented
// entirely in pure Go on top of the root module
// github.com/malcolmston/opencv (imported as cv) and the Go standard library.
// There is no cgo, no CUDA runtime and no GPU.
//
// # Honest scope: there is no GPU
//
// This package performs NO GPU computation and manages NO device memory. Every
// query and every container operation runs on the CPU against ordinary
// host-resident [cv.Mat] storage. The CUDA vocabulary is reproduced so that
// code written against OpenCV's cv::cuda core reads and ports naturally, but
// the following are deliberate, honest fictions:
//
//   - Device queries report a single CPU pseudo-device.
//     [GetCudaEnabledDeviceCount] returns 0 — exactly what stock OpenCV built
//     without CUDA returns — because no CUDA-capable device exists. The
//     [DeviceInfo] type still describes a pseudo-device (its name is derived
//     from the Go runtime, its "multiprocessor" count from the CPU count) so
//     that diagnostic code has something coherent to print, but
//     [DeviceInfo.IsCompatible] is false and [DeviceSupports] always reports
//     false: nothing here is a real CUDA device.
//   - [GpuMat] is a thin wrapper around a *cv.Mat. "Uploading" and
//     "downloading" ([GpuMat.Upload], [GpuMat.Download], [NewGpuMat]) copy the
//     matrix rather than moving it across a PCIe bus, because host and device
//     memory are the same memory here. The copy semantics are preserved (the
//     stored/returned Mat does not alias the caller's), which keeps behaviour
//     predictable, but there is no acceleration.
//   - [Stream] and [Event] are no-ops. Every container operation runs
//     synchronously before it returns, so [Stream.WaitForCompletion],
//     [Event.Record], [Event.WaitForCompletion], [StreamWaitEvent] and
//     [ElapsedTime] have no asynchronous work to coordinate. ElapsedTime always
//     reports zero.
//   - [GpuMatND], [BufferPool] and [SetBufferPoolUsage] are documented no-op
//     stand-ins that carry no device state.
//
// # The uint8 constraint
//
// The root package's [cv.Mat] stores 8-bit unsigned samples (depth CV_8U).
// Every [GpuMat] therefore has depth CV_8U and its [MatType] is fully
// determined by the channel count. Conversions that would produce fractional or
// out-of-range values — [GpuMat.ConvertTo], [GpuMat.SetTo] — round with the
// root package's +0.5 bias and saturate into [0,255].
//
// # What is faithful
//
// The container arithmetic is real: round-tripping through [GpuMat.Upload] and
// [GpuMat.Download] preserves samples exactly; [GpuMat.CopyMakeBorder],
// [GpuMat.Reshape], [GpuMat.RowRange], [GpuMat.ColRange] and
// [GpuMat.SwapChannels] compute the same results OpenCV would. Only the device
// and asynchrony are fictional.
package cudacore
