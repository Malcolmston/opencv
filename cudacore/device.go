package cudacore

import (
	"fmt"
	"os"
	"runtime"
	"sync"
)

// currentDevice tracks the "selected" pseudo-device index. It exists only so
// that [GetDevice] and [SetDevice] round-trip; no real device is ever selected.
var (
	deviceMu      sync.Mutex
	currentDevice int
)

// GetCudaEnabledDeviceCount reports the number of CUDA-capable devices, the
// analogue of cv::cuda::getCudaEnabledDeviceCount. This build has no CUDA
// support, so it always returns 0 — exactly what stock OpenCV compiled without
// CUDA returns. The CPU pseudo-device exposed by [DeviceInfo] is not counted
// here because it is not a real CUDA device.
func GetCudaEnabledDeviceCount() int {
	return 0
}

// GetDevice returns the index of the currently selected pseudo-device (0 by
// default), the analogue of cv::cuda::getDevice.
func GetDevice() int {
	deviceMu.Lock()
	defer deviceMu.Unlock()
	return currentDevice
}

// SetDevice records device as the selected pseudo-device index, the analogue of
// cv::cuda::setDevice. Because there is no real device it performs no hardware
// switch; it only stores the value so [GetDevice] observes it. It panics if
// device is negative.
func SetDevice(device int) {
	if device < 0 {
		panic(fmt.Sprintf("cudacore: SetDevice requires a non-negative index, got %d", device))
	}
	deviceMu.Lock()
	currentDevice = device
	deviceMu.Unlock()
}

// ResetDevice mirrors cv::cuda::resetDevice, which frees all device resources of
// the current context. There are none here, so it merely resets the selected
// pseudo-device index back to 0.
func ResetDevice() {
	deviceMu.Lock()
	currentDevice = 0
	deviceMu.Unlock()
}

// FeatureSet names a CUDA compute-capability feature set, mirroring
// cv::cuda::FeatureSet. The values match OpenCV's encoding (major*10 + minor for
// compute levels; the atomic/native-double flags use OpenCV's shifted values).
type FeatureSet int

const (
	// FeatureSetCompute10 is compute capability 1.0.
	FeatureSetCompute10 FeatureSet = 10
	// FeatureSetCompute11 is compute capability 1.1.
	FeatureSetCompute11 FeatureSet = 11
	// FeatureSetCompute12 is compute capability 1.2.
	FeatureSetCompute12 FeatureSet = 12
	// FeatureSetCompute13 is compute capability 1.3.
	FeatureSetCompute13 FeatureSet = 13
	// FeatureSetCompute20 is compute capability 2.0.
	FeatureSetCompute20 FeatureSet = 20
	// FeatureSetCompute21 is compute capability 2.1.
	FeatureSetCompute21 FeatureSet = 21
	// FeatureSetCompute30 is compute capability 3.0.
	FeatureSetCompute30 FeatureSet = 30
	// FeatureSetCompute35 is compute capability 3.5.
	FeatureSetCompute35 FeatureSet = 35
	// GlobalAtomics denotes support for global-memory atomic operations.
	GlobalAtomics FeatureSet = 1 << 0
	// SharedAtomics denotes support for shared-memory atomic operations.
	SharedAtomics FeatureSet = 1 << 1
	// NativeDouble denotes native double-precision support.
	NativeDouble FeatureSet = 1 << 2
)

// DeviceSupports reports whether the current device provides the given feature
// set, the analogue of cv::cuda::deviceSupports. There is no CUDA device here,
// so it always returns false regardless of the requested feature.
func DeviceSupports(_ FeatureSet) bool {
	return false
}

// DeviceInfo describes a pseudo-device, mirroring cv::cuda::DeviceInfo. Every
// field is derived from the Go runtime so diagnostics have coherent values to
// print, but this never describes a real CUDA device: [DeviceInfo.IsCompatible]
// is false and the compute-capability version is 0.0.
type DeviceInfo struct {
	id int
}

// NewDeviceInfo returns a DeviceInfo for the given pseudo-device index, the
// analogue of constructing cv::cuda::DeviceInfo(device). It panics if device is
// negative.
func NewDeviceInfo(device int) *DeviceInfo {
	if device < 0 {
		panic(fmt.Sprintf("cudacore: NewDeviceInfo requires a non-negative index, got %d", device))
	}
	return &DeviceInfo{id: device}
}

// DeviceID returns the pseudo-device index this DeviceInfo describes.
func (d *DeviceInfo) DeviceID() int {
	return d.id
}

// Name returns a human-readable name for the pseudo-device. It is derived from
// the Go runtime (architecture and CPU count) and explicitly labels the device
// as a CPU pseudo-device so no caller mistakes it for CUDA hardware.
func (d *DeviceInfo) Name() string {
	return fmt.Sprintf("CPU pseudo-device (%s, %d cores)", runtime.GOARCH, runtime.NumCPU())
}

// TotalMemory returns the total device memory in bytes, the analogue of
// cv::cuda::DeviceInfo::totalMemory. There is no dedicated device memory, so it
// reports 0.
func (d *DeviceInfo) TotalMemory() uint64 {
	return 0
}

// FreeMemory returns the free device memory in bytes, the analogue of
// cv::cuda::DeviceInfo::freeMemory. There is no dedicated device memory, so it
// reports 0.
func (d *DeviceInfo) FreeMemory() uint64 {
	return 0
}

// MajorVersion returns the CUDA compute-capability major version. There is no
// CUDA device, so it is 0.
func (d *DeviceInfo) MajorVersion() int {
	return 0
}

// MinorVersion returns the CUDA compute-capability minor version. There is no
// CUDA device, so it is 0.
func (d *DeviceInfo) MinorVersion() int {
	return 0
}

// MultiProcessorCount returns the number of streaming multiprocessors, the
// analogue of cv::cuda::DeviceInfo::multiProcessorCount. It reports the host CPU
// count as the pseudo-device's parallelism.
func (d *DeviceInfo) MultiProcessorCount() int {
	return runtime.NumCPU()
}

// IsCompatible reports whether the device's binaries are compatible with the
// current OpenCV build, the analogue of cv::cuda::DeviceInfo::isCompatible.
// There is no CUDA device, so it is always false.
func (d *DeviceInfo) IsCompatible() bool {
	return false
}

// Query refreshes the DeviceInfo from the (pseudo) device, the analogue of
// cv::cuda::DeviceInfo::queryMemory-style refresh. The fields are derived on
// demand, so this is a no-op that returns the receiver for chaining.
func (d *DeviceInfo) Query() *DeviceInfo {
	return d
}

// PrintCudaDeviceInfo writes a multi-line description of the pseudo-device to
// standard output, the analogue of cv::cuda::printCudaDeviceInfo. It makes the
// absence of real CUDA hardware explicit.
func PrintCudaDeviceInfo(device int) {
	info := NewDeviceInfo(device)
	fmt.Fprintf(os.Stdout, "*** CUDA Device Query (cudacore, CPU-backed) ***\n")
	fmt.Fprintf(os.Stdout, "Device count: %d (no CUDA-capable device)\n", GetCudaEnabledDeviceCount())
	fmt.Fprintf(os.Stdout, "Device %d: \"%s\"\n", info.DeviceID(), info.Name())
	fmt.Fprintf(os.Stdout, "  CUDA Capability Major/Minor version number:    %d.%d\n", info.MajorVersion(), info.MinorVersion())
	fmt.Fprintf(os.Stdout, "  Total amount of global memory:                 %d bytes\n", info.TotalMemory())
	fmt.Fprintf(os.Stdout, "  Multiprocessors (host CPU cores):              %d\n", info.MultiProcessorCount())
	fmt.Fprintf(os.Stdout, "  Device is compatible with this build:          %t\n", info.IsCompatible())
}

// PrintShortCudaDeviceInfo writes a one-line description of the pseudo-device to
// standard output, the analogue of cv::cuda::printShortCudaDeviceInfo.
func PrintShortCudaDeviceInfo(device int) {
	info := NewDeviceInfo(device)
	fmt.Fprintf(os.Stdout, "Device %d: \"%s\", compute %d.%d, %d cores (CPU-backed, not CUDA)\n",
		info.DeviceID(), info.Name(), info.MajorVersion(), info.MinorVersion(), info.MultiProcessorCount())
}
