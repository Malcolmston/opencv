package cudacore_test

import (
	"runtime"
	"testing"

	"github.com/malcolmston/opencv/cudacore"
)

func TestDeviceCountIsZero(t *testing.T) {
	if got := cudacore.GetCudaEnabledDeviceCount(); got != 0 {
		t.Fatalf("GetCudaEnabledDeviceCount = %d, want 0 (no CUDA)", got)
	}
}

func TestGetSetResetDevice(t *testing.T) {
	cudacore.ResetDevice()
	if cudacore.GetDevice() != 0 {
		t.Fatalf("device after reset = %d, want 0", cudacore.GetDevice())
	}
	cudacore.SetDevice(2)
	if cudacore.GetDevice() != 2 {
		t.Fatalf("device after SetDevice(2) = %d, want 2", cudacore.GetDevice())
	}
	cudacore.ResetDevice()
	if cudacore.GetDevice() != 0 {
		t.Fatalf("device after second reset = %d, want 0", cudacore.GetDevice())
	}
	assertPanics(t, "SetDevice(-1)", func() { cudacore.SetDevice(-1) })
}

func TestDeviceSupportsAlwaysFalse(t *testing.T) {
	for _, fs := range []cudacore.FeatureSet{
		cudacore.FeatureSetCompute10,
		cudacore.FeatureSetCompute35,
		cudacore.GlobalAtomics,
		cudacore.NativeDouble,
	} {
		if cudacore.DeviceSupports(fs) {
			t.Fatalf("DeviceSupports(%d) = true, want false", fs)
		}
	}
}

func TestDeviceInfo(t *testing.T) {
	d := cudacore.NewDeviceInfo(0)
	if d.DeviceID() != 0 {
		t.Fatalf("DeviceID = %d, want 0", d.DeviceID())
	}
	if d.Name() == "" {
		t.Fatalf("Name should be non-empty")
	}
	if d.TotalMemory() != 0 || d.FreeMemory() != 0 {
		t.Fatalf("pseudo-device should report 0 memory")
	}
	if d.MajorVersion() != 0 || d.MinorVersion() != 0 {
		t.Fatalf("pseudo-device should report compute 0.0")
	}
	if d.MultiProcessorCount() != runtime.NumCPU() {
		t.Fatalf("MultiProcessorCount = %d, want %d", d.MultiProcessorCount(), runtime.NumCPU())
	}
	if d.IsCompatible() {
		t.Fatalf("pseudo-device must not be CUDA-compatible")
	}
	if d.Query() != d {
		t.Fatalf("Query should return the receiver")
	}
	assertPanics(t, "NewDeviceInfo(-1)", func() { cudacore.NewDeviceInfo(-1) })
}

func TestPrintDeviceInfoDoesNotPanic(t *testing.T) {
	cudacore.PrintCudaDeviceInfo(0)
	cudacore.PrintShortCudaDeviceInfo(0)
}
