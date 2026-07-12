package cudacore_test

import (
	"reflect"
	"testing"

	cv "github.com/malcolmston/opencv"
	"github.com/malcolmston/opencv/cudacore"
)

func TestHostMem(t *testing.T) {
	h := cudacore.NewHostMem(2, 3, cudacore.CV_8UC1)
	if h.Empty() {
		t.Fatalf("fresh HostMem should not be empty")
	}
	r, c := h.Size()
	if r != 2 || c != 3 {
		t.Fatalf("HostMem size = (%d,%d), want (2,3)", r, c)
	}
	m := h.CreateMatHeader()
	if m == nil || m.Rows != 2 || m.Cols != 3 {
		t.Fatalf("CreateMatHeader returned %v", m)
	}
	// The header shares storage: a GpuMat can upload from it.
	m.Data[0] = 42
	g := cudacore.NewGpuMat(h.CreateMatHeader())
	if g.Download().Data[0] != 42 {
		t.Fatalf("HostMem header did not share storage")
	}
	assertPanics(t, "NewHostMem bad size", func() { cudacore.NewHostMem(0, 1, cudacore.CV_8UC1) })
}

func TestGpuMatND(t *testing.T) {
	nd := cudacore.NewGpuMatND([]int{2, 3, 4}, cudacore.CV_8UC3)
	if nd.Empty() {
		t.Fatalf("GpuMatND should not be empty")
	}
	if nd.Dims() != 3 {
		t.Fatalf("Dims = %d, want 3", nd.Dims())
	}
	if !reflect.DeepEqual(nd.Size(), []int{2, 3, 4}) {
		t.Fatalf("Size = %v", nd.Size())
	}
	if nd.Channels() != 3 {
		t.Fatalf("Channels = %d, want 3", nd.Channels())
	}
	// Size returns a copy; mutating it must not affect the array.
	nd.Size()[0] = 99
	if nd.Size()[0] != 2 {
		t.Fatalf("Size should return a defensive copy")
	}
	nd.Release()
	if !nd.Empty() {
		t.Fatalf("Release should empty the GpuMatND")
	}
	assertPanics(t, "NewGpuMatND bad size", func() { cudacore.NewGpuMatND([]int{2, 0}, cudacore.CV_8UC1) })
}

func TestBufferPool(t *testing.T) {
	cudacore.SetBufferPoolUsage(true)
	defer cudacore.SetBufferPoolUsage(false)

	pool := cudacore.NewBufferPool(cudacore.NewStream())
	buf := pool.GetBuffer(2, 2, cudacore.CV_8UC1)
	r, c := buf.Size()
	if r != 2 || c != 2 {
		t.Fatalf("GetBuffer size = (%d,%d), want (2,2)", r, c)
	}
	// Fresh buffers are zero-filled and independent.
	buf.SetTo(cv.NewScalar(1))
	other := pool.GetBuffer(2, 2, cudacore.CV_8UC1)
	if other.Download().Data[0] != 0 {
		t.Fatalf("GetBuffer should return a fresh zero buffer each call")
	}
	assertPanics(t, "GetBuffer bad size", func() { pool.GetBuffer(0, 2, cudacore.CV_8UC1) })
}

func TestEnsureSizeIsEnoughNilPanics(t *testing.T) {
	assertPanics(t, "nil GpuMat", func() {
		cudacore.EnsureSizeIsEnough(2, 2, cudacore.CV_8UC1, nil)
	})
	assertPanics(t, "bad size", func() {
		var g cudacore.GpuMat
		cudacore.EnsureSizeIsEnough(0, 2, cudacore.CV_8UC1, &g)
	})
}
