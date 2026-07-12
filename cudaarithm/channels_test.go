package cudaarithm

import (
	"testing"

	cv "github.com/malcolmston/opencv"
)

func TestSplitMergeRoundTrip(t *testing.T) {
	src := cv.NewMat(2, 2, 3)
	for i := range src.Data {
		src.Data[i] = uint8(i * 3)
	}
	g := NewGpuMat(src)
	planes := Split(g)
	if len(planes) != 3 {
		t.Fatalf("Split returned %d planes, want 3", len(planes))
	}
	merged := Merge(planes).Download()
	if !sameData(merged, src) {
		t.Error("Merge(Split(x)) should reproduce x")
	}
}

func TestTransposeAndFlip(t *testing.T) {
	src := ramp(2, 3, 0, 1)
	g := NewGpuMat(src)
	if !sameData(Transpose(g).Download(), cv.Transpose(src)) {
		t.Error("Transpose does not match cv.Transpose")
	}
	if !sameData(Flip(g, cv.FlipHorizontal).Download(), cv.Flip(src, cv.FlipHorizontal)) {
		t.Error("Flip does not match cv.Flip")
	}
}

func TestLUT(t *testing.T) {
	src := ramp(1, 4, 0, 1) // 0,1,2,3
	var table [256]uint8
	for i := range table {
		table[i] = uint8(255 - i)
	}
	got := LUT(NewGpuMat(src), table[:]).Download()
	for i, s := range src.Data {
		if got.Data[i] != table[s] {
			t.Errorf("LUT[%d] = %d, want %d", i, got.Data[i], table[s])
		}
	}
}

func TestLUTWrongSizePanics(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("expected panic on wrong table size")
		}
	}()
	LUT(NewGpuMat(constMat(1, 1, 0)), make([]uint8, 10))
}

func TestNormAndNormalize(t *testing.T) {
	src := cv.NewMat(1, 3, 1)
	copy(src.Data, []uint8{1, 2, 2}) // L1=5, L2=3, Inf=2
	g := NewGpuMat(src)

	if got := Norm(g, NormL1); got != 5 {
		t.Errorf("NormL1 = %v, want 5", got)
	}
	if got := Norm(g, NormL2); got != 3 {
		t.Errorf("NormL2 = %v, want 3", got)
	}
	if got := Norm(g, NormInf); got != 2 {
		t.Errorf("NormInf = %v, want 2", got)
	}

	// Normalize to L1 = 100: scale = 100/5 = 20 -> {20,40,40}.
	nrm := Normalize(g, 100, 0, NormL1).Download()
	for i, w := range []uint8{20, 40, 40} {
		if nrm.Data[i] != w {
			t.Errorf("Normalize L1 [%d] = %d, want %d", i, nrm.Data[i], w)
		}
	}

	// MinMax normalize delegates to cv.Normalize.
	mm := Normalize(g, 0, 255, NormMinMax).Download()
	if !sameData(mm, cv.Normalize(src, 0, 255)) {
		t.Error("NormMinMax should match cv.Normalize")
	}
}

func TestNormMinMaxPanics(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("expected panic for Norm with NormMinMax")
		}
	}()
	Norm(NewGpuMat(constMat(1, 1, 1)), NormMinMax)
}
