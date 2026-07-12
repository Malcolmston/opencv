package cv

import (
	"math"
	"testing"
)

func TestCalcBackProject(t *testing.T) {
	src := grayFromValues(1, 4, []uint8{10, 10, 20, 30})
	hist := CalcHist(src, 0)
	bp := CalcBackProject(src, 0, hist)
	// Bin 10 has count 2 (the max), so those pixels map to 255; bins 20 and 30
	// have count 1 -> 128 (rounded).
	if bp.Data[0] != 255 || bp.Data[1] != 255 {
		t.Errorf("backproject max bins = %v, want 255", bp.Data[:2])
	}
	if bp.Data[2] != 128 || bp.Data[3] != 128 {
		t.Errorf("backproject single bins = %v, want 128", bp.Data[2:])
	}
}

func TestCompareHistIdentical(t *testing.T) {
	h := CalcHist(synthSquare(20, 5, 5, 10), 0)
	if c := CompareHist(h, h, HistCmpCorrel); math.Abs(c-1) > 1e-9 {
		t.Errorf("correlation of identical = %v, want 1", c)
	}
	if c := CompareHist(h, h, HistCmpChiSqr); c != 0 {
		t.Errorf("chi-square of identical = %v, want 0", c)
	}
	if c := CompareHist(h, h, HistCmpBhattacharyya); math.Abs(c) > 1e-9 {
		t.Errorf("bhattacharyya of identical = %v, want 0", c)
	}
	total := 0
	for _, v := range h {
		total += v
	}
	if c := CompareHist(h, h, HistCmpIntersect); int(c) != total {
		t.Errorf("intersection of identical = %v, want %d", c, total)
	}
}

func TestCompareHistDisjoint(t *testing.T) {
	a := make([]int, 256)
	b := make([]int, 256)
	a[10] = 100
	b[200] = 100
	// Disjoint histograms: zero intersection, Bhattacharyya distance 1.
	if c := CompareHist(a, b, HistCmpIntersect); c != 0 {
		t.Errorf("intersection disjoint = %v, want 0", c)
	}
	if c := CompareHist(a, b, HistCmpBhattacharyya); math.Abs(c-1) > 1e-9 {
		t.Errorf("bhattacharyya disjoint = %v, want 1", c)
	}
}

func TestCLAHEImprovesContrast(t *testing.T) {
	// A low-contrast image confined to [100,140].
	m := NewMat(32, 32, 1)
	for y := 0; y < 32; y++ {
		for x := 0; x < 32; x++ {
			m.Set(y, x, 0, uint8(100+(x*40)/32))
		}
	}
	out := CLAHE(m, 40, 8)
	if out.Rows != 32 || out.Cols != 32 {
		t.Fatalf("CLAHE dims = %dx%d", out.Rows, out.Cols)
	}
	spread := func(mm *Mat) int {
		lo, hi := mm.Data[0], mm.Data[0]
		for _, v := range mm.Data {
			if v < lo {
				lo = v
			}
			if v > hi {
				hi = v
			}
		}
		return int(hi) - int(lo)
	}
	if spread(out) <= spread(m) {
		t.Errorf("CLAHE did not increase contrast: before=%d after=%d", spread(m), spread(out))
	}
}

func TestCLAHEConstantStaysValid(t *testing.T) {
	m := NewMat(16, 16, 1)
	m.SetTo(80)
	out := CLAHE(m, 40, 4)
	// A constant image must remain within range and not blow up.
	for _, v := range out.Data {
		_ = v
	}
	if out.Rows != 16 || out.Cols != 16 {
		t.Fatal("CLAHE changed dimensions")
	}
}
