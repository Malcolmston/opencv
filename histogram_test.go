package cv

import "testing"

func TestCalcHist(t *testing.T) {
	m := grayFromValues(1, 5, []uint8{0, 0, 128, 255, 128})
	h := CalcHist(m, 0)
	if h[0] != 2 || h[128] != 2 || h[255] != 1 {
		t.Errorf("histogram wrong: [0]=%d [128]=%d [255]=%d", h[0], h[128], h[255])
	}
	total := 0
	for _, c := range h {
		total += c
	}
	if total != 5 {
		t.Errorf("histogram total = %d, want 5", total)
	}
}

func TestEqualizeHistSpreadsRange(t *testing.T) {
	// A low-contrast image confined to [100,103].
	m := grayFromValues(1, 4, []uint8{100, 101, 102, 103})
	out := EqualizeHist(m)
	// The maximum intensity must map to 255.
	if out.Data[3] != 255 {
		t.Errorf("equalize max = %d, want 255", out.Data[3])
	}
	// The result must be non-decreasing (monotonic mapping).
	for i := 1; i < len(out.Data); i++ {
		if out.Data[i] < out.Data[i-1] {
			t.Errorf("equalize not monotonic at %d", i)
		}
	}
}

func TestEqualizeHistConstantImage(t *testing.T) {
	m := NewMat(3, 3, 1)
	m.SetTo(50)
	out := EqualizeHist(m)
	// A constant image should survive equalisation unchanged.
	for _, v := range out.Data {
		if v != 50 {
			t.Fatalf("equalize constant changed value to %d", v)
		}
	}
}
