package cv

import "testing"

func TestThresholdBinary(t *testing.T) {
	m := grayFromValues(1, 4, []uint8{10, 100, 150, 200})
	out, used := Threshold(m, 120, 255, ThreshBinary)
	if used != 120 {
		t.Errorf("threshold used = %v, want 120", used)
	}
	want := []uint8{0, 0, 255, 255}
	for i, w := range want {
		if out.Data[i] != w {
			t.Errorf("binary[%d] = %d, want %d", i, out.Data[i], w)
		}
	}
}

func TestThresholdVariants(t *testing.T) {
	m := grayFromValues(1, 3, []uint8{50, 120, 200})
	inv, _ := Threshold(m, 100, 255, ThreshBinaryInv)
	if inv.Data[0] != 255 || inv.Data[2] != 0 {
		t.Errorf("binary-inv = %v", inv.Data)
	}
	trunc, _ := Threshold(m, 100, 255, ThreshTrunc)
	if trunc.Data[0] != 50 || trunc.Data[2] != 100 {
		t.Errorf("trunc = %v", trunc.Data)
	}
	tozero, _ := Threshold(m, 100, 255, ThreshToZero)
	if tozero.Data[0] != 0 || tozero.Data[2] != 200 {
		t.Errorf("tozero = %v", tozero.Data)
	}
}

func TestOtsuBimodal(t *testing.T) {
	// 100 pixels: half at 50, half at 200.
	m := NewMat(10, 10, 1)
	for i := range m.Data {
		if i < 50 {
			m.Data[i] = 50
		} else {
			m.Data[i] = 200
		}
	}
	out, used := Threshold(m, 0, 255, ThreshBinary|ThreshOtsu)
	if used < 50 || used > 199 {
		t.Errorf("otsu level = %v, want in [50,199]", used)
	}
	// The two clusters must be cleanly separated.
	for i := range m.Data {
		if m.Data[i] == 50 && out.Data[i] != 0 {
			t.Fatalf("otsu misclassified a 50 pixel")
		}
		if m.Data[i] == 200 && out.Data[i] != 255 {
			t.Fatalf("otsu misclassified a 200 pixel")
		}
	}
}

func TestAdaptiveThresholdMean(t *testing.T) {
	m := NewMat(5, 5, 1)
	m.SetTo(100)
	m.Set(2, 2, 0, 200) // a bright spot above local mean
	out := AdaptiveThreshold(m, 255, AdaptiveThreshMeanC, ThreshBinary, 3, 5)
	if out.At(2, 2, 0) != 255 {
		t.Errorf("adaptive: bright spot should be foreground, got %d", out.At(2, 2, 0))
	}
}

func TestInRangeMask(t *testing.T) {
	m := grayFromValues(1, 4, []uint8{10, 50, 90, 130})
	mask := InRange(m, []uint8{40}, []uint8{100})
	want := []uint8{0, 255, 255, 0}
	for i, w := range want {
		if mask.Data[i] != w {
			t.Errorf("inrange[%d] = %d, want %d", i, mask.Data[i], w)
		}
	}
}
