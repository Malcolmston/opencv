package cv

import (
	"math"
	"testing"
)

func TestMeanAndSum(t *testing.T) {
	m := NewMat(2, 2, 1)
	vals := []uint8{10, 20, 30, 40}
	copy(m.Data, vals)
	mean := Mean(m)
	if mean[0] != 25 {
		t.Errorf("Mean = %v, want 25", mean[0])
	}
	sum := SumElems(m)
	if sum[0] != 100 {
		t.Errorf("Sum = %v, want 100", sum[0])
	}
}

func TestMeanStdDev(t *testing.T) {
	m := NewMat(1, 4, 1)
	copy(m.Data, []uint8{2, 4, 4, 6})
	mean, sd := MeanStdDev(m)
	if mean[0] != 4 {
		t.Errorf("mean = %v, want 4", mean[0])
	}
	// population variance = ((2-4)^2+0+0+(6-4)^2)/4 = 2 -> sd = sqrt(2)
	if math.Abs(sd[0]-math.Sqrt2) > 1e-9 {
		t.Errorf("stddev = %v, want sqrt2", sd[0])
	}
}

func TestNormsAndPSNR(t *testing.T) {
	m := NewMat(1, 3, 1)
	copy(m.Data, []uint8{1, 2, 2})
	if NormL1Mat(m) != 5 {
		t.Errorf("L1 = %v", NormL1Mat(m))
	}
	if math.Abs(NormL2Mat(m)-3) > 1e-9 {
		t.Errorf("L2 = %v, want 3", NormL2Mat(m))
	}
	if NormInfMat(m) != 2 {
		t.Errorf("Inf = %v", NormInfMat(m))
	}
	a := NewMat(2, 2, 1)
	b := a.Clone()
	if !math.IsInf(PSNR(a, b), 1) {
		t.Error("identical PSNR should be +Inf")
	}
}
