package cudaarithm

import (
	"math"
	"testing"

	cv "github.com/malcolmston/opencv"
)

func TestSumFamily(t *testing.T) {
	src := cv.NewMat(1, 3, 1)
	copy(src.Data, []uint8{1, 2, 3})
	g := NewGpuMat(src)

	if got := Sum(g); got[0] != 6 {
		t.Errorf("Sum = %v, want 6", got[0])
	}
	if got := AbsSum(g); got[0] != 6 {
		t.Errorf("AbsSum = %v, want 6", got[0])
	}
	if got := SqrSum(g); got[0] != 14 { // 1+4+9
		t.Errorf("SqrSum = %v, want 14", got[0])
	}
}

func TestMultiChannelSum(t *testing.T) {
	src := cv.NewMat(1, 2, 3)
	copy(src.Data, []uint8{1, 2, 3, 4, 5, 6})
	got := Sum(NewGpuMat(src))
	want := []float64{5, 7, 9}
	for c := range want {
		if got[c] != want[c] {
			t.Errorf("Sum channel %d = %v, want %v", c, got[c], want[c])
		}
	}
}

func TestMeanStdDev(t *testing.T) {
	src := cv.NewMat(1, 4, 1)
	copy(src.Data, []uint8{2, 4, 4, 6}) // mean 4, variance (4+0+0+4)/4=2
	mean, stddev := MeanStdDev(NewGpuMat(src))
	if mean[0] != 4 {
		t.Errorf("mean = %v, want 4", mean[0])
	}
	if math.Abs(stddev[0]-math.Sqrt(2)) > 1e-9 {
		t.Errorf("stddev = %v, want sqrt(2)", stddev[0])
	}
}

func TestMinMaxLoc(t *testing.T) {
	src := cv.NewMat(2, 3, 1)
	copy(src.Data, []uint8{5, 9, 2, 7, 1, 8})
	g := NewGpuMat(src)

	lo, hi := MinMax(g)
	if lo != 1 || hi != 9 {
		t.Errorf("MinMax = (%v,%v), want (1,9)", lo, hi)
	}
	lo, hi, minX, minY, maxX, maxY := MinMaxLoc(g)
	if lo != 1 || minX != 1 || minY != 1 {
		t.Errorf("min loc = (%d,%d) val %v, want (1,1) val 1", minX, minY, lo)
	}
	if hi != 9 || maxX != 1 || maxY != 0 {
		t.Errorf("max loc = (%d,%d) val %v, want (1,0) val 9", maxX, maxY, hi)
	}
}

func TestCountNonZero(t *testing.T) {
	src := cv.NewMat(1, 5, 1)
	copy(src.Data, []uint8{0, 3, 0, 7, 1})
	if got := CountNonZero(NewGpuMat(src)); got != 3 {
		t.Errorf("CountNonZero = %d, want 3", got)
	}
}
