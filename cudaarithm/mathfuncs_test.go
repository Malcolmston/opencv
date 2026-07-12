package cudaarithm

import (
	"math"
	"testing"

	cv "github.com/malcolmston/opencv"
)

func TestSqrtPowExpLog(t *testing.T) {
	src := cv.NewMat(1, 4, 1)
	copy(src.Data, []uint8{0, 4, 9, 16})
	g := NewGpuMat(src)

	sqrt := Sqrt(g).Download()
	for i, w := range []uint8{0, 2, 3, 4} {
		if sqrt.Data[i] != w {
			t.Errorf("Sqrt[%d] = %d, want %d", i, sqrt.Data[i], w)
		}
	}

	// Pow with 0.5 is sqrt; check it matches.
	pow := Pow(g, 0.5).Download()
	if !sameData(pow, sqrt) {
		t.Error("Pow(x,0.5) should equal Sqrt(x)")
	}

	small := cv.NewMat(1, 3, 1)
	copy(small.Data, []uint8{0, 1, 2})
	gs := NewGpuMat(small)
	exp := Exp(gs).Download()
	for i, s := range small.Data {
		want := roundToUint8(math.Exp(float64(s)))
		if exp.Data[i] != want {
			t.Errorf("Exp[%d] = %d, want %d", i, exp.Data[i], want)
		}
	}

	lg := cv.NewMat(1, 3, 1)
	copy(lg.Data, []uint8{0, 1, 100})
	log := Log(NewGpuMat(lg)).Download()
	if log.Data[0] != 0 || log.Data[1] != 0 { // log(0)->0, log(1)=0
		t.Errorf("Log of 0 and 1 = %d,%d want 0,0", log.Data[0], log.Data[1])
	}
	if log.Data[2] != roundToUint8(math.Log(100)) {
		t.Errorf("Log(100) = %d, want %d", log.Data[2], roundToUint8(math.Log(100)))
	}
}

func TestMagnitudePhaseRoundTrip(t *testing.T) {
	// Classic 3-4-5 triangle: magnitude should be exactly 5.
	x := constMat(2, 2, 3)
	y := constMat(2, 2, 4)
	gx, gy := NewGpuMat(x), NewGpuMat(y)

	mag := Magnitude(gx, gy).Download()
	for _, v := range mag.Data {
		if v != 5 {
			t.Fatalf("magnitude = %d, want 5", v)
		}
	}

	// CartToPolar then PolarToCart should approximately recover x and y.
	m, a := CartToPolar(gx, gy, false)
	rx, ry := PolarToCart(m, a, false)
	dx := rx.Download()
	dy := ry.Download()
	for i := range dx.Data {
		if absInt(int(dx.Data[i])-3) > 1 {
			t.Errorf("recovered x[%d] = %d, want ~3", i, dx.Data[i])
		}
		if absInt(int(dy.Data[i])-4) > 1 {
			t.Errorf("recovered y[%d] = %d, want ~4", i, dy.Data[i])
		}
	}
}

func TestPhaseDegrees(t *testing.T) {
	// Vector (1,1) has phase 45 degrees.
	x := constMat(1, 1, 10)
	y := constMat(1, 1, 10)
	ph := Phase(NewGpuMat(x), NewGpuMat(y), true).Download()
	if ph.Data[0] != 45 {
		t.Errorf("phase = %d degrees, want 45", ph.Data[0])
	}
}

func absInt(v int) int {
	if v < 0 {
		return -v
	}
	return v
}
