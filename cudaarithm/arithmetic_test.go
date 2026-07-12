package cudaarithm

import (
	"testing"

	cv "github.com/malcolmston/opencv"
)

// ramp builds a rows×cols single-channel Mat whose samples are start, start+step,
// start+2*step, ... modulo 256.
func ramp(rows, cols int, start, step int) *cv.Mat {
	m := cv.NewMat(rows, cols, 1)
	v := start
	for i := range m.Data {
		m.Data[i] = uint8(v % 256)
		v += step
	}
	return m
}

// constMat builds a rows×cols single-channel Mat filled with value.
func constMat(rows, cols int, value uint8) *cv.Mat {
	m := cv.NewMat(rows, cols, 1)
	for i := range m.Data {
		m.Data[i] = value
	}
	return m
}

func sameData(a, b *cv.Mat) bool {
	if a.Rows != b.Rows || a.Cols != b.Cols || a.Channels != b.Channels {
		return false
	}
	for i := range a.Data {
		if a.Data[i] != b.Data[i] {
			return false
		}
	}
	return true
}

func TestElementwiseMatchRootOps(t *testing.T) {
	a := ramp(4, 5, 10, 7)
	b := ramp(4, 5, 200, 3)
	ga, gb := NewGpuMat(a), NewGpuMat(b)
	stream := NewStream()

	cases := []struct {
		name string
		got  *cv.Mat
		want *cv.Mat
	}{
		{"Add", Add(ga, gb, stream).Download(), cv.Add(a, b)},
		{"Subtract", Subtract(ga, gb).Download(), cv.Subtract(a, b)},
		{"Multiply", Multiply(ga, gb, 0.01).Download(), cv.Multiply(a, b, 0.01)},
		{"Divide", Divide(ga, gb, 2.0).Download(), cv.Divide(a, b, 2.0)},
		{"AbsDiff", AbsDiff(ga, gb).Download(), cv.AbsDiff(a, b)},
		{"AddWeighted", AddWeighted(ga, 0.3, gb, 0.7, 5).Download(), cv.AddWeighted(a, 0.3, b, 0.7, 5)},
		{"BitwiseAnd", BitwiseAnd(ga, gb).Download(), cv.BitwiseAnd(a, b)},
		{"BitwiseOr", BitwiseOr(ga, gb).Download(), cv.BitwiseOr(a, b)},
		{"BitwiseXor", BitwiseXor(ga, gb).Download(), cv.BitwiseXor(a, b)},
		{"BitwiseNot", BitwiseNot(ga).Download(), cv.BitwiseNot(a)},
		{"Min", Min(ga, gb).Download(), cv.Min(a, b)},
		{"Max", Max(ga, gb).Download(), cv.Max(a, b)},
	}
	for _, c := range cases {
		if !sameData(c.got, c.want) {
			t.Errorf("%s: GpuMat result does not match root cv op", c.name)
		}
	}
	stream.WaitForCompletion()
}

func TestAbsIsCopy(t *testing.T) {
	a := ramp(3, 3, 0, 30)
	g := NewGpuMat(a)
	got := Abs(g).Download()
	if !sameData(got, a) {
		t.Fatal("Abs of unsigned data should equal the input")
	}
}

func TestCompare(t *testing.T) {
	a := cv.NewMat(1, 4, 1)
	b := cv.NewMat(1, 4, 1)
	copy(a.Data, []uint8{10, 20, 30, 40})
	copy(b.Data, []uint8{10, 25, 20, 40})
	ga, gb := NewGpuMat(a), NewGpuMat(b)

	tests := []struct {
		op   CmpOp
		want []uint8
	}{
		{CmpEQ, []uint8{255, 0, 0, 255}},
		{CmpGT, []uint8{0, 0, 255, 0}},
		{CmpGE, []uint8{255, 0, 255, 255}},
		{CmpLT, []uint8{0, 255, 0, 0}},
		{CmpLE, []uint8{255, 255, 0, 255}},
		{CmpNE, []uint8{0, 255, 255, 0}},
	}
	for _, tc := range tests {
		got := Compare(ga, gb, tc.op).Download()
		for i, w := range tc.want {
			if got.Data[i] != w {
				t.Errorf("op %d index %d: got %d want %d", tc.op, i, got.Data[i], w)
			}
		}
	}
}

func TestThreshold(t *testing.T) {
	src := ramp(1, 6, 0, 50) // 0,50,100,150,200,250
	g := NewGpuMat(src)
	dst, used := Threshold(g, 120, 255, cv.ThreshBinary)
	if used != 120 {
		t.Fatalf("threshold used = %v, want 120", used)
	}
	want := []uint8{0, 0, 0, 255, 255, 255}
	got := dst.Download()
	for i, w := range want {
		if got.Data[i] != w {
			t.Errorf("index %d: got %d want %d", i, got.Data[i], w)
		}
	}
}

func TestShapeMismatchPanics(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("expected panic on shape mismatch")
		}
	}()
	Add(NewGpuMat(constMat(2, 2, 1)), NewGpuMat(constMat(3, 3, 1)))
}
