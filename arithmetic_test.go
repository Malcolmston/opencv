package cv

import "testing"

func TestAddSaturates(t *testing.T) {
	a := grayFromValues(1, 3, []uint8{100, 200, 50})
	b := grayFromValues(1, 3, []uint8{100, 100, 10})
	out := Add(a, b)
	want := []uint8{200, 255, 60}
	for i, w := range want {
		if out.Data[i] != w {
			t.Errorf("Add[%d] = %d, want %d", i, out.Data[i], w)
		}
	}
}

func TestSubtractSaturates(t *testing.T) {
	a := grayFromValues(1, 3, []uint8{100, 30, 255})
	b := grayFromValues(1, 3, []uint8{50, 60, 5})
	out := Subtract(a, b)
	want := []uint8{50, 0, 250}
	for i, w := range want {
		if out.Data[i] != w {
			t.Errorf("Subtract[%d] = %d, want %d", i, out.Data[i], w)
		}
	}
}

func TestAbsDiff(t *testing.T) {
	a := grayFromValues(1, 2, []uint8{10, 200})
	b := grayFromValues(1, 2, []uint8{50, 100})
	out := AbsDiff(a, b)
	if out.Data[0] != 40 || out.Data[1] != 100 {
		t.Errorf("AbsDiff = %v, want [40 100]", out.Data)
	}
}

func TestAddWeightedExact(t *testing.T) {
	a := grayFromValues(1, 2, []uint8{100, 200})
	b := grayFromValues(1, 2, []uint8{50, 40})
	out := AddWeighted(a, 0.5, b, 0.5, 0)
	// 0.5*100+0.5*50 = 75; 0.5*200+0.5*40 = 120.
	if out.Data[0] != 75 || out.Data[1] != 120 {
		t.Errorf("AddWeighted = %v, want [75 120]", out.Data)
	}
}

func TestMultiplyDivide(t *testing.T) {
	a := grayFromValues(1, 2, []uint8{10, 20})
	b := grayFromValues(1, 2, []uint8{3, 4})
	m := Multiply(a, b, 1)
	if m.Data[0] != 30 || m.Data[1] != 80 {
		t.Errorf("Multiply = %v, want [30 80]", m.Data)
	}
	d := Divide(a, b, 1)
	// 10/3 = 3.33 -> 3; 20/4 = 5.
	if d.Data[0] != 3 || d.Data[1] != 5 {
		t.Errorf("Divide = %v, want [3 5]", d.Data)
	}
	// Division by zero yields 0.
	z := grayFromValues(1, 1, []uint8{0})
	one := grayFromValues(1, 1, []uint8{9})
	if Divide(one, z, 1).Data[0] != 0 {
		t.Error("Divide by zero should be 0")
	}
}

func TestBitwiseOps(t *testing.T) {
	a := grayFromValues(1, 1, []uint8{0xF0})
	b := grayFromValues(1, 1, []uint8{0x3C})
	if BitwiseAnd(a, b).Data[0] != 0x30 {
		t.Errorf("AND = %#x, want 0x30", BitwiseAnd(a, b).Data[0])
	}
	if BitwiseOr(a, b).Data[0] != 0xFC {
		t.Errorf("OR = %#x, want 0xFC", BitwiseOr(a, b).Data[0])
	}
	if BitwiseXor(a, b).Data[0] != 0xCC {
		t.Errorf("XOR = %#x, want 0xCC", BitwiseXor(a, b).Data[0])
	}
	if BitwiseNot(a).Data[0] != 0x0F {
		t.Errorf("NOT = %#x, want 0x0F", BitwiseNot(a).Data[0])
	}
}

func TestMinMaxMat(t *testing.T) {
	a := grayFromValues(1, 3, []uint8{10, 200, 30})
	b := grayFromValues(1, 3, []uint8{50, 100, 30})
	mn := Min(a, b)
	mx := Max(a, b)
	if mn.Data[0] != 10 || mn.Data[1] != 100 || mn.Data[2] != 30 {
		t.Errorf("Min = %v", mn.Data)
	}
	if mx.Data[0] != 50 || mx.Data[1] != 200 || mx.Data[2] != 30 {
		t.Errorf("Max = %v", mx.Data)
	}
}

func TestConvertScaleAbs(t *testing.T) {
	src := grayFromValues(1, 2, []uint8{10, 100})
	// 10*-2 + 0 = -20 -> |−20| = 20; 100*-2 = -200 -> 200.
	out := ConvertScaleAbs(src, -2, 0)
	if out.Data[0] != 20 || out.Data[1] != 200 {
		t.Errorf("ConvertScaleAbs = %v, want [20 200]", out.Data)
	}
}

func TestNormalizeMinMax(t *testing.T) {
	src := grayFromValues(1, 3, []uint8{50, 100, 150})
	out := Normalize(src, 0, 255)
	// 50->0, 150->255, 100->127 or 128.
	if out.Data[0] != 0 || out.Data[2] != 255 {
		t.Errorf("Normalize ends = %v, want [0 .. 255]", out.Data)
	}
	if out.Data[1] < 126 || out.Data[1] > 129 {
		t.Errorf("Normalize mid = %d, want ~127", out.Data[1])
	}
	// Constant image maps everything to alpha.
	c := grayFromValues(1, 2, []uint8{7, 7})
	nc := Normalize(c, 10, 200)
	if nc.Data[0] != 10 || nc.Data[1] != 10 {
		t.Errorf("Normalize constant = %v, want [10 10]", nc.Data)
	}
}

func TestArithmeticShapeMismatchPanics(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Error("expected panic on shape mismatch")
		}
	}()
	Add(NewMat(2, 2, 1), NewMat(2, 3, 1))
}
