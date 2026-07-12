package cv

import "testing"

func TestFilter2DSepMatchesBlur(t *testing.T) {
	m := NewMat(6, 6, 1)
	for i := range m.Data {
		m.Data[i] = uint8((i * 7) % 256)
	}
	// A separable normalised 1x3 / 3x1 box equals a 3x3 mean blur.
	k := []float64{1.0 / 3, 1.0 / 3, 1.0 / 3}
	sep := Filter2DSep(m, k, k, 0)
	blur := Blur(m, 3)
	for i := range sep.Data {
		d := int(sep.Data[i]) - int(blur.Data[i])
		if d < -1 || d > 1 {
			t.Fatalf("sep vs blur differ at %d: %d vs %d", i, sep.Data[i], blur.Data[i])
		}
	}
}

func TestBilateralConstantImage(t *testing.T) {
	m := NewMat(9, 9, 1)
	m.SetTo(120)
	out := BilateralFilter(m, 5, 30, 3)
	for _, v := range out.Data {
		if v != 120 {
			t.Fatalf("bilateral of constant = %d, want 120", v)
		}
	}
}

func TestBilateralPreservesEdge(t *testing.T) {
	// A sharp vertical step edge: left half 0, right half 200.
	m := NewMat(9, 9, 1)
	for y := 0; y < 9; y++ {
		for x := 5; x < 9; x++ {
			m.Set(y, x, 0, 200)
		}
	}
	out := BilateralFilter(m, 5, 20, 3)
	// With a tight range sigma the edge stays close to its original levels.
	if out.At(4, 0, 0) > 20 {
		t.Errorf("dark side blurred too much: %d", out.At(4, 0, 0))
	}
	if out.At(4, 8, 0) < 180 {
		t.Errorf("bright side blurred too much: %d", out.At(4, 8, 0))
	}
}
