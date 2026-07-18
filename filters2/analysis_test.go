package filters2

import (
	"math"
	"testing"

	cv "github.com/malcolmston/opencv"
)

func rampX(rows, cols int, base, slope float64) *cv.Mat {
	m := cv.NewMat(rows, cols, 1)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			m.Data[y*cols+x] = clampU8(base + slope*float64(x))
		}
	}
	return m
}

func maxAbs(f *FloatImage) float64 {
	var m float64
	for _, v := range f.Data {
		if a := math.Abs(v); a > m {
			m = a
		}
	}
	return m
}

// --- LoG / DoG vanish on a constant region ---

func TestLoGDoGConstant(t *testing.T) {
	g := constGray(11, 11, 90)
	if m := maxAbs(LaplacianOfGaussian(g, 1.4, 0)); m > 1e-6 {
		t.Errorf("LoG on constant max|.| = %g, want ~0", m)
	}
	if m := maxAbs(DifferenceOfGaussians(g, 1.0, 2.0, 0)); m > 1e-6 {
		t.Errorf("DoG on constant max|.| = %g, want ~0", m)
	}
}

// --- Marr-Hildreth finds an edge at a step ---

func TestMarrHildrethStep(t *testing.T) {
	rows, cols := 9, 12
	m := cv.NewMat(rows, cols, 1)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			if x < 6 {
				m.Data[y*cols+x] = 30
			} else {
				m.Data[y*cols+x] = 220
			}
		}
	}
	edges := MarrHildrethEdges(m, 1.2, 1.0)
	// Some edge pixel must appear in the middle rows near the transition.
	found := false
	for y := 2; y < rows-2; y++ {
		for x := 4; x <= 7; x++ {
			if edges.Data[y*cols+x] == 255 {
				found = true
			}
		}
	}
	if !found {
		t.Errorf("Marr-Hildreth found no edge near the step")
	}
	// A flat corner far from the edge must not be an edge.
	if edges.Data[0] == 255 {
		t.Errorf("Marr-Hildreth marked a flat corner as an edge")
	}
}

// --- steerable G1: steering identity and response to a ramp ---

func TestSteerableG1(t *testing.T) {
	m := rampX(9, 9, 20, 12)
	s := SteerG1(m, 1.5, 0)

	// Steer(0) must equal the x basis exactly.
	z := s.Steer(0)
	for i := range z.Data {
		if z.Data[i] != s.Gx.Data[i] {
			t.Fatalf("Steer(0) != Gx at %d", i)
		}
	}
	// Steer(pi/2) must equal the y basis to floating tolerance.
	z = s.Steer(math.Pi / 2)
	for i := range z.Data {
		if math.Abs(z.Data[i]-s.Gy.Data[i]) > 1e-9 {
			t.Fatalf("Steer(pi/2) != Gy at %d: %g vs %g", i, z.Data[i], s.Gy.Data[i])
		}
	}
	// For a horizontal ramp the x-derivative is strong and the y-derivative
	// negligible in the interior.
	r := 9 / 2
	if math.Abs(s.Gx.At(r, r)) < 1 {
		t.Errorf("Gx interior too small for a ramp: %g", s.Gx.At(r, r))
	}
	if math.Abs(s.Gy.At(r, r)) > 1e-6 {
		t.Errorf("Gy interior should vanish for a horizontal ramp: %g", s.Gy.At(r, r))
	}
	// Dominant orientation of a vertical-edge ramp is horizontal (0 rad).
	if o := s.DominantOrientation().At(r, r); math.Abs(o) > 1e-6 {
		t.Errorf("dominant orientation = %g, want 0", o)
	}
}

// --- steerable G2: closed-form energy matches a dense orientation sweep ---

func TestSteerableG2Energy(t *testing.T) {
	// A textured deterministic image.
	rows, cols := 10, 10
	m := cv.NewMat(rows, cols, 1)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			m.Data[y*cols+x] = uint8((x*x + y*3) % 200)
		}
	}
	s := SteerG2(m, 1.5, 0)
	energy := s.OrientationEnergy()
	// Steer(0) equals Gxx exactly.
	if z := s.Steer(0); z.Data[0] != s.Gxx.Data[0] {
		t.Errorf("Steer(0) != Gxx")
	}
	// The closed-form max-magnitude response must dominate every sampled angle.
	for i := 0; i < rows*cols; i++ {
		y, x := i/cols, i%cols
		var best float64
		for k := 0; k < 180; k++ {
			theta := math.Pi * float64(k) / 180
			v := math.Abs(s.Steer(theta).At(y, x))
			if v > best {
				best = v
			}
		}
		if math.Abs(energy.Data[i]) < best-1e-6 {
			t.Fatalf("closed-form energy %g < sampled max %g at %d", energy.Data[i], best, i)
		}
	}
}

// --- Gabor is orientation-selective for a grating ---

func TestGaborOrientationSelectivity(t *testing.T) {
	// A vertical grating: intensity varies along x with period 6.
	rows, cols := 21, 21
	period := 6.0
	m := cv.NewMat(rows, cols, 1)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			v := 128 + 100*math.Cos(2*math.Pi*float64(x)/period)
			m.Data[y*cols+x] = clampU8(v)
		}
	}
	// A filter tuned to the grating (theta=0, lambda=period) versus one tuned
	// to the orthogonal orientation.
	tuned := GaborParams{Sigma: 4, Theta: 0, Lambda: period, Gamma: 0.5, Psi: 0}
	ortho := GaborParams{Sigma: 4, Theta: math.Pi / 2, Lambda: period, Gamma: 0.5, Psi: 0}
	mt := GaborMagnitude(m, 15, tuned)
	mo := GaborMagnitude(m, 15, ortho)
	cy, cx := rows/2, cols/2
	if mt.At(cy, cx) <= mo.At(cy, cx) {
		t.Errorf("tuned Gabor response %g not greater than orthogonal %g", mt.At(cy, cx), mo.At(cy, cx))
	}
}

func TestGaborBankShape(t *testing.T) {
	bank := GaborBank(4, []float64{4, 8}, 3, 0.5, 0)
	if len(bank) != 8 {
		t.Fatalf("bank size = %d, want 8", len(bank))
	}
	g := constGray(15, 15, 100)
	resp := GaborBankResponse(g, 11, bank)
	if len(resp) != 8 {
		t.Fatalf("responses = %d, want 8", len(resp))
	}
	energy := GaborEnergy(g, 11, bank)
	if energy.Rows != 15 || energy.Cols != 15 || energy.Channels != 1 {
		t.Errorf("energy shape = %dx%dx%d", energy.Rows, energy.Cols, energy.Channels)
	}
}

// --- FloatImage helpers ---

func TestFloatImageOps(t *testing.T) {
	f := NewFloatImage(2, 3)
	f.Set(0, 0, -4)
	f.Set(1, 2, 8)
	min, max := f.MinMax()
	if min != -4 || max != 8 {
		t.Errorf("MinMax = (%g,%g), want (-4,8)", min, max)
	}
	// Normalize maps min->0 and max->255.
	n := f.Normalize()
	if n.Data[0] != 0 {
		t.Errorf("normalized min = %d, want 0", n.Data[0])
	}
	if n.Data[1*3+2] != 255 {
		t.Errorf("normalized max = %d, want 255", n.Data[1*3+2])
	}
	// Abs, Scale, Add, Sub.
	if f.Abs().At(0, 0) != 4 {
		t.Errorf("Abs failed")
	}
	if f.Scale(2).At(1, 2) != 16 {
		t.Errorf("Scale failed")
	}
	g := f.Clone()
	if f.Add(g).At(1, 2) != 16 || f.Sub(g).At(1, 2) != 0 {
		t.Errorf("Add/Sub failed")
	}
	// Magnitude of (3,4) is 5.
	a := NewFloatImage(1, 1)
	b := NewFloatImage(1, 1)
	a.Data[0], b.Data[0] = 3, 4
	if Magnitude(a, b).Data[0] != 5 {
		t.Errorf("Magnitude(3,4) != 5")
	}
}

func TestConvolveIdentity(t *testing.T) {
	f := NewFloatImage(3, 3)
	for i := range f.Data {
		f.Data[i] = float64(i)
	}
	id := [][]float64{{0, 0, 0}, {0, 1, 0}, {0, 0, 0}}
	out := Convolve(f, id)
	for i := range f.Data {
		if out.Data[i] != f.Data[i] {
			t.Fatalf("identity convolution changed sample %d", i)
		}
	}
}

func TestMatToFloatRoundTrip(t *testing.T) {
	g := constGray(4, 4, 77)
	f := MatToFloatImage(g)
	back := f.ToMat()
	for i := range g.Data {
		if back.Data[i] != g.Data[i] {
			t.Fatalf("round trip changed sample %d", i)
		}
	}
}

// --- benchmark of the heaviest routine ---

func BenchmarkNonLocalMeans(b *testing.B) {
	m := cv.NewMat(48, 48, 1)
	for i := range m.Data {
		m.Data[i] = uint8((i * 7) % 256)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = NonLocalMeans(m, 12, 2, 5)
	}
}
