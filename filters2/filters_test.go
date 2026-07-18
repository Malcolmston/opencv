package filters2

import (
	"testing"

	cv "github.com/malcolmston/opencv"
)

// --- test image builders ---

func constGray(rows, cols int, v uint8) *cv.Mat {
	m := cv.NewMat(rows, cols, 1)
	for i := range m.Data {
		m.Data[i] = v
	}
	return m
}

func constColor(rows, cols, ch int, v uint8) *cv.Mat {
	m := cv.NewMat(rows, cols, ch)
	for i := range m.Data {
		m.Data[i] = v
	}
	return m
}

func rangeOf(m *cv.Mat) (min, max uint8) {
	min, max = m.Data[0], m.Data[0]
	for _, v := range m.Data {
		if v < min {
			min = v
		}
		if v > max {
			max = v
		}
	}
	return min, max
}

func assertConstant(t *testing.T, m *cv.Mat, want uint8, name string) {
	t.Helper()
	for i, v := range m.Data {
		if v != want {
			t.Fatalf("%s: sample %d = %d, want constant %d", name, i, v, want)
			return
		}
	}
}

// --- constant-image invariance (many filters must be identity on flats) ---

func TestConstantInvariance(t *testing.T) {
	const v = 128
	g := constGray(9, 9, v)
	c := constColor(7, 8, 3, v)

	cases := []struct {
		name string
		out  *cv.Mat
	}{
		{"BilateralFilter", BilateralFilter(g, 5, 30, 3)},
		{"BilateralFilterColor", BilateralFilter(c, 5, 30, 3)},
		{"JointBilateral", JointBilateralFilter(g, g, 5, 30, 3)},
		{"GuidedFilter", GuidedFilter(g, g, 2, 0.01)},
		{"NonLocalMeans", NonLocalMeans(g, 10, 1, 3)},
		{"AnisotropicDiffusionExp", AnisotropicDiffusion(g, 10, 20, 0.2, DiffusionExponential)},
		{"AnisotropicDiffusionQuad", AnisotropicDiffusion(g, 10, 20, 0.2, DiffusionQuadratic)},
		{"Kuwahara", KuwaharaFilter(g, 5)},
		{"Median", MedianFilter(g, 3)},
		{"Min", MinFilter(g, 3)},
		{"Max", MaxFilter(g, 3)},
		{"Midpoint", MidpointFilter(g, 3)},
		{"AlphaTrimmed", AlphaTrimmedMeanFilter(g, 3, 2)},
		{"AdaptiveMedian", AdaptiveMedianFilter(g, 5)},
		{"UnsharpMask", UnsharpMask(g, 1.0, 1.5, 0)},
		{"GaussianBlur", GaussianBlur(g, 1.2)},
	}
	for _, tc := range cases {
		assertConstant(t, tc.out, v, tc.name)
	}
}

func TestHighBoostOnConstant(t *testing.T) {
	g := constGray(6, 6, 100)
	// out = boost*v - blur; on a flat, blur == v, so out = (boost-1)*v.
	assertConstant(t, HighBoostFilter(g, 1.2, 1.0), 0, "HighBoost boost=1")
	assertConstant(t, HighBoostFilter(g, 1.2, 2.0), 100, "HighBoost boost=2")
}

// --- averaging filters stay within the input range (maximum principle) ---

func TestWithinRange(t *testing.T) {
	// A deterministic pseudo-random-ish pattern.
	m := cv.NewMat(12, 12, 1)
	for y := 0; y < 12; y++ {
		for x := 0; x < 12; x++ {
			m.Data[y*12+x] = uint8((y*37 + x*91) % 256)
		}
	}
	lo, hi := rangeOf(m)
	check := func(name string, out *cv.Mat) {
		olo, ohi := rangeOf(out)
		if olo < lo || ohi > hi {
			t.Errorf("%s: output range [%d,%d] escapes input range [%d,%d]", name, olo, ohi, lo, hi)
		}
	}
	check("Bilateral", BilateralFilter(m, 5, 40, 3))
	check("NonLocalMeans", NonLocalMeans(m, 15, 1, 3))
	check("AnisotropicDiffusion", AnisotropicDiffusion(m, 20, 25, 0.25, DiffusionExponential))
	check("Median", MedianFilter(m, 3))
	check("Kuwahara", KuwaharaFilter(m, 5))
	check("GuidedFilter", GuidedFilter(m, m, 2, 0.02))
}

// --- median removes an isolated impulse (known answer) ---

func TestMedianRemovesImpulse(t *testing.T) {
	m := constGray(5, 5, 10)
	m.Data[2*5+2] = 255 // salt in the centre
	out := MedianFilter(m, 3)
	if got := out.Data[2*5+2]; got != 10 {
		t.Errorf("median centre = %d, want 10", got)
	}
}

func TestAdaptiveMedianRemovesImpulseKeepsFlat(t *testing.T) {
	m := constGray(7, 7, 50)
	m.Data[3*7+3] = 255
	m.Data[1*7+5] = 0
	out := AdaptiveMedianFilter(m, 5)
	if got := out.Data[3*7+3]; got != 50 {
		t.Errorf("adaptive median removed salt to %d, want 50", got)
	}
	if got := out.Data[1*7+5]; got != 50 {
		t.Errorf("adaptive median removed pepper to %d, want 50", got)
	}
	// A pixel far from any impulse is untouched.
	if got := out.Data[5*7+1]; got != 50 {
		t.Errorf("adaptive median altered a flat pixel to %d, want 50", got)
	}
}

func TestMinMaxFilters(t *testing.T) {
	m := constGray(5, 5, 20)
	m.Data[2*5+2] = 200
	// Max spreads the bright pixel to its 3x3 neighbourhood.
	mx := MaxFilter(m, 3)
	if mx.Data[2*5+2] != 200 || mx.Data[1*5+1] != 200 {
		t.Errorf("max filter did not spread the peak")
	}
	// Min erases it.
	mn := MinFilter(m, 3)
	if mn.Data[2*5+2] != 20 {
		t.Errorf("min filter centre = %d, want 20", mn.Data[2*5+2])
	}
}

func TestMidpointFilter(t *testing.T) {
	m := constGray(3, 3, 10)
	m.Data[0] = 40
	// Window at centre spans values {10..40}; midpoint = (10+40)/2 = 25.
	out := MidpointFilter(m, 3)
	if out.Data[1*3+1] != 25 {
		t.Errorf("midpoint centre = %d, want 25", out.Data[1*3+1])
	}
}

func TestAlphaTrimmedRemovesImpulse(t *testing.T) {
	m := constGray(5, 5, 30)
	m.Data[2*5+2] = 255
	// Trimming the extremes discards the impulse; the rest are all 30.
	out := AlphaTrimmedMeanFilter(m, 3, 1)
	if out.Data[2*5+2] != 30 {
		t.Errorf("alpha-trimmed centre = %d, want 30", out.Data[2*5+2])
	}
}

// --- Kuwahara picks the flat quadrant across an edge ---

func TestKuwaharaEdge(t *testing.T) {
	// Vertical edge: columns < 2 are 10, columns >= 2 are 200.
	rows, cols := 5, 5
	m := cv.NewMat(rows, cols, 1)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			if x < 2 {
				m.Data[y*cols+x] = 10
			} else {
				m.Data[y*cols+x] = 200
			}
		}
	}
	out := KuwaharaFilter(m, 3)
	// The last low column (x=1) has a flat 2x2 quadrant of value 10, so it is
	// smoothed to 10; the first high column (x=2) likewise resolves to 200.
	// Every quadrant is uniform on one side of the edge, so edge pixels take a
	// pure region value rather than an averaged (blurred) one.
	if got := out.Data[2*cols+1]; got != 10 {
		t.Errorf("kuwahara at low edge column = %d, want 10", got)
	}
	if got := out.Data[2*cols+2]; got != 200 {
		t.Errorf("kuwahara at high edge column = %d, want 200", got)
	}
	// Deep inside the flat regions the value is unchanged.
	if got := out.Data[2*cols+0]; got != 10 {
		t.Errorf("kuwahara flat-left = %d, want 10", got)
	}
	if got := out.Data[2*cols+4]; got != 200 {
		t.Errorf("kuwahara flat-right = %d, want 200", got)
	}
}

// --- guided filter self-guided identity on a varying image (eps -> 0) ---

func TestGuidedSelfIdentity(t *testing.T) {
	// A diagonal ramp so every window has non-zero variance.
	rows, cols := 8, 8
	m := cv.NewMat(rows, cols, 1)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			m.Data[y*cols+x] = uint8(10 + 8*x + 8*y)
		}
	}
	out := GuidedFilter(m, m, 2, 1e-9)
	for i := range m.Data {
		if d := int(out.Data[i]) - int(m.Data[i]); d < -1 || d > 1 {
			t.Fatalf("guided self-filter differs at %d: got %d want %d", i, out.Data[i], m.Data[i])
		}
	}
}

// --- unsharp / high-boost sharpen a step edge (increase local contrast) ---

func TestUnsharpIncreasesContrast(t *testing.T) {
	rows, cols := 1, 9
	m := cv.NewMat(rows, cols, 1)
	for x := 0; x < cols; x++ {
		if x < 4 {
			m.Data[x] = 80
		} else {
			m.Data[x] = 160
		}
	}
	out := UnsharpMask(m, 1.0, 2.0, 0)
	// Overshoot: just left of the edge should dip below 80, just right rise above 160.
	if out.Data[3] >= 80 {
		t.Errorf("no undershoot at x=3: %d", out.Data[3])
	}
	if out.Data[4] <= 160 {
		t.Errorf("no overshoot at x=4: %d", out.Data[4])
	}
}
