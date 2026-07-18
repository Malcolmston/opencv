package photo2

import (
	"math"
	"testing"

	cv "github.com/malcolmston/opencv"
)

// mkRGB builds a 3-channel Mat from a flat RGB slice (len rows*cols*3).
func mkRGB(t *testing.T, rows, cols int, data []uint8) *cv.Mat {
	t.Helper()
	if len(data) != rows*cols*3 {
		t.Fatalf("mkRGB: bad length %d for %dx%dx3", len(data), rows, cols)
	}
	m := cv.NewMat(rows, cols, 3)
	copy(m.Data, data)
	return m
}

// mkGray builds a single-channel Mat.
func mkGray(t *testing.T, rows, cols int, data []uint8) *cv.Mat {
	t.Helper()
	if len(data) != rows*cols {
		t.Fatalf("mkGray: bad length")
	}
	m := cv.NewMat(rows, cols, 1)
	copy(m.Data, data)
	return m
}

// constRGB builds a rows x cols image filled with (r,g,b).
func constRGB(rows, cols int, r, g, b uint8) *cv.Mat {
	m := cv.NewMat(rows, cols, 3)
	for i := 0; i < rows*cols; i++ {
		m.Data[i*3+0] = r
		m.Data[i*3+1] = g
		m.Data[i*3+2] = b
	}
	return m
}

func absDiff(a, b uint8) int {
	d := int(a) - int(b)
	if d < 0 {
		return -d
	}
	return d
}

func TestToFromFloatRoundTrip(t *testing.T) {
	img := mkRGB(t, 1, 2, []uint8{0, 128, 255, 10, 20, 30})
	planes := ToFloat(img)
	if len(planes) != 3 {
		t.Fatalf("expected 3 planes, got %d", len(planes))
	}
	if math.Abs(planes[0].Data[0]-0) > 1e-9 || math.Abs(planes[2].Data[0]-1) > 1e-9 {
		t.Fatalf("ToFloat scaling wrong: %v", planes[0].Data)
	}
	back := FromFloat(planes)
	for i := range img.Data {
		if absDiff(img.Data[i], back.Data[i]) > 0 {
			t.Fatalf("roundtrip mismatch at %d: %d vs %d", i, img.Data[i], back.Data[i])
		}
	}
}

func TestGrayscaleKnown(t *testing.T) {
	// Pure green -> 0.7152*255 = 182.4 -> 182.
	img := constRGB(1, 1, 0, 255, 0)
	g := Grayscale(img)
	if absDiff(g.Data[0], 182) > 1 {
		t.Fatalf("green luma = %d, want ~182", g.Data[0])
	}
	// Pure white -> 255.
	if Grayscale(constRGB(1, 1, 255, 255, 255)).Data[0] != 255 {
		t.Fatalf("white luma not 255")
	}
}

func TestLuminanceConstant(t *testing.T) {
	l := Luminance(constRGB(2, 2, 100, 100, 100))
	for _, v := range l.Data {
		if math.Abs(v-100.0/255) > 1e-9 {
			t.Fatalf("luminance = %v", v)
		}
	}
}

func TestNormalize(t *testing.T) {
	f := cv.NewFloatMat(1, 3)
	f.Data = []float64{2, 4, 6}
	n := Normalize(f)
	want := []float64{0, 0.5, 1}
	for i := range want {
		if math.Abs(n.Data[i]-want[i]) > 1e-9 {
			t.Fatalf("Normalize %v", n.Data)
		}
	}
}

func TestGaussianBlurConstant(t *testing.T) {
	img := constRGB(5, 5, 77, 88, 99)
	out := GaussianBlur(img, 1.5)
	for i := range img.Data {
		if absDiff(img.Data[i], out.Data[i]) > 1 {
			t.Fatalf("blur of constant changed pixel: %d vs %d", img.Data[i], out.Data[i])
		}
	}
}

func TestBoxBlurConstant(t *testing.T) {
	img := constRGB(6, 6, 50, 60, 70)
	out := BoxBlur(img, 2)
	for i := range img.Data {
		if absDiff(img.Data[i], out.Data[i]) > 1 {
			t.Fatalf("box blur of constant changed pixel")
		}
	}
}

func TestGaussianBlurSymmetricImpulse(t *testing.T) {
	// A centred impulse must blur into a symmetric result.
	g := mkGray(t, 5, 5, make([]uint8, 25))
	g.Data[2*5+2] = 255
	planes := ToFloat(g)
	b := GaussianBlurFloat(planes[0], 1.0)
	if math.Abs(b.At(2, 1)-b.At(2, 3)) > 1e-9 || math.Abs(b.At(1, 2)-b.At(3, 2)) > 1e-9 {
		t.Fatalf("blur not symmetric")
	}
	// The centre stays the maximum and neighbours are strictly positive but
	// smaller (energy spreads outward from the impulse).
	if b.At(2, 2) <= b.At(2, 1) || b.At(2, 1) <= 0 {
		t.Fatalf("impulse did not spread correctly: centre=%v side=%v", b.At(2, 2), b.At(2, 1))
	}
}

func TestLaplacianPyramidReconstruct(t *testing.T) {
	// Deterministic ramp; reconstruction must be near-exact.
	rows, cols := 16, 16
	f := cv.NewFloatMat(rows, cols)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			f.Data[y*cols+x] = float64(x*3+y*2) / 255
		}
	}
	lap := LaplacianPyramid(f, 4)
	rec := ReconstructLaplacianPyramid(lap)
	for i := range f.Data {
		if math.Abs(f.Data[i]-rec.Data[i]) > 1e-9 {
			t.Fatalf("pyramid reconstruction error at %d: %v vs %v", i, f.Data[i], rec.Data[i])
		}
	}
}

func TestPyrDownSize(t *testing.T) {
	f := cv.NewFloatMat(10, 7)
	d := PyrDown(f)
	if d.Rows != 5 || d.Cols != 4 {
		t.Fatalf("PyrDown size = %dx%d, want 5x4", d.Rows, d.Cols)
	}
}
