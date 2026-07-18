package moments2

import (
	"math"
	"testing"

	cv "github.com/malcolmston/opencv"
)

// filledMat returns a rows x cols single-channel Mat with every pixel set to v.
func filledMat(rows, cols int, v uint8) *cv.Mat {
	m := cv.NewMat(rows, cols, 1)
	for i := range m.Data {
		m.Data[i] = v
	}
	return m
}

// filledDisk returns a size x size single-channel Mat with a centred filled disk
// of the given radius set to 255.
func filledDisk(size int, radius float64) *cv.Mat {
	m := cv.NewMat(size, size, 1)
	c := float64(size-1) / 2
	for y := 0; y < size; y++ {
		for x := 0; x < size; x++ {
			if math.Hypot(float64(x)-c, float64(y)-c) <= radius {
				m.Data[y*size+x] = 255
			}
		}
	}
	return m
}

func approx(t *testing.T, name string, got, want, tol float64) {
	t.Helper()
	if math.Abs(got-want) > tol {
		t.Errorf("%s = %g, want %g (tol %g)", name, got, want, tol)
	}
}

func TestImageMomentsRectangle(t *testing.T) {
	// 3 rows x 5 cols filled with 1.
	m := ImageMoments(filledMat(3, 5, 1))
	approx(t, "M00", m.M00, 15, 0)
	approx(t, "M10", m.M10, 30, 0)
	approx(t, "M01", m.M01, 15, 0)
	approx(t, "M20", m.M20, 90, 0)
	approx(t, "M02", m.M02, 25, 0)
	approx(t, "M11", m.M11, 30, 0)
	approx(t, "Mu20", m.Mu20, 30, 1e-9)
	approx(t, "Mu02", m.Mu02, 10, 1e-9)
	approx(t, "Mu11", m.Mu11, 0, 1e-9)
	c := m.Centroid()
	approx(t, "cx", c.X, 2, 1e-12)
	approx(t, "cy", c.Y, 1, 1e-12)
	approx(t, "Nu20", m.Nu20, 30.0/225.0, 1e-12)
}

func TestMaskMomentsIgnoresValue(t *testing.T) {
	// A mask of 255 must yield the same geometry as a mask of 1.
	a := MaskMoments(filledMat(3, 5, 255))
	if a.M00 != 15 {
		t.Fatalf("mask M00 = %g, want 15", a.M00)
	}
	ca := a.Centroid()
	approx(t, "mask cx", ca.X, 2, 1e-12)
	// ImageMoments weights by value, so M00 scales by 255.
	img := ImageMoments(filledMat(3, 5, 255))
	approx(t, "img M00", img.M00, 15*255, 0)
}

func TestRawCentralNormalizedGeneral(t *testing.T) {
	src := filledMat(3, 5, 1)
	approx(t, "RawMoment(0,0)", RawMoment(src, 0, 0), 15, 0)
	approx(t, "RawMoment(2,0)", RawMoment(src, 2, 0), 90, 0)
	approx(t, "CentralMoment(2,0)", CentralMoment(src, 2, 0), 30, 1e-9)
	approx(t, "CentralMoment(0,2)", CentralMoment(src, 0, 2), 10, 1e-9)
	approx(t, "NormCentral(2,0)", NormalizedCentralMoment(src, 2, 0), 30.0/225.0, 1e-12)
}

func TestContourMomentsRectangle(t *testing.T) {
	rect := []cv.Point{{X: 0, Y: 0}, {X: 4, Y: 0}, {X: 4, Y: 2}, {X: 0, Y: 2}}
	m := ContourMoments(rect)
	approx(t, "area", m.M00, 8, 1e-9)
	c := m.Centroid()
	approx(t, "cx", c.X, 2, 1e-9)
	approx(t, "cy", c.Y, 1, 1e-9)
	approx(t, "Mu20", m.Mu20, 8*16.0/12.0, 1e-6)
	approx(t, "Mu02", m.Mu02, 8*4.0/12.0, 1e-6)
	approx(t, "Mu11", m.Mu11, 0, 1e-6)
	approx(t, "Mu30", m.Mu30, 0, 1e-6)
	approx(t, "Mu03", m.Mu03, 0, 1e-6)
}

func TestHuTranslationInvariance(t *testing.T) {
	small := cv.NewMat(30, 30, 1)
	big := cv.NewMat(50, 50, 1)
	// An L-shaped region placed at two different offsets.
	stamp := func(m *cv.Mat, ox, oy int) {
		for y := 0; y < 10; y++ {
			for x := 0; x < 6; x++ {
				m.Data[(y+oy)*m.Cols+(x+ox)] = 255
			}
		}
		for y := 7; y < 10; y++ {
			for x := 6; x < 14; x++ {
				m.Data[(y+oy)*m.Cols+(x+ox)] = 255
			}
		}
	}
	stamp(small, 2, 2)
	stamp(big, 20, 30)
	ha := HuMoments(ImageMoments(small))
	hb := HuMoments(ImageMoments(big))
	for i := 0; i < 7; i++ {
		if math.Abs(ha[i]-hb[i]) > 1e-9 {
			t.Errorf("Hu[%d] not translation invariant: %g vs %g", i, ha[i], hb[i])
		}
	}
	// A shape matched against itself scores zero.
	if c := MatchShapes(ImageMoments(small), ImageMoments(big), MatchI2); c > 1e-9 {
		t.Errorf("MatchShapes of translated shape = %g, want ~0", c)
	}
}

func TestHuScaleInvariance(t *testing.T) {
	square := func(side int) *cv.Mat {
		m := cv.NewMat(side+4, side+4, 1)
		for y := 2; y < side+2; y++ {
			for x := 2; x < side+2; x++ {
				m.Data[y*m.Cols+x] = 255
			}
		}
		return m
	}
	h1 := HuMoments(ImageMoments(square(20)))
	h2 := HuMoments(ImageMoments(square(40)))
	approx(t, "Hu[0] scale", h1[0], h2[0], 5e-3)
}

func TestMatchShapesMethods(t *testing.T) {
	m := ImageMoments(filledMat(10, 10, 255))
	for _, method := range []MatchMethod{MatchI1, MatchI2, MatchI3} {
		if c := MatchShapes(m, m, method); c != 0 {
			t.Errorf("self-match method %d = %g, want 0", method, c)
		}
	}
}
