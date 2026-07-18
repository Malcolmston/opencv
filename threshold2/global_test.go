package threshold2

import (
	"testing"

	cv "github.com/malcolmston/opencv"
)

// grayImage builds a single-channel Mat from a fill function.
func grayImage(rows, cols int, fill func(y, x int) uint8) *cv.Mat {
	m := cv.NewMat(rows, cols, 1)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			m.Data[y*cols+x] = fill(y, x)
		}
	}
	return m
}

// twoBands returns a rows x cols image whose left half is value a and right
// half is value b.
func twoBands(rows, cols int, a, b uint8) *cv.Mat {
	return grayImage(rows, cols, func(_, x int) uint8 {
		if x < cols/2 {
			return a
		}
		return b
	})
}

func TestMeanThreshold(t *testing.T) {
	// Half the pixels are 0, half are 100: mean is 50.
	img := twoBands(8, 8, 0, 100)
	got, err := MeanThreshold(img)
	if err != nil {
		t.Fatal(err)
	}
	if got != 50 {
		t.Fatalf("MeanThreshold = %d, want 50", got)
	}
}

func TestMedianThreshold(t *testing.T) {
	// Three quarters are 40, one quarter is 200; median level is 40.
	img := grayImage(4, 4, func(_, x int) uint8 {
		if x == 3 {
			return 200
		}
		return 40
	})
	got, err := MedianThreshold(img)
	if err != nil {
		t.Fatal(err)
	}
	if got != 40 {
		t.Fatalf("MedianThreshold = %d, want 40", got)
	}
}

func TestIsoDataThreshold(t *testing.T) {
	// Equal masses at 50 and 200 converge to the midpoint 125.
	img := twoBands(8, 8, 50, 200)
	got, err := IsoDataThreshold(img)
	if err != nil {
		t.Fatal(err)
	}
	if got != 125 {
		t.Fatalf("IsoDataThreshold = %d, want 125", got)
	}
}

func TestPercentileThreshold(t *testing.T) {
	img := grayImage(4, 4, func(_, x int) uint8 {
		if x == 3 {
			return 200
		}
		return 40
	})
	got, err := PercentileThreshold(img, 0.5)
	if err != nil {
		t.Fatal(err)
	}
	if got != 40 {
		t.Fatalf("PercentileThreshold(0.5) = %d, want 40", got)
	}
	if _, err := PercentileThreshold(img, 1.5); err == nil {
		t.Fatal("expected error for out-of-range fraction")
	}
}

func TestOtsuSeparates(t *testing.T) {
	img := twoBands(8, 8, 50, 200)
	dst, thr, err := Otsu(img, ObjectBright)
	if err != nil {
		t.Fatal(err)
	}
	if thr < 50 || thr >= 200 {
		t.Fatalf("Otsu threshold = %d, want in [50,200)", thr)
	}
	for y := 0; y < 8; y++ {
		for x := 0; x < 8; x++ {
			want := uint8(0)
			if x >= 4 {
				want = 255
			}
			if got := dst.Data[y*8+x]; got != want {
				t.Fatalf("Otsu mask at (%d,%d) = %d, want %d", y, x, got, want)
			}
		}
	}
}

func TestPolarityInversion(t *testing.T) {
	img := twoBands(4, 4, 50, 200)
	bright, _ := Binarize(img, 100, ObjectBright)
	dark, _ := Binarize(img, 100, ObjectDark)
	for i := range bright.Data {
		if bright.Data[i] == dark.Data[i] {
			t.Fatalf("polarity did not invert at %d", i)
		}
	}
}

// clusteredImage places a dark cluster (60-62) in the left half and a bright
// cluster (188-190) in the right half.
func clusteredImage() *cv.Mat {
	return grayImage(6, 6, func(y, x int) uint8 {
		if x < 3 {
			return uint8(60 + (y % 3))
		}
		return uint8(188 + (y % 3))
	})
}

func TestSeparatingEstimators(t *testing.T) {
	img := clusteredImage()
	cases := []struct {
		name string
		fn   func(*cv.Mat) (int, error)
	}{
		{"Otsu", OtsuThreshold},
		{"Triangle", TriangleThreshold},
		{"Li", LiThreshold},
		{"Kapur", KapurThreshold},
		{"Yen", YenThreshold},
		{"Moments", MomentsThreshold},
		{"Kittler", KittlerThreshold},
		{"IsoData", IsoDataThreshold},
	}
	for _, c := range cases {
		got, err := c.fn(img)
		if err != nil {
			t.Fatalf("%s: %v", c.name, err)
		}
		if got < 62 || got > 188 {
			t.Fatalf("%s threshold = %d, want in [62,188]", c.name, got)
		}
	}
}

func TestMinimumIntermodes(t *testing.T) {
	img := clusteredImage()
	mn, err := MinimumThreshold(img)
	if err != nil {
		t.Fatal(err)
	}
	if mn < 62 || mn > 188 {
		t.Fatalf("MinimumThreshold = %d, want valley in [62,188]", mn)
	}
	im, err := IntermodesThreshold(img)
	if err != nil {
		t.Fatal(err)
	}
	if im < 62 || im > 188 {
		t.Fatalf("IntermodesThreshold = %d, want in [62,188]", im)
	}
}

func TestForegroundRatio(t *testing.T) {
	img := twoBands(8, 8, 0, 100)
	r, err := ForegroundRatio(img, 50, ObjectBright)
	if err != nil {
		t.Fatal(err)
	}
	if r != 0.5 {
		t.Fatalf("ForegroundRatio = %v, want 0.5", r)
	}
}

func TestAutoDispatch(t *testing.T) {
	img := twoBands(8, 8, 0, 100)
	for m := MethodOtsu; m <= MethodKittler; m++ {
		if m == MethodMinimum || m == MethodIntermodes {
			continue // require a bimodal-after-smoothing histogram
		}
		if _, err := AutoThreshold(img, m); err != nil {
			t.Fatalf("AutoThreshold(%s): %v", m, err)
		}
	}
	if _, _, err := Auto(img, MethodOtsu, ObjectBright); err != nil {
		t.Fatal(err)
	}
}

func TestEmptyErrors(t *testing.T) {
	var empty cv.Mat
	if _, err := OtsuThreshold(&empty); err != ErrEmpty {
		t.Fatalf("expected ErrEmpty, got %v", err)
	}
}
