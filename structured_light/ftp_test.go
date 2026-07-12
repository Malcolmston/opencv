package structured_light

import (
	"math"
	"testing"

	cv "github.com/malcolmston/opencv"
)

// makeFringe builds a vertical-fringe image I(x)=127.5(1+cos(2π f x/W + φ0)).
func makeFringe(rows, cols, freq int, phi0 float64) *cv.Mat {
	m := cv.NewMat(rows, cols, 1)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			phi := 2*math.Pi*float64(freq)*float64(x)/float64(cols) + phi0
			m.Set(y, x, 0, clampRound(127.5*(1+math.Cos(phi))))
		}
	}
	return m
}

func TestFTPDecodesKnownFringe(t *testing.T) {
	rows, cols, freq := 4, 64, 6
	phi0 := 0.7
	img := makeFringe(rows, cols, freq, phi0)

	wrapped := FTPWrappedPhase(img, false)
	// The recovered wrapped phase should equal wrap(2π f x/W + φ0 + C) for some
	// constant carrier phase C. Unwrap along x and fit against the known ramp
	// after removing the per-row offset at x=0.
	abs := UnwrapPhaseMap(wrapped, rows, cols, false)
	maxErr := 0.0
	for y := 0; y < rows; y++ {
		base := abs[y*cols]
		for x := 0; x < cols; x++ {
			want := 2 * math.Pi * float64(freq) * float64(x) / float64(cols)
			got := abs[y*cols+x] - base
			if e := math.Abs(got - want); e > maxErr {
				maxErr = e
			}
		}
	}
	if maxErr > 0.05 {
		t.Fatalf("FTP phase max error %.4f exceeds tolerance", maxErr)
	}
}

func TestFTPHorizontalAndExplicitBand(t *testing.T) {
	rows, cols, freq := 48, 3, 4
	// Horizontal fringes: transpose the construction (phase varies with y).
	img := cv.NewMat(rows, cols, 1)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			phi := 2 * math.Pi * float64(freq) * float64(y) / float64(rows)
			img.Set(y, x, 0, clampRound(127.5*(1+math.Cos(phi))))
		}
	}
	wrapped := FTPWrappedPhaseBand(img, true, freq, 2)
	abs := UnwrapPhaseMap(wrapped, rows, cols, true)
	maxErr := 0.0
	for x := 0; x < cols; x++ {
		base := abs[x]
		for y := 0; y < rows; y++ {
			want := 2 * math.Pi * float64(freq) * float64(y) / float64(rows)
			got := abs[y*cols+x] - base
			if e := math.Abs(got - want); e > maxErr {
				maxErr = e
			}
		}
	}
	if maxErr > 0.05 {
		t.Fatalf("horizontal FTP phase max error %.4f exceeds tolerance", maxErr)
	}
}

func TestDFTRoundTrip(t *testing.T) {
	x := make([]complex128, 16)
	for i := range x {
		x[i] = complex(math.Sin(float64(i))+0.3*float64(i), 0)
	}
	back := idft(dft(x))
	for i := range x {
		if d := math.Hypot(real(back[i]-x[i]), imag(back[i]-x[i])); d > 1e-9 {
			t.Fatalf("idft(dft(x)) differs at %d by %.3e", i, d)
		}
	}
}

func TestDetectCarrier(t *testing.T) {
	// A pure 5-cycle cosine over 40 samples must peak at bin 5.
	n := 40
	sig := make([]float64, n)
	for i := range sig {
		sig[i] = math.Cos(2 * math.Pi * 5 * float64(i) / float64(n))
	}
	if c := detectCarrier(sig); c != 5 {
		t.Fatalf("detectCarrier = %d, want 5", c)
	}
}
