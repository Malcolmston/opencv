package colorspaces2

import (
	"math"
	"testing"
)

// approx reports whether a and b are within tol.
func approx(a, b, tol float64) bool { return math.Abs(a-b) <= tol }

func rgbApprox(a, b RGB, tol float64) bool {
	return approx(a.R, b.R, tol) && approx(a.G, b.G, tol) && approx(a.B, b.B, tol)
}

func TestRGBToHSVKnown(t *testing.T) {
	cases := []struct {
		in  RGB
		out HSV
	}{
		{RGB{1, 0, 0}, HSV{0, 1, 1}},
		{RGB{0, 1, 0}, HSV{120, 1, 1}},
		{RGB{0, 0, 1}, HSV{240, 1, 1}},
		{RGB{1, 1, 1}, HSV{0, 0, 1}},
		{RGB{0, 0, 0}, HSV{0, 0, 0}},
		{RGB{0.5, 0.5, 0.5}, HSV{0, 0, 0.5}},
	}
	for _, c := range cases {
		got := RGBToHSV(c.in)
		if !approx(got.H, c.out.H, 1e-9) || !approx(got.S, c.out.S, 1e-9) || !approx(got.V, c.out.V, 1e-9) {
			t.Errorf("RGBToHSV(%+v)=%+v want %+v", c.in, got, c.out)
		}
	}
}

func TestHSVRoundTrip(t *testing.T) {
	for _, c := range []RGB{{0.1, 0.5, 0.9}, {0.3, 0.7, 0.2}, {0.9, 0.1, 0.4}, {0, 0.5, 1}} {
		if got := HSVToRGB(RGBToHSV(c)); !rgbApprox(got, c, 1e-9) {
			t.Errorf("HSV round trip %+v -> %+v", c, got)
		}
	}
}

func TestHSLRoundTrip(t *testing.T) {
	for _, c := range []RGB{{0.1, 0.5, 0.9}, {0.3, 0.7, 0.2}, {0.9, 0.1, 0.4}} {
		if got := HSLToRGB(RGBToHSL(c)); !rgbApprox(got, c, 1e-9) {
			t.Errorf("HSL round trip %+v -> %+v", c, got)
		}
	}
}

func TestRGBToHSLKnown(t *testing.T) {
	got := RGBToHSL(RGB{1, 0, 0})
	if !approx(got.H, 0, 1e-9) || !approx(got.S, 1, 1e-9) || !approx(got.L, 0.5, 1e-9) {
		t.Errorf("red HSL = %+v", got)
	}
}

func TestRGBToXYZWhite(t *testing.T) {
	got := RGBToXYZ(RGB{1, 1, 1})
	if !approx(got.X, WhitePointD65.X, 1e-4) || !approx(got.Y, 1, 1e-4) || !approx(got.Z, WhitePointD65.Z, 1e-4) {
		t.Errorf("white XYZ = %+v, want D65 %+v", got, WhitePointD65)
	}
}

func TestXYZRoundTrip(t *testing.T) {
	for _, c := range []RGB{{0.2, 0.4, 0.6}, {0.8, 0.1, 0.5}, {1, 1, 1}, {0, 0, 0}} {
		if got := XYZToRGB(RGBToXYZ(c)); !rgbApprox(got, c, 1e-6) {
			t.Errorf("XYZ round trip %+v -> %+v", c, got)
		}
	}
}

func TestRGBToLabKnown(t *testing.T) {
	// Canonical CIE L*a*b* value of pure sRGB red under D65.
	got := RGBToLab(RGB{1, 0, 0})
	if !approx(got.L, 53.2408, 1e-3) || !approx(got.A, 80.0925, 1e-3) || !approx(got.B, 67.2032, 1e-3) {
		t.Errorf("red Lab = %+v", got)
	}
	// White must map to L=100, a=b=0.
	w := RGBToLab(RGB{1, 1, 1})
	if !approx(w.L, 100, 1e-3) || !approx(w.A, 0, 1e-3) || !approx(w.B, 0, 1e-3) {
		t.Errorf("white Lab = %+v", w)
	}
}

func TestLabRoundTrip(t *testing.T) {
	for _, c := range []RGB{{0.1, 0.5, 0.9}, {0.3, 0.7, 0.2}, {0.6, 0.6, 0.6}} {
		if got := LabToRGB(RGBToLab(c)); !rgbApprox(got, c, 1e-5) {
			t.Errorf("Lab round trip %+v -> %+v", c, got)
		}
	}
}

func TestLuvRoundTrip(t *testing.T) {
	for _, c := range []RGB{{0.1, 0.5, 0.9}, {0.3, 0.7, 0.2}, {0.6, 0.6, 0.6}, {0, 0, 0}} {
		if got := LuvToRGB(RGBToLuv(c)); !rgbApprox(got, c, 1e-5) {
			t.Errorf("Luv round trip %+v -> %+v", c, got)
		}
	}
}

func TestRGBToLuvWhite(t *testing.T) {
	w := RGBToLuv(RGB{1, 1, 1})
	if !approx(w.L, 100, 1e-3) || !approx(w.U, 0, 1e-3) || !approx(w.V, 0, 1e-3) {
		t.Errorf("white Luv = %+v", w)
	}
}

func TestRGBToYCbCrKnown(t *testing.T) {
	got := RGBToYCbCr(RGB{1, 0, 0})
	if !approx(got.Y, 0.299, 1e-6) || !approx(got.Cb, -0.168736, 1e-6) || !approx(got.Cr, 0.5, 1e-6) {
		t.Errorf("red YCbCr = %+v", got)
	}
	// Gray has zero chroma.
	g := RGBToYCbCr(RGB{0.5, 0.5, 0.5})
	if !approx(g.Cb, 0, 1e-9) || !approx(g.Cr, 0, 1e-9) {
		t.Errorf("gray YCbCr chroma = %+v", g)
	}
}

func TestYCbCrRoundTrip(t *testing.T) {
	for _, c := range []RGB{{0.1, 0.5, 0.9}, {0.3, 0.7, 0.2}} {
		if got := YCbCrToRGB(RGBToYCbCr(c)); !rgbApprox(got, c, 1e-6) {
			t.Errorf("YCbCr round trip %+v -> %+v", c, got)
		}
	}
}

func TestYUVRoundTrip(t *testing.T) {
	for _, c := range []RGB{{0.1, 0.5, 0.9}, {0.3, 0.7, 0.2}} {
		// The BT.601 inverse coefficients are rounded constants, so the round
		// trip is only good to a few thousandths.
		if got := YUVToRGB(RGBToYUV(c)); !rgbApprox(got, c, 2e-3) {
			t.Errorf("YUV round trip %+v -> %+v", c, got)
		}
	}
}

func TestCMYKKnown(t *testing.T) {
	got := RGBToCMYK(RGB{0.2, 0.4, 0.6})
	want := CMYK{C: 2.0 / 3.0, M: 1.0 / 3.0, Y: 0, K: 0.4}
	if !approx(got.C, want.C, 1e-9) || !approx(got.M, want.M, 1e-9) ||
		!approx(got.Y, want.Y, 1e-9) || !approx(got.K, want.K, 1e-9) {
		t.Errorf("CMYK = %+v want %+v", got, want)
	}
	if got := RGBToCMYK(RGB{0, 0, 0}); got.K != 1 {
		t.Errorf("black CMYK K = %v want 1", got.K)
	}
}

func TestCMYKRoundTrip(t *testing.T) {
	for _, c := range []RGB{{0.2, 0.4, 0.6}, {0.9, 0.1, 0.5}, {1, 1, 1}, {0, 0, 0}} {
		if got := CMYKToRGB(RGBToCMYK(c)); !rgbApprox(got, c, 1e-9) {
			t.Errorf("CMYK round trip %+v -> %+v", c, got)
		}
	}
}

func TestRGBUint8Helpers(t *testing.T) {
	c := NewRGBFromUint8(255, 128, 0)
	r, g, b := c.ToUint8()
	if r != 255 || g != 128 || b != 0 {
		t.Errorf("uint8 round trip = %d,%d,%d", r, g, b)
	}
	// Clamp of an out-of-gamut value.
	cl := RGB{1.5, -0.2, 0.5}.Clamp()
	if cl.R != 1 || cl.G != 0 || cl.B != 0.5 {
		t.Errorf("clamp = %+v", cl)
	}
	if !approx(RGB{1, 1, 1}.Luma(), 1, 1e-9) {
		t.Errorf("luma of white != 1")
	}
}
