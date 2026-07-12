package mcc_test

import (
	"math"
	"testing"

	"github.com/malcolmston/opencv/mcc"
)

func close3(a, b [3]float64, tol float64) bool {
	return math.Abs(a[0]-b[0]) <= tol && math.Abs(a[1]-b[1]) <= tol && math.Abs(a[2]-b[2]) <= tol
}

func TestLabLChRoundTrip(t *testing.T) {
	labs := [][3]float64{
		{50, 20, -30}, {70, -40, 15}, {30, 0, 0}, {88, 3, 60}, {12, -5, -5},
	}
	for _, lab := range labs {
		lch := mcc.LabToLCh(lab)
		back := mcc.LChToLab(lch)
		if !close3(lab, back, 1e-9) {
			t.Errorf("Lab<->LCh round trip %v -> %v -> %v", lab, lch, back)
		}
		// Chroma is nonnegative; hue in [0,360).
		if lch[1] < 0 || lch[2] < 0 || lch[2] >= 360 {
			t.Errorf("LCh out of range: %v", lch)
		}
	}
	// Known hue: pure +a* axis is hue 0, pure +b* axis is hue 90.
	if h := mcc.LabToLCh([3]float64{50, 10, 0})[2]; math.Abs(h) > 1e-9 {
		t.Errorf("hue of +a* = %.4f, want 0", h)
	}
	if h := mcc.LabToLCh([3]float64{50, 0, 10})[2]; math.Abs(h-90) > 1e-9 {
		t.Errorf("hue of +b* = %.4f, want 90", h)
	}
}

func TestXYZxyYRoundTrip(t *testing.T) {
	xyzs := [][3]float64{
		mcc.WhiteD65, mcc.WhiteD50, {0.2, 0.15, 0.4}, {0.5, 0.5, 0.5},
	}
	for _, xyz := range xyzs {
		xyY := mcc.XYZToxyY(xyz)
		back := mcc.XYYToXYZ(xyY)
		if !close3(xyz, back, 1e-12) {
			t.Errorf("XYZ<->xyY round trip %v -> %v -> %v", xyz, xyY, back)
		}
	}
	// D65 chromaticity is approximately (0.3127, 0.3290).
	xy := mcc.XYZToxyY(mcc.WhiteD65)
	if math.Abs(xy[0]-0.3127) > 1e-3 || math.Abs(xy[1]-0.3290) > 1e-3 {
		t.Errorf("D65 chromaticity = (%.4f,%.4f), want ~(0.3127,0.3290)", xy[0], xy[1])
	}
	// Black is degenerate.
	if got := mcc.XYZToxyY([3]float64{0, 0, 0}); got != [3]float64{0, 0, 0} {
		t.Errorf("black xyY = %v, want zero", got)
	}
}

func TestXYZLabRoundTripWhitePoints(t *testing.T) {
	whites := [][3]float64{mcc.WhiteD65, mcc.WhiteD50, mcc.WhiteA}
	xyzs := [][3]float64{{0.3, 0.32, 0.28}, {0.05, 0.04, 0.03}, {0.7, 0.75, 0.6}}
	for _, w := range whites {
		// The white point itself maps to L*=100, a*=b*=0.
		lab := mcc.XYZToLab(w, w)
		if math.Abs(lab[0]-100) > 1e-9 || math.Abs(lab[1]) > 1e-9 || math.Abs(lab[2]) > 1e-9 {
			t.Errorf("white %v -> Lab %v, want (100,0,0)", w, lab)
		}
		for _, xyz := range xyzs {
			back := mcc.LabToXYZ(mcc.XYZToLab(xyz, w), w)
			if !close3(xyz, back, 1e-9) {
				t.Errorf("XYZ<->Lab round trip under %v: %v -> %v", w, xyz, back)
			}
		}
	}
}

func TestLinearRGBXYZConsistency(t *testing.T) {
	// LinearRGBToXYZ of the linearised sRGB white should match D65 (Y=1).
	lw := mcc.SRGBToLinear(1)
	xyz := mcc.LinearRGBToXYZ(lw, lw, lw)
	if math.Abs(xyz[1]-1) > 1e-6 {
		t.Errorf("white Y=%.6f, want 1", xyz[1])
	}
	// Compare against the package's uint8 RGBToXYZ entry point.
	ref := mcc.RGBToXYZ(255, 255, 255)
	if !close3(xyz, ref, 1e-6) {
		t.Errorf("LinearRGBToXYZ white %v vs RGBToXYZ %v", xyz, ref)
	}
}

func TestXYZRGBRoundTrip(t *testing.T) {
	// XYZToRGB should invert RGBToXYZ for in-gamut colors within rounding.
	for _, rgb := range [][3]uint8{{200, 100, 50}, {30, 60, 90}, {128, 128, 128}, {255, 255, 255}} {
		xyz := mcc.RGBToXYZ(rgb[0], rgb[1], rgb[2])
		back := mcc.XYZToRGB(xyz)
		for c := 0; c < 3; c++ {
			if d := int(back[c]) - int(rgb[c]); d < -1 || d > 1 {
				t.Errorf("XYZ<->RGB round trip %v -> %v (channel %d off by %d)", rgb, back, c, d)
			}
		}
	}
}

func TestLabToRGBNonD65(t *testing.T) {
	// A neutral Lab under D50 should render to a near-neutral sRGB (equal
	// channels) after adaptation to D65.
	rgb := mcc.LabToRGB([3]float64{100, 0, 0}, mcc.WhiteD50)
	if maxAbsDiff(rgb) > 3 {
		t.Errorf("D50 white -> sRGB %v, want near-neutral", rgb)
	}
	// And it should be bright.
	if rgb[0] < 240 {
		t.Errorf("D50 white -> sRGB %v, want bright", rgb)
	}
}

func maxAbsDiff(rgb [3]uint8) int {
	mn, mx := int(rgb[0]), int(rgb[0])
	for _, v := range rgb {
		if int(v) < mn {
			mn = int(v)
		}
		if int(v) > mx {
			mx = int(v)
		}
	}
	return mx - mn
}

func TestGammaRoundTrip(t *testing.T) {
	for _, c := range []float64{0, 0.1, 0.5, 0.9, 1} {
		if got := mcc.GammaCompress(mcc.GammaExpand(c, 2.2), 2.2); math.Abs(got-c) > 1e-12 {
			t.Errorf("gamma round trip %.3f -> %.12f", c, got)
		}
	}
	// Expanding with gamma 1 is the identity.
	if mcc.GammaExpand(0.42, 1) != 0.42 {
		t.Error("GammaExpand with gamma 1 should be identity")
	}
}

func TestChromaticAdaptationRoundTrip(t *testing.T) {
	methods := []mcc.AdaptationMethod{mcc.Bradford, mcc.VonKries, mcc.XYZScaling}
	whites := [][3]float64{mcc.WhiteD65, mcc.WhiteD50, mcc.WhiteA}
	colors := [][3]float64{{0.3, 0.32, 0.28}, {0.1, 0.05, 0.2}, {0.6, 0.65, 0.7}}
	for _, m := range methods {
		for _, src := range whites {
			for _, dst := range whites {
				for _, xyz := range colors {
					adapted := mcc.ChromaticAdaptation(xyz, src, dst, m)
					back := mcc.ChromaticAdaptation(adapted, dst, src, m)
					if !close3(xyz, back, 1e-9) {
						t.Errorf("method %d %v->%v round trip %v -> %v", m, src, dst, xyz, back)
					}
				}
			}
		}
	}
}

func TestAdaptationWhiteMapsToWhite(t *testing.T) {
	// Adapting the source white to a destination illuminant must produce the
	// destination white exactly (the defining property of the transform).
	for _, m := range []mcc.AdaptationMethod{mcc.Bradford, mcc.VonKries, mcc.XYZScaling} {
		got := mcc.ChromaticAdaptation(mcc.WhiteD65, mcc.WhiteD65, mcc.WhiteA, m)
		if !close3(got, mcc.WhiteA, 1e-9) {
			t.Errorf("method %d: D65 white -> %v, want WhiteA %v", m, got, mcc.WhiteA)
		}
	}
	// Same-illuminant adaptation is the identity.
	id := mcc.AdaptationMatrix(mcc.WhiteD50, mcc.WhiteD50, mcc.Bradford)
	got := mcc.ApplyMatrix3(id, [3]float64{0.4, 0.5, 0.6})
	if !close3(got, [3]float64{0.4, 0.5, 0.6}, 1e-12) {
		t.Errorf("identity adaptation changed color: %v", got)
	}
}
