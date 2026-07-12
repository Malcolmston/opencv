package mcc_test

import (
	"math"
	"testing"

	"github.com/malcolmston/opencv/mcc"
)

// sharmaPairs are reference test pairs from Sharma, Wu & Dalal (2005), "The
// CIEDE2000 Color-Difference Formula", each with its published dE00 value. A
// correct CIEDE2000 implementation must reproduce these exactly.
var sharmaPairs = []struct {
	lab1, lab2 [3]float64
	want       float64
}{
	{[3]float64{50.0000, 2.6772, -79.7751}, [3]float64{50.0000, 0.0000, -82.7485}, 2.0425},
	{[3]float64{50.0000, 3.1571, -77.2803}, [3]float64{50.0000, 0.0000, -82.7485}, 2.8615},
	{[3]float64{50.0000, 2.8361, -74.0200}, [3]float64{50.0000, 0.0000, -82.7485}, 3.4412},
	{[3]float64{50.0000, -1.3802, -84.2814}, [3]float64{50.0000, 0.0000, -82.7485}, 1.0000},
	{[3]float64{50.0000, -1.1848, -84.8006}, [3]float64{50.0000, 0.0000, -82.7485}, 1.0000},
	{[3]float64{50.0000, -0.9009, -85.5211}, [3]float64{50.0000, 0.0000, -82.7485}, 1.0000},
	{[3]float64{50.0000, 0.0000, 0.0000}, [3]float64{50.0000, -1.0000, 2.0000}, 2.3669},
	{[3]float64{50.0000, -1.0000, 2.0000}, [3]float64{50.0000, 0.0000, 0.0000}, 2.3669},
	{[3]float64{50.0000, 2.4900, -0.0010}, [3]float64{50.0000, -2.4900, 0.0009}, 7.1792},
	{[3]float64{50.0000, 2.4900, -0.0010}, [3]float64{50.0000, -2.4900, 0.0011}, 7.2195},
	{[3]float64{50.0000, -0.0010, 2.4900}, [3]float64{50.0000, 0.0009, -2.4900}, 4.8045},
	{[3]float64{60.2574, -34.0099, 36.2677}, [3]float64{60.4626, -34.1751, 39.4387}, 1.2644},
	{[3]float64{63.0109, -31.0961, -5.8663}, [3]float64{62.8187, -29.7946, -4.0864}, 1.2630},
	{[3]float64{35.0831, -44.1164, 3.7933}, [3]float64{35.0232, -40.0716, 1.5901}, 1.8645},
	{[3]float64{22.7233, 20.0904, -46.6940}, [3]float64{23.0331, 14.9730, -42.5619}, 2.0373},
	{[3]float64{2.0776, 0.0795, -1.1350}, [3]float64{0.9033, -0.0636, -0.5514}, 0.9082},
}

func TestDeltaE2000Sharma(t *testing.T) {
	for i, p := range sharmaPairs {
		got := mcc.DeltaE2000(p.lab1, p.lab2)
		if math.Abs(got-p.want) > 1e-4 {
			t.Errorf("pair %d: DeltaE2000=%.5f, want %.4f", i, got, p.want)
		}
	}
}

func TestDeltaE2000Symmetry(t *testing.T) {
	for i, p := range sharmaPairs {
		a := mcc.DeltaE2000(p.lab1, p.lab2)
		b := mcc.DeltaE2000(p.lab2, p.lab1)
		if math.Abs(a-b) > 1e-9 {
			t.Errorf("pair %d: asymmetric %.6f vs %.6f", i, a, b)
		}
	}
}

func TestDeltaE2000Identity(t *testing.T) {
	lab := [3]float64{42, 12, -30}
	if d := mcc.DeltaE2000(lab, lab); d != 0 {
		t.Errorf("DeltaE2000 of identical colors = %v, want 0", d)
	}
	if d := mcc.DeltaE2000Weighted(lab, lab, 2, 1, 1); d != 0 {
		t.Errorf("weighted DeltaE2000 of identical colors = %v, want 0", d)
	}
}

func TestDeltaE94(t *testing.T) {
	// For two neutral (a=b=0) colors chroma and hue differences vanish, so CIE94
	// reduces to |dL| / kL.
	g1 := [3]float64{60, 0, 0}
	g2 := [3]float64{50, 0, 0}
	if d := mcc.DeltaE94(g1, g2); math.Abs(d-10) > 1e-9 {
		t.Errorf("DeltaE94 neutral = %.6f, want 10", d)
	}
	if d := mcc.DeltaE94Textiles(g1, g2); math.Abs(d-5) > 1e-9 {
		t.Errorf("DeltaE94Textiles neutral = %.6f, want 5 (kL=2)", d)
	}
	if d := mcc.DeltaE94(g1, g1); d != 0 {
		t.Errorf("DeltaE94 identity = %v, want 0", d)
	}
	// CIE94 downweights chroma for saturated colors, so it should be <= CIE76.
	c1 := [3]float64{50, 60, 20}
	c2 := [3]float64{50, 50, 25}
	if mcc.DeltaE94(c1, c2) > mcc.DeltaE76(c1, c2)+1e-9 {
		t.Error("DeltaE94 should not exceed DeltaE76 for these saturated colors")
	}
}

func TestDeltaECMC(t *testing.T) {
	c1 := [3]float64{50, 60, 20}
	c2 := [3]float64{50, 50, 25}
	if d := mcc.DeltaECMC(c1, c1, 2, 1); d != 0 {
		t.Errorf("CMC identity = %v, want 0", d)
	}
	acc := mcc.DeltaECMCAcceptability(c1, c2)
	per := mcc.DeltaECMCPerceptibility(c1, c2)
	if acc <= 0 || per <= 0 {
		t.Errorf("CMC differences should be positive: acc=%.4f per=%.4f", acc, per)
	}
	// Acceptability loosens the lightness tolerance (l=2), so for a difference
	// with a lightness component it should not exceed perceptibility (l=1).
	d1 := [3]float64{40, 10, 10}
	d2 := [3]float64{55, 12, 8}
	if mcc.DeltaECMCAcceptability(d1, d2) > mcc.DeltaECMCPerceptibility(d1, d2)+1e-9 {
		t.Error("CMC(2:1) should be <= CMC(1:1) when a lightness difference is present")
	}
	// CMC is asymmetric in general.
	if mcc.DeltaECMC(c1, c2, 2, 1) == mcc.DeltaECMC(c2, c1, 2, 1) {
		// Not a hard failure everywhere, but for these colors it must differ.
		t.Error("CMC should be asymmetric for these colors")
	}
}

func TestDeltaE2000RGB(t *testing.T) {
	if d := mcc.DeltaE2000RGB([3]uint8{100, 120, 130}, [3]uint8{100, 120, 130}); d != 0 {
		t.Errorf("DeltaE2000RGB identity = %v, want 0", d)
	}
	if d := mcc.DeltaE2000RGB([3]uint8{255, 0, 0}, [3]uint8{0, 255, 0}); d < 50 {
		t.Errorf("DeltaE2000RGB red vs green = %.2f, want large", d)
	}
}
