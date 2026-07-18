package tracking

import "testing"

func TestHornSchunckDirection(t *testing.T) {
	prev := synthTexture(40, 40, 0, 0)
	next := synthTexture(40, 40, 1, 0) // small +x motion
	flow := CalcOpticalFlowHornSchunck(prev, next, 5, 128)
	// Average the interior flow to avoid boundary artefacts.
	var su, sv, n float64
	for y := 8; y < 32; y++ {
		for x := 8; x < 32; x++ {
			f := flow.At(y, x)
			su += f.X
			sv += f.Y
			n++
		}
	}
	mu := su / n
	mv := sv / n
	requireTrue(t, mu > 0.3, "mean u = %v, want clearly positive", mu)
	requireTrue(t, mu < 1.6, "mean u = %v, want ~1", mu)
	requireTrue(t, mv > -0.35 && mv < 0.35, "mean v = %v, want ~0", mv)
}

func TestFarnebackTranslation(t *testing.T) {
	prev := synthTexture(48, 48, 0, 0)
	next := synthTexture(48, 48, 2, 1)
	flow := CalcOpticalFlowFarneback(prev, next, DefaultFarnebackParams())
	// Check a central patch recovers the (2, 1) shift.
	var su, sv, n float64
	for y := 16; y < 32; y++ {
		for x := 16; x < 32; x++ {
			f := flow.At(y, x)
			su += f.X
			sv += f.Y
			n++
		}
	}
	mu := su / n
	mv := sv / n
	requireTrue(t, approx(mu, 2.0, 0.6), "mean u = %v, want ~2", mu)
	requireTrue(t, approx(mv, 1.0, 0.6), "mean v = %v, want ~1", mv)
}

func TestFarnebackZeroMotion(t *testing.T) {
	img := synthTexture(32, 32, 0, 0)
	flow := CalcOpticalFlowFarneback(img, img, DefaultFarnebackParams())
	requireTrue(t, flow.MaxMagnitude() < 0.5, "static scene should have ~0 flow, got %v", flow.MaxMagnitude())
}
