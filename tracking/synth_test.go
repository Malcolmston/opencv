package tracking

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// grayFrame renders a w×h single-channel frame with a dark background and a
// textured bright patch of half-size half centred at (cx, cy). The patch carries
// five bright feature squares placed at fixed fractions of half, so the texture
// gives Lucas-Kanade well-conditioned corners and moves/scales rigidly with the
// patch. It is deterministic.
func grayFrame(w, h, cx, cy, half int) *cv.Mat {
	m := cv.NewMat(h, w, 1)
	for i := range m.Data {
		m.Data[i] = 30
	}
	set := func(x, y int, v uint8) {
		if x >= 0 && x < w && y >= 0 && y < h {
			m.Data[y*w+x] = v
		}
	}
	// Patch base.
	for y := cy - half; y <= cy+half; y++ {
		for x := cx - half; x <= cx+half; x++ {
			set(x, y, 90)
		}
	}
	// Feature squares at asymmetric fractional offsets of half.
	feats := []struct {
		fx, fy float64
		val    uint8
	}{
		{-0.55, -0.55, 220},
		{0.55, -0.55, 180},
		{-0.55, 0.55, 200},
		{0.55, 0.55, 240},
		{0.05, -0.15, 150},
	}
	fs := half / 3
	if fs < 2 {
		fs = 2
	}
	for _, f := range feats {
		fcx := cx + int(math.Round(f.fx*float64(half)))
		fcy := cy + int(math.Round(f.fy*float64(half)))
		for y := fcy - fs; y <= fcy+fs; y++ {
			for x := fcx - fs; x <= fcx+fs; x++ {
				set(x, y, f.val)
			}
		}
	}
	return m
}

// colorFrame renders a w×h RGB frame with a saturated green background and a
// solid red square (half-size half) centred at (cx, cy). Red and green have
// well-separated hues, so a hue back-projection of the square isolates it from
// the background. It is deterministic.
func colorFrame(w, h, cx, cy, half int) *cv.Mat {
	m := cv.NewMat(h, w, 3)
	for p := 0; p < m.Total(); p++ {
		m.Data[p*3+0] = 0
		m.Data[p*3+1] = 255
		m.Data[p*3+2] = 0
	}
	for y := cy - half; y <= cy+half; y++ {
		for x := cx - half; x <= cx+half; x++ {
			if x >= 0 && x < w && y >= 0 && y < h {
				i := (y*w + x) * 3
				m.Data[i+0] = 255
				m.Data[i+1] = 0
				m.Data[i+2] = 0
			}
		}
	}
	return m
}

// blobProb renders a w×h single-channel probability image with an anisotropic
// Gaussian blob (standard deviations sx, sy) centred at (cx, cy), scaled so the
// peak is 255. It stands in for a back-projection map. It is deterministic.
func blobProb(w, h, cx, cy int, sx, sy float64) *cv.Mat {
	m := cv.NewMat(h, w, 1)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			dx := float64(x - cx)
			dy := float64(y - cy)
			v := 255 * math.Exp(-(dx*dx/(2*sx*sx) + dy*dy/(2*sy*sy)))
			m.Data[y*w+x] = uint8(math.Round(v))
		}
	}
	return m
}

// boxCenter returns the fractional centre of a box.
func boxCenter(r cv.Rect) (float64, float64) {
	return float64(r.X) + float64(r.Width)/2, float64(r.Y) + float64(r.Height)/2
}
