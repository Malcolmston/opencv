package optflow

import (
	"image"
	"math"
	"testing"

	cv "github.com/malcolmston/opencv"
)

// texturedShift builds a pair of grayscale images where next is prev translated
// by (dx, dy). The pattern is a smooth, band-limited sinusoidal texture so that
// spatial gradients are well defined everywhere and the brightness-constancy
// linearisation used by Horn-Schunck holds for small motions. Sampling next at
// (x-dx, y-dy) realises a rightward/downward shift of (dx, dy).
func texturedShift(rows, cols int, dx, dy float64) (prev, next *cv.Mat) {
	prev = cv.NewMat(rows, cols, 1)
	next = cv.NewMat(rows, cols, 1)
	pattern := func(x, y float64) uint8 {
		val := 128 +
			60*math.Sin(2*math.Pi*x/16) +
			50*math.Cos(2*math.Pi*y/13) +
			20*math.Sin(2*math.Pi*(x+y)/9)
		if val < 0 {
			val = 0
		}
		if val > 255 {
			val = 255
		}
		return uint8(math.Round(val))
	}
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			prev.Set(y, x, 0, pattern(float64(x), float64(y)))
			next.Set(y, x, 0, pattern(float64(x)-dx, float64(y)-dy))
		}
	}
	return prev, next
}

func TestHornSchunckRecoversTranslation(t *testing.T) {
	const dx, dy = 1.0, 0.0
	prev, next := texturedShift(48, 48, dx, dy)
	flow := CalcOpticalFlowDenseHS(prev, next, 15.0, 200)
	mu, mv := flow.MeanFlow(6)
	if math.Abs(mu-dx) > 0.25 || math.Abs(mv-dy) > 0.25 {
		t.Fatalf("Horn-Schunck mean flow = (%.3f, %.3f), want ≈ (%.1f, %.1f)", mu, mv, dx, dy)
	}
}

func TestHornSchunckRecoversDiagonalTranslation(t *testing.T) {
	const dx, dy = 1.0, -1.0
	prev, next := texturedShift(48, 48, dx, dy)
	flow := CalcOpticalFlowDenseHS(prev, next, 20.0, 300)
	mu, mv := flow.MeanFlow(6)
	if math.Abs(mu-dx) > 0.3 || math.Abs(mv-dy) > 0.3 {
		t.Fatalf("Horn-Schunck mean flow = (%.3f, %.3f), want ≈ (%.1f, %.1f)", mu, mv, dx, dy)
	}
}

// patchShift places a deterministic textured block on a flat background, then
// produces next with the block moved by (dx, dy) integer pixels.
func patchShift(rows, cols, bx, by, bw, bh, dx, dy int) (prev, next *cv.Mat) {
	prev = cv.NewMat(rows, cols, 1)
	next = cv.NewMat(rows, cols, 1)
	prev.SetTo(40)
	next.SetTo(40)
	tex := func(x, y int) uint8 {
		return uint8((x*53 + y*97 + x*y*7) % 200)
	}
	for j := 0; j < bh; j++ {
		for i := 0; i < bw; i++ {
			v := tex(i, j)
			prev.Set(by+j, bx+i, 0, v)
			next.Set(by+j+dy, bx+i+dx, 0, v)
		}
	}
	return prev, next
}

func TestDISTracksShiftedPatch(t *testing.T) {
	const dx, dy = 3, 2
	prev, next := patchShift(64, 64, 20, 20, 20, 20, dx, dy)
	flow := CalcOpticalFlowDIS(prev, next, 4, 2, 3)

	// Average the flow over the interior of the shifted block, away from its
	// edges where the match is ambiguous.
	var su, sv float64
	var n int
	for y := 26; y < 34; y++ {
		for x := 26; x < 34; x++ {
			u, v := flow.At(y, x)
			su += u
			sv += v
			n++
		}
	}
	mu, mv := su/float64(n), sv/float64(n)
	if math.Abs(mu-dx) > 0.75 || math.Abs(mv-dy) > 0.75 {
		t.Fatalf("DIS mean patch flow = (%.3f, %.3f), want ≈ (%d, %d)", mu, mv, dx, dy)
	}
}

func TestFlowToColorProduces3Channel(t *testing.T) {
	flow := NewFlowField(20, 30)
	// Give the field some structure: radial outward flow.
	for y := 0; y < flow.Rows; y++ {
		for x := 0; x < flow.Cols; x++ {
			flow.set(y, x, float64(x-15), float64(y-10))
		}
	}
	img := FlowToColor(flow)
	if img.Channels != 3 {
		t.Fatalf("FlowToColor channels = %d, want 3", img.Channels)
	}
	if img.Rows != 20 || img.Cols != 30 {
		t.Fatalf("FlowToColor size = %dx%d, want 20x30", img.Rows, img.Cols)
	}
	// A non-trivial field must produce at least some non-white pixels.
	nonWhite := false
	for i := 0; i < len(img.Data); i++ {
		if img.Data[i] != 255 {
			nonWhite = true
			break
		}
	}
	if !nonWhite {
		t.Fatal("FlowToColor produced an all-white image for a non-zero field")
	}
}

func TestFlowToColorZeroFieldIsWhite(t *testing.T) {
	flow := NewFlowField(8, 8)
	img := FlowToColor(flow)
	for i := 0; i < len(img.Data); i++ {
		if img.Data[i] != 255 {
			t.Fatalf("zero flow pixel %d = %d, want 255 (white)", i, img.Data[i])
		}
	}
}

func TestWarpByFlowReconstructsShifted(t *testing.T) {
	const dx, dy = 2.0, 1.0
	prev, next := texturedShift(48, 48, dx, dy)
	// Exact flow from prev to next is a constant (dx, dy) everywhere.
	flow := NewFlowField(48, 48)
	for y := 0; y < 48; y++ {
		for x := 0; x < 48; x++ {
			flow.set(y, x, dx, dy)
		}
	}
	warped := WarpByFlow(prev, flow)

	// Compare warped to next over the interior (borders differ under replication).
	var sad float64
	var n int
	for y := 4; y < 44; y++ {
		for x := 4; x < 44; x++ {
			d := float64(warped.At(y, x, 0)) - float64(next.At(y, x, 0))
			sad += math.Abs(d)
			n++
		}
	}
	mad := sad / float64(n)
	if mad > 2.0 {
		t.Fatalf("WarpByFlow mean abs diff to next = %.3f, want < 2.0", mad)
	}
}

func TestWarpByFlowMultiChannel(t *testing.T) {
	img := cv.NewMat(16, 16, 3)
	for y := 0; y < 16; y++ {
		for x := 0; x < 16; x++ {
			img.Set(y, x, 0, uint8(x*8))
			img.Set(y, x, 1, uint8(y*8))
			img.Set(y, x, 2, uint8((x+y)*4))
		}
	}
	flow := NewFlowField(16, 16) // zero flow: identity warp
	warped := WarpByFlow(img, flow)
	for i := 0; i < len(img.Data); i++ {
		if warped.Data[i] != img.Data[i] {
			t.Fatalf("identity warp changed sample %d: %d != %d", i, warped.Data[i], img.Data[i])
		}
	}
}

func TestSparseToDenseRecoversTranslation(t *testing.T) {
	const dx, dy = 1.0, 0.0
	prev, next := texturedShift(48, 48, dx, dy)
	// Seed a grid of points across the textured region.
	var pts []image.Point
	for y := 6; y < 42; y += 6 {
		for x := 6; x < 42; x += 6 {
			pts = append(pts, image.Point{X: x, Y: y})
		}
	}
	flow := CalcOpticalFlowSparseToDense(prev, next, pts)
	mu, mv := flow.MeanFlow(6)
	if math.Abs(mu-dx) > 0.35 || math.Abs(mv-dy) > 0.35 {
		t.Fatalf("SparseToDense mean flow = (%.3f, %.3f), want ≈ (%.1f, %.1f)", mu, mv, dx, dy)
	}
}

func TestSparseToDenseAutoSeeds(t *testing.T) {
	prev, next := texturedShift(40, 40, 1.0, 0.0)
	flow := CalcOpticalFlowSparseToDense(prev, next, nil)
	if flow.Rows != 40 || flow.Cols != 40 {
		t.Fatalf("SparseToDense size = %dx%d, want 40x40", flow.Rows, flow.Cols)
	}
	mu, _ := flow.MeanFlow(6)
	if mu < 0.4 {
		t.Fatalf("SparseToDense auto-seed mean u = %.3f, want a clear rightward flow", mu)
	}
}

func TestFlowFieldMagnitude(t *testing.T) {
	flow := NewFlowField(2, 2)
	flow.set(0, 0, 3, 4) // magnitude 5
	flow.set(1, 1, 0, 0)
	mags := flow.Magnitude()
	if math.Abs(mags[0]-5) > 1e-9 {
		t.Fatalf("magnitude[0] = %.3f, want 5", mags[0])
	}
	if math.Abs(flow.MaxMagnitude()-5) > 1e-9 {
		t.Fatalf("MaxMagnitude = %.3f, want 5", flow.MaxMagnitude())
	}
}

func TestDeterminism(t *testing.T) {
	prev, next := texturedShift(32, 32, 1.0, 0.0)
	a := CalcOpticalFlowDenseHS(prev, next, 15, 50)
	b := CalcOpticalFlowDenseHS(prev, next, 15, 50)
	for i := range a.Data {
		if a.Data[i] != b.Data[i] {
			t.Fatalf("Horn-Schunck is non-deterministic at %d: %v != %v", i, a.Data[i], b.Data[i])
		}
	}
	c := CalcOpticalFlowDIS(prev, next, 4, 2, 2)
	d := CalcOpticalFlowDIS(prev, next, 4, 2, 2)
	for i := range c.Data {
		if c.Data[i] != d.Data[i] {
			t.Fatalf("DIS is non-deterministic at %d: %v != %v", i, c.Data[i], d.Data[i])
		}
	}
}
