package cudaoptflow

import (
	"math"
	"testing"

	cv "github.com/malcolmston/opencv"
	"github.com/malcolmston/opencv/optflow"
)

// basePattern builds a deterministic, broadband, locally-unique grayscale base
// image large enough to be sampled with a margin, so shifted crops contain no
// wrap-around. It is hashed value noise lightly smoothed so gradients are
// meaningful; unlike a sum of sinusoids it is not periodic, which keeps
// coarse-to-fine and block-matching estimators from locking onto the wrong phase.
func basePattern(rows, cols int) [][]float64 {
	b := make([][]float64, rows)
	for r := 0; r < rows; r++ {
		b[r] = make([]float64, cols)
		for c := 0; c < cols; c++ {
			h := uint32(r)*73856093 ^ uint32(c)*19349663 ^ 0x9e3779b9
			h *= 2654435761
			h ^= h >> 15
			b[r][c] = float64(h & 0xff)
		}
	}
	// Box-blur passes give smooth, differentiable texture; then rescale to the
	// full 0..255 range so gradients stay strong (high contrast) for tracking.
	for pass := 0; pass < 2; pass++ {
		b = boxBlur3(b)
	}
	lo, hi := b[0][0], b[0][0]
	for r := range b {
		for c := range b[r] {
			if b[r][c] < lo {
				lo = b[r][c]
			}
			if b[r][c] > hi {
				hi = b[r][c]
			}
		}
	}
	if hi > lo {
		for r := range b {
			for c := range b[r] {
				b[r][c] = (b[r][c] - lo) / (hi - lo) * 255
			}
		}
	}
	return b
}

func boxBlur3(in [][]float64) [][]float64 {
	rows, cols := len(in), len(in[0])
	out := make([][]float64, rows)
	for r := 0; r < rows; r++ {
		out[r] = make([]float64, cols)
		for c := 0; c < cols; c++ {
			var sum float64
			var n int
			for dr := -1; dr <= 1; dr++ {
				for dc := -1; dc <= 1; dc++ {
					rr, cc := r+dr, c+dc
					if rr < 0 || rr >= rows || cc < 0 || cc >= cols {
						continue
					}
					sum += in[rr][cc]
					n++
				}
			}
			out[r][c] = sum / float64(n)
		}
	}
	return out
}

// shiftedFrames returns a prev/next grayscale pair of size h x w where the scene
// content is translated by (dx, dy) pixels, so the true optical flow from prev
// to next is (dx, dy) everywhere. A margin keeps every sampled pixel inside the
// base pattern.
func shiftedFrames(h, w, dx, dy int) (prev, next *cv.Mat) {
	const margin = 12
	base := basePattern(h+2*margin, w+2*margin)
	prev = cv.NewMat(h, w, 1)
	next = cv.NewMat(h, w, 1)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			prev.Set(y, x, 0, uint8(base[y+margin][x+margin]+0.5))
			next.Set(y, x, 0, uint8(base[y+margin-dy][x+margin-dx]+0.5))
		}
	}
	return prev, next
}

func denseGpuPair(h, w, dx, dy int) (*GpuMat, *GpuMat) {
	p, n := shiftedFrames(h, w, dx, dy)
	return GpuMatFromMat(p), GpuMatFromMat(n)
}

// assertMeanFlow checks that the interior mean of a flow field is within tol of
// the expected translation.
func assertMeanFlow(t *testing.T, name string, f *optflow.FlowField, wantU, wantV, tol float64) {
	t.Helper()
	u, v := f.MeanFlow(8)
	if math.Abs(u-wantU) > tol || math.Abs(v-wantV) > tol {
		t.Errorf("%s: mean flow = (%.3f, %.3f), want (%.1f, %.1f) +/- %.2f", name, u, v, wantU, wantV, tol)
	}
}

func TestFarnebackRecoversTranslation(t *testing.T) {
	prev, next := denseGpuPair(64, 64, 3, 2)
	o := NewFarnebackOpticalFlow(4, 4)
	f := o.Calc(prev, next, NewStream())
	assertMeanFlow(t, "Farneback", f, 3, 2, 0.6)
}

func TestDualTVL1RecoversTranslation(t *testing.T) {
	prev, next := denseGpuPair(64, 64, 3, 2)
	o := NewOpticalFlowDualTVL1()
	f := o.Calc(prev, next, nil)
	assertMeanFlow(t, "TVL1", f, 3, 2, 0.75)
}

func TestNvidiaHWRecoversTranslation(t *testing.T) {
	prev, next := denseGpuPair(64, 64, 3, 2)
	o := NewNvidiaHWOpticalFlow(4, 2, 3)
	f := o.Calc(prev, next, nil)
	assertMeanFlow(t, "NvidiaHW/DIS", f, 3, 2, 0.75)
}

func TestDensePyrLKRecoversTranslation(t *testing.T) {
	prev, next := denseGpuPair(64, 64, 3, 2)
	o := NewDensePyrLKOpticalFlow(13, 3, 30)
	f := o.Calc(prev, next, nil)
	assertMeanFlow(t, "DensePyrLK", f, 3, 2, 0.9)
}

func TestBroxRecoversTranslation(t *testing.T) {
	prev, next := denseGpuPair(64, 64, 3, 2)
	o := NewBroxOpticalFlow(0.2, 50, 0.5, 3, 12, 20)
	f := o.Calc(prev, next, nil)
	assertMeanFlow(t, "Brox", f, 3, 2, 0.6)
}

func TestBroxBrightnessOnly(t *testing.T) {
	prev, next := denseGpuPair(48, 48, 2, 1)
	o := NewBroxOpticalFlow(0.05, 0, 0.5, 5, 20, 30)
	f := o.Calc(prev, next, nil)
	assertMeanFlow(t, "Brox(gamma=0)", f, 2, 1, 0.6)
}

func TestDefaultBroxRecoversTranslation(t *testing.T) {
	prev, next := denseGpuPair(48, 48, 2, 2)
	o := DefaultBroxOpticalFlow()
	f := o.Calc(prev, next, nil)
	assertMeanFlow(t, "DefaultBrox", f, 2, 2, 0.7)
}

func TestSparsePyrLKRecoversTranslation(t *testing.T) {
	prev, next := denseGpuPair(80, 80, 3, 2)
	pts := []cv.Point{{X: 25, Y: 25}, {X: 52, Y: 28}, {X: 30, Y: 52}, {X: 50, Y: 50}}
	o := NewSparsePyrLKOpticalFlow(15, 2, 30)
	nextPts, status, errs := o.Calc(prev, next, pts, NewStream())
	if len(nextPts) != len(pts) || len(status) != len(pts) || len(errs) != len(pts) {
		t.Fatalf("parallel slice lengths wrong: %d %d %d", len(nextPts), len(status), len(errs))
	}
	for i, p := range pts {
		if status[i] != 1 {
			t.Errorf("point %d lost (status=%d)", i, status[i])
			continue
		}
		wantX, wantY := p.X+3, p.Y+2
		if abs(nextPts[i].X-wantX) > 1 || abs(nextPts[i].Y-wantY) > 1 {
			t.Errorf("point %d tracked to (%d,%d), want (%d,%d)", i, nextPts[i].X, nextPts[i].Y, wantX, wantY)
		}
	}
}

func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}

func TestFlowGpuMatRoundTrip(t *testing.T) {
	f := optflow.NewFlowField(6, 5)
	for y := 0; y < 6; y++ {
		for x := 0; x < 5; x++ {
			f.Set(y, x, float64(x)*0.25-1.0, float64(y)*0.5-1.5)
		}
	}
	g := FlowToGpuMat(f)
	if r, c := g.Size(); r != 6 || c != 5 {
		t.Fatalf("encoded size = (%d,%d), want (6,5)", r, c)
	}
	if g.Channels() != 2 {
		t.Fatalf("encoded channels = %d, want 2", g.Channels())
	}
	back := g.ToFlowField()
	step := 1.0 / DefaultFlowScale
	for y := 0; y < 6; y++ {
		for x := 0; x < 5; x++ {
			wu, wv := f.At(y, x)
			gu, gv := back.At(y, x)
			if math.Abs(gu-wu) > step || math.Abs(gv-wv) > step {
				t.Errorf("(%d,%d): round trip (%.3f,%.3f) vs (%.3f,%.3f)", y, x, gu, gv, wu, wv)
			}
		}
	}
}

func TestEncodeDecodeCustomScale(t *testing.T) {
	f := optflow.NewFlowField(3, 3)
	f.Set(1, 1, 5.0, -4.0)
	g := EncodeFlow(f, 4.0)
	back := DecodeFlow(g, 4.0)
	u, v := back.At(1, 1)
	if math.Abs(u-5.0) > 0.25 || math.Abs(v+4.0) > 0.25 {
		t.Errorf("custom-scale round trip = (%.3f,%.3f), want (5,-4)", u, v)
	}
}

func TestGpuMatBasics(t *testing.T) {
	src := cv.NewMat(4, 5, 3)
	src.Set(1, 2, 0, 200)
	g := GpuMatFromMat(src)
	if g.Empty() {
		t.Fatal("GpuMat unexpectedly empty")
	}
	if r, c := g.Size(); r != 4 || c != 5 {
		t.Fatalf("size = (%d,%d), want (4,5)", r, c)
	}
	if g.Channels() != 3 {
		t.Fatalf("channels = %d, want 3", g.Channels())
	}
	// Upload clones: mutating the source must not change the GpuMat.
	src.Set(1, 2, 0, 0)
	if g.Download().At(1, 2, 0) != 200 {
		t.Error("GpuMat did not clone on upload")
	}
	cl := g.Clone()
	cl.Mat().Set(1, 2, 0, 5)
	if g.Download().At(1, 2, 0) != 200 {
		t.Error("Clone is not independent")
	}
	g.Release()
	if !g.Empty() {
		t.Error("Release did not empty the GpuMat")
	}
}

func TestStreamNoOp(t *testing.T) {
	s := NewStream()
	s.WaitForCompletion()
	if !s.QueryIfComplete() {
		t.Error("QueryIfComplete should always be true")
	}
	var nilStream *Stream
	// A nil stream is a valid default stream argument; Calc must tolerate it.
	prev, next := denseGpuPair(32, 32, 1, 0)
	_ = NewFarnebackOpticalFlow(3, 3).Calc(prev, next, nilStream)
}

func TestConstructorPanics(t *testing.T) {
	cases := []func(){
		func() { NewSparsePyrLKOpticalFlow(0, 3, 30) },
		func() { NewDensePyrLKOpticalFlow(13, -1, 30) },
		func() { NewFarnebackOpticalFlow(0, 3) },
		func() { NewNvidiaHWOpticalFlow(1, 1, -1) },
		func() { NewBroxOpticalFlow(0.2, 50, 1.5, 3, 10, 20) },
		func() { NewBroxOpticalFlow(0.2, 50, 0.5, 0, 10, 20) },
		func() { EncodeFlow(nil, 8) },
		func() { EncodeFlow(optflow.NewFlowField(2, 2), 0) },
	}
	for i, fn := range cases {
		func() {
			defer func() {
				if recover() == nil {
					t.Errorf("case %d did not panic", i)
				}
			}()
			fn()
		}()
	}
}
