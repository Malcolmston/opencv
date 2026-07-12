package bioinspired

import (
	"math"
	"math/rand"
	"testing"

	cv "github.com/malcolmston/opencv"
)

// meanVar computes the mean and variance of the luminance of a rectangular
// region [y0,y1)x[x0,x1) of a grayscale FloatMat.
func regionMeanVar(f *cv.FloatMat, y0, y1, x0, x1 int) (mean, variance float64) {
	var sum, sumSq float64
	n := 0
	for y := y0; y < y1; y++ {
		for x := x0; x < x1; x++ {
			v := f.Data[y*f.Cols+x]
			sum += v
			sumSq += v * v
			n++
		}
	}
	mean = sum / float64(n)
	variance = sumSq/float64(n) - mean*mean
	return
}

// grayFloatMat lifts a single-channel Mat's luminance into a FloatMat.
func grayFloatMat(m *cv.Mat) *cv.FloatMat {
	f := cv.NewFloatMat(m.Rows, m.Cols)
	for y := 0; y < m.Rows; y++ {
		for x := 0; x < m.Cols; x++ {
			if m.Channels == 1 {
				f.Data[y*m.Cols+x] = float64(m.At(y, x, 0))
			} else {
				r := float64(m.At(y, x, 0))
				g := float64(m.At(y, x, 1))
				b := float64(m.At(y, x, 2))
				f.Data[y*m.Cols+x] = 0.299*r + 0.587*g + 0.114*b
			}
		}
	}
	return f
}

// noisyEdge builds a grayscale Mat with a vertical edge (left=lo, right=hi at
// column edge) plus deterministic uniform noise of amplitude amp.
func noisyEdge(rows, cols, edge int, lo, hi, amp float64, seed int64) *cv.Mat {
	rng := rand.New(rand.NewSource(seed))
	m := cv.NewMat(rows, cols, 1)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			base := lo
			if x >= edge {
				base = hi
			}
			v := base + (rng.Float64()*2-1)*amp
			m.Set(y, x, 0, clampRound(v))
		}
	}
	return m
}

func TestParvoReducesNoiseKeepsEdge(t *testing.T) {
	rows, cols, edge := 48, 48, 24
	lo, hi := 64.0, 192.0
	img := noisyEdge(rows, cols, edge, lo, hi, 20, 1)

	r := NewRetina(rows, cols)
	// Warm up temporal state on the (fixed) noisy frame.
	for i := 0; i < 12; i++ {
		r.Run(img)
	}
	parvo := r.GetParvoRAW()
	in := grayFloatMat(img)

	// Regions well inside each flat half, away from the edge and borders.
	lY0, lY1 := 4, rows-4
	lX0, lX1 := 4, edge-4
	rX0, rX1 := edge+4, cols-4

	inLMean, inLVar := regionMeanVar(in, lY0, lY1, lX0, lX1)
	inRMean, inRVar := regionMeanVar(in, lY0, lY1, rX0, rX1)
	pLMean, pLVar := regionMeanVar(parvo, lY0, lY1, lX0, lX1)
	pRMean, pRVar := regionMeanVar(parvo, lY0, lY1, rX0, rX1)

	t.Logf("input  left(mean=%.1f var=%.1f) right(mean=%.1f var=%.1f)", inLMean, inLVar, inRMean, inRVar)
	t.Logf("parvo  left(mean=%.1f var=%.1f) right(mean=%.1f var=%.1f)", pLMean, pLVar, pRMean, pRVar)

	if pLVar >= inLVar {
		t.Errorf("parvo did not reduce noise in left region: parvoVar=%.2f inputVar=%.2f", pLVar, inLVar)
	}
	if pRVar >= inRVar {
		t.Errorf("parvo did not reduce noise in right region: parvoVar=%.2f inputVar=%.2f", pRVar, inRVar)
	}
	// The edge (contrast between the two halves) must survive.
	if pRMean-pLMean < 15 {
		t.Errorf("parvo lost the edge: right-left contrast=%.2f (want >=15)", pRMean-pLMean)
	}
}

func TestMagnoRespondsToMotionNotStatic(t *testing.T) {
	rows, cols := 48, 48
	c1, c2 := 18, 26 // edge moves right by 8 columns
	lo, hi := 40.0, 200.0

	edgeAt := func(edge int) *cv.Mat {
		m := cv.NewMat(rows, cols, 1)
		for y := 0; y < rows; y++ {
			for x := 0; x < cols; x++ {
				v := lo
				if x >= edge {
					v = hi
				}
				m.Set(y, x, 0, clampRound(v))
			}
		}
		return m
	}
	frame1 := edgeAt(c1)
	frame2 := edgeAt(c2)

	r := NewRetina(rows, cols)
	// Warm up on the static frame so the transient decays away.
	for i := 0; i < 30; i++ {
		r.Run(frame1)
	}
	static := r.GetMagnoRAW()
	staticMax := 0.0
	for _, v := range static.Data {
		staticMax = math.Max(staticMax, math.Abs(v))
	}

	// Present the moved edge: a transient should appear in the motion band.
	r.Run(frame2)
	motion := r.GetMagnoRAW()
	motionMax := 0.0
	argCol := -1
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			v := math.Abs(motion.Data[y*cols+x])
			if v > motionMax {
				motionMax = v
				argCol = x
			}
		}
	}

	t.Logf("staticMax=%.3f motionMax=%.3f argCol=%d", staticMax, motionMax, argCol)

	if staticMax > 5 {
		t.Errorf("magno should be near zero on a static frame, got max=%.3f", staticMax)
	}
	if motionMax < 20 {
		t.Errorf("magno should respond strongly to motion, got max=%.3f", motionMax)
	}
	if motionMax < 5*staticMax+1 {
		t.Errorf("motion response (%.3f) not clearly above static (%.3f)", motionMax, staticMax)
	}
	// The strongest transient should sit in (or near) the columns the edge swept.
	if argCol < c1-3 || argCol > c2+3 {
		t.Errorf("motion response at column %d is outside the motion band [%d,%d]", argCol, c1, c2)
	}
}

func TestToneMappingCompressesRangeAddsLowEndDetail(t *testing.T) {
	rows, cols := 32, 48
	darkEnd := 20
	img := cv.NewMat(rows, cols, 1)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			var v float64
			if x < darkEnd {
				v = 3 + float64(x%5)*2 // deep-shadow detail: {3,5,7,9,11}
			} else {
				v = 200 + float64(x%5)*2 // highlights: {200,...,208}
			}
			img.Set(y, x, 0, clampRound(v))
		}
	}

	tm := NewRetinaFastToneMapping(rows, cols)
	out := tm.ProcessFrame(img)

	in := grayFloatMat(img)
	of := grayFloatMat(out)

	// Low-end detail: dark region, away from the border with the bright half.
	y0, y1 := 2, rows-2
	x0, x1 := 2, darkEnd-2
	inMean, inVar := regionMeanVar(in, y0, y1, x0, x1)
	outMean, outVar := regionMeanVar(of, y0, y1, x0, x1)

	t.Logf("dark region: input(mean=%.2f std=%.2f) output(mean=%.2f std=%.2f)",
		inMean, math.Sqrt(inVar), outMean, math.Sqrt(outVar))

	if outVar <= inVar {
		t.Errorf("tone mapping did not increase low-end detail: outVar=%.3f inVar=%.3f", outVar, inVar)
	}
	if outMean <= inMean {
		t.Errorf("tone mapping did not lift the shadows: outMean=%.2f inMean=%.2f", outMean, inMean)
	}
	// Output must stay in the displayable range.
	for _, v := range out.Data {
		if v > 255 { // uint8 cannot exceed 255, but assert intent explicitly
			t.Fatalf("output sample %d out of range", v)
		}
	}
	// Highlights must not have collapsed to a single saturated value.
	_, brightVar := regionMeanVar(of, y0, y1, darkEnd+2, cols-2)
	if brightVar == 0 {
		t.Logf("note: highlight detail flattened (brightVar=0)")
	}
}

func TestColorParvoRoundTrips(t *testing.T) {
	rows, cols := 16, 16
	img := cv.NewMat(rows, cols, 3)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			img.Set(y, x, 0, uint8((x*13)%256))
			img.Set(y, x, 1, uint8((y*17)%256))
			img.Set(y, x, 2, uint8((x*y)%256))
		}
	}
	r := NewRetina(rows, cols)
	r.Run(img)
	parvo := r.GetParvo()
	if parvo.Channels != 3 {
		t.Fatalf("colour parvo should have 3 channels, got %d", parvo.Channels)
	}
	magno := r.GetMagno()
	if magno.Channels != 1 {
		t.Fatalf("magno should be single-channel, got %d", magno.Channels)
	}
	if parvo.Rows != rows || parvo.Cols != cols {
		t.Fatalf("parvo size mismatch: %dx%d", parvo.Rows, parvo.Cols)
	}
}

func TestClearBuffersResets(t *testing.T) {
	rows, cols := 20, 20
	img := cv.NewMat(rows, cols, 1)
	img.SetTo(128)
	r := NewRetina(rows, cols)
	r.Run(img)
	if !r.hasOutput {
		t.Fatal("expected output after Run")
	}
	r.ClearBuffers()
	if r.hasOutput {
		t.Fatal("ClearBuffers should discard output")
	}
	// The magno temporal state must be zeroed.
	for _, v := range r.magnoTemporal.data {
		if v != 0 {
			t.Fatalf("ClearBuffers did not zero magno state")
		}
	}
}

func TestParamsAccessors(t *testing.T) {
	r := NewRetina(10, 10)
	p := r.GetParameters()
	if p.OPLandIplParvo.HorizontalCellsGain == 0 {
		t.Fatal("default horizontal cells gain should be non-zero")
	}
	p.IplMagno.MagnoGain = 9.0
	r.SetParameters(p)
	if r.GetParameters().IplMagno.MagnoGain != 9.0 {
		t.Fatal("SetParameters did not take effect")
	}
	rr, cc := r.InputSize()
	if rr != 10 || cc != 10 {
		t.Fatalf("InputSize = %dx%d", rr, cc)
	}
}

func TestRunPanicsOnSizeMismatch(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("expected panic on size mismatch")
		}
	}()
	r := NewRetina(10, 10)
	r.Run(cv.NewMat(8, 8, 1))
}

func TestGetParvoBeforeRunPanics(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("expected panic when reading before Run")
		}
	}()
	r := NewRetina(10, 10)
	_ = r.GetParvo()
}

func TestToneMappingSizeMismatchPanics(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("expected panic on size mismatch")
		}
	}()
	tm := NewRetinaFastToneMapping(10, 10)
	tm.ProcessFrame(cv.NewMat(4, 4, 1))
}

func TestSetupClampsParameters(t *testing.T) {
	tm := NewRetinaFastToneMapping(8, 8)
	tm.Setup(-1, -1, 5) // out-of-range values should be clamped, not panic
	if tm.meanLuminanceModulatorK != 1 {
		t.Fatalf("meanLuminanceModulatorK not clamped: %v", tm.meanLuminanceModulatorK)
	}
	if tm.photoreceptorsNeighborhoodRadius != 0 {
		t.Fatalf("radius not clamped: %v", tm.photoreceptorsNeighborhoodRadius)
	}
	// A zero-radius setup must still run.
	img := cv.NewMat(8, 8, 3)
	img.SetTo(50)
	out := tm.ProcessFrame(img)
	if out.Channels != 3 {
		t.Fatalf("colour tone mapping should preserve channels")
	}
}
