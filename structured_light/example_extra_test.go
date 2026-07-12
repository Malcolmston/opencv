package structured_light_test

import (
	"fmt"
	"math"

	sl "github.com/malcolmston/opencv/structured_light"
)

// ExampleMultiFrequencyUnwrap recovers a fine phase ramp spanning many fringes
// from coarse and fine wrapped phase maps.
func ExampleMultiFrequencyUnwrap() {
	rows, cols := 1, 100
	freqs := []float64{1, 8}
	var levels []sl.FrequencyPhase
	for _, f := range freqs {
		w := make([]float64, rows*cols)
		for x := 0; x < cols; x++ {
			w[x] = sl.WrapPhase(2 * math.Pi * f * float64(x) / float64(cols))
		}
		levels = append(levels, sl.FrequencyPhase{Frequency: f, Wrapped: w})
	}
	abs, err := sl.MultiFrequencyUnwrap(levels, rows, cols, false)
	if err != nil {
		panic(err)
	}
	// The absolute phase at freq 8 rises to almost 8·2π ≈ 50 rad, well past 2π.
	fmt.Printf("start=%.2f end=%.2f fringes=%.0f\n", abs[0], abs[cols-1], math.Round((abs[cols-1]-abs[0])/(2*math.Pi)))
	// Output:
	// start=0.00 end=49.76 fringes=8
}

// ExampleTriangulatePoint reconstructs a 3-D point from its camera and projector
// projections under a synthetic calibration.
func ExampleTriangulatePoint() {
	k := [3][3]float64{{800, 0, 320}, {0, 800, 240}, {0, 0, 1}}
	id := [3][3]float64{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}
	cam := sl.NewPinhole(k, id, [3]float64{0, 0, 0})
	proj := sl.NewPinhole(k, id, [3]float64{-0.2, 0, 0})

	world := [3]float64{0.1, -0.05, 1.2}
	uc, vc := cam.Project(world)
	up, vp := proj.Project(world)
	got := sl.TriangulatePoint(cam, proj, uc, vc, up, vp)
	fmt.Printf("%.3f %.3f %.3f\n", got[0], got[1], got[2])
	// Output:
	// 0.100 -0.050 1.200
}

// ExampleCodePattern shows selecting the natural-binary encoding instead of the
// default reflected Gray code.
func ExampleCodePattern() {
	c := sl.NewCodePattern(sl.GrayCodeParams{Width: 32, Height: 16}, sl.EncodingBinary)
	fmt.Printf("encoding=%s images=%d\n", c.Encoding, c.NumberOfPatternImages())
	// Output:
	// encoding=binary images=18
}
