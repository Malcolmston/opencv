package structured_light_test

import (
	"fmt"

	cv "github.com/malcolmston/opencv"
	sl "github.com/malcolmston/opencv/structured_light"
)

// ExampleGrayCodePattern shows generating a Gray-code stack, simulating a
// capture with an identity camera→projector mapping, and decoding it back.
func ExampleGrayCodePattern() {
	g := sl.NewGrayCodePattern(sl.GrayCodeParams{Width: 32, Height: 16})
	fmt.Println("pattern images:", g.NumberOfPatternImages())

	patterns := g.Generate()

	// Simulate a capture: the camera sees the projector directly (identity),
	// so a pixel's decoded coordinate equals its own position.
	rows, cols := 16, 32
	captured := make([]*cv.Mat, len(patterns))
	for i, p := range patterns {
		captured[i] = p.Clone()
	}
	white, black := g.ReferenceImages()

	dec, err := g.Decode(captured, white, black)
	if err != nil {
		panic(err)
	}
	col, row, ok := dec.At(7, 20)
	fmt.Printf("pixel (20,7) -> col=%d row=%d valid=%v\n", col, row, ok)
	_ = rows
	_ = cols
	// Output:
	// pattern images: 18
	// pixel (20,7) -> col=20 row=7 valid=true
}

// ExampleSinusoidalPattern shows generating fringe patterns and recovering the
// unwrapped absolute phase, which is proportional to the projector column.
func ExampleSinusoidalPattern() {
	s := sl.NewSinusoidalPattern(sl.Params{
		Width:              64,
		Height:             2,
		NumOfPatternImages: 4,
		Frequency:          2,
	})
	patterns := s.Generate()
	wrapped := s.ComputeWrappedPhase(patterns)
	abs := sl.UnwrapPhaseMap(wrapped, 2, 64, false)

	// Absolute phase rises monotonically across the row.
	p0 := abs[0]
	if p0 > -0.005 && p0 < 0.005 {
		p0 = 0
	}
	fmt.Printf("phase[0]=%.2f increasing=%v\n", p0, abs[10] > abs[0] && abs[63] > abs[10])
	// Output:
	// phase[0]=0.00 increasing=true
}
