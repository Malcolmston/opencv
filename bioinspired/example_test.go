package bioinspired_test

import (
	"fmt"

	cv "github.com/malcolmston/opencv"
	"github.com/malcolmston/opencv/bioinspired"
)

// ExampleRetina shows the parvo and magno channels of the retina model. The
// magno channel stays near zero while the input is static, then fires when an
// edge moves between frames.
func ExampleRetina() {
	rows, cols := 32, 32

	edgeAt := func(edge int) *cv.Mat {
		m := cv.NewMat(rows, cols, 1)
		for y := 0; y < rows; y++ {
			for x := 0; x < cols; x++ {
				v := uint8(40)
				if x >= edge {
					v = 200
				}
				m.Set(y, x, 0, v)
			}
		}
		return m
	}

	r := bioinspired.NewRetina(rows, cols)

	// Present a static edge until the transient response settles.
	still := edgeAt(12)
	for i := 0; i < 20; i++ {
		r.Run(still)
	}
	staticMax := maxAbs(r.GetMagnoRAW().Data)

	// Move the edge: the magno channel now responds.
	r.Run(edgeAt(18))
	motionMax := maxAbs(r.GetMagnoRAW().Data)

	fmt.Printf("static magno strong: %v\n", staticMax > 5)
	fmt.Printf("motion magno strong: %v\n", motionMax > 20)
	// Output:
	// static magno strong: false
	// motion magno strong: true
}

// ExampleRetinaFastToneMapping compresses a deep-shadow image so its low-end
// detail becomes visible.
func ExampleRetinaFastToneMapping() {
	rows, cols := 8, 8
	img := cv.NewMat(rows, cols, 1)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			img.Set(y, x, 0, uint8(2+x)) // faint gradient in deep shadow
		}
	}

	tm := bioinspired.NewRetinaFastToneMapping(rows, cols)
	out := tm.ProcessFrame(img)

	// The shadow gradient is lifted well above its original range.
	fmt.Printf("shadows lifted: %v\n", out.At(0, 7, 0) > img.At(0, 7, 0))
	// Output:
	// shadows lifted: true
}

func maxAbs(data []float64) float64 {
	m := 0.0
	for _, v := range data {
		if v < 0 {
			v = -v
		}
		if v > m {
			m = v
		}
	}
	return m
}
