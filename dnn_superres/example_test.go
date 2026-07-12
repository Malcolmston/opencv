package dnn_superres_test

import (
	"fmt"

	cv "github.com/malcolmston/opencv"
	"github.com/malcolmston/opencv/dnn_superres"
)

// ExampleDnnSuperResImpl shows the OpenCV-style stateful workflow: construct the
// engine, choose an algorithm and scale, then upsample.
func ExampleDnnSuperResImpl() {
	src := cv.NewMat(8, 8, 3)
	src.SetTo(120)

	sr := dnn_superres.NewDnnSuperResImpl()
	if err := sr.SetModel("bicubic", 3); err != nil {
		panic(err)
	}
	out, err := sr.Upsample(src)
	if err != nil {
		panic(err)
	}
	fmt.Printf("%s x%d: %dx%d -> %dx%d\n",
		sr.GetAlgorithm(), sr.GetScale(), src.Rows, src.Cols, out.Rows, out.Cols)
	// Output: bicubic x3: 8x8 -> 24x24
}

// ExampleUpsampleLanczos shows the free-function API without the stateful
// wrapper.
func ExampleUpsampleLanczos() {
	src := cv.NewMat(5, 5, 1)
	src.SetTo(200)
	out, err := dnn_superres.UpsampleLanczos(src, 2)
	if err != nil {
		panic(err)
	}
	fmt.Printf("%dx%d, center=%d\n", out.Rows, out.Cols, out.At(5, 5, 0))
	// Output: 10x10, center=200
}

// ExamplePSNR measures reconstruction quality between two equally-sized images.
func ExamplePSNR() {
	a := cv.NewMat(4, 4, 1)
	b := cv.NewMat(4, 4, 1)
	p, err := dnn_superres.PSNR(a, b)
	if err != nil {
		panic(err)
	}
	fmt.Println(p) // identical images -> +Inf
	// Output: +Inf
}
