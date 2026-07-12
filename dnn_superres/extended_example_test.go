package dnn_superres_test

import (
	"fmt"

	cv "github.com/malcolmston/opencv"
	"github.com/malcolmston/opencv/dnn_superres"
)

// ExampleUpsampleLapSRN shows the LapSRN-style progressive upscaler enlarging a
// tiny image by ×8 through successive edge-guided doublings.
func ExampleUpsampleLapSRN() {
	src := cv.NewMat(4, 4, 3)
	src.SetTo(100)
	out, err := dnn_superres.UpsampleLapSRN(src, 8)
	if err != nil {
		panic(err)
	}
	fmt.Printf("%dx%d\n", out.Rows, out.Cols)
	// Output: 32x32
}

// ExampleUpsampleESPCN shows the ESPCN-style sub-pixel (pixel-shuffle) upscaler
// with fixed polyphase filters at an arbitrary scale.
func ExampleUpsampleESPCN() {
	src := cv.NewMat(5, 5, 1)
	src.SetTo(80)
	out, err := dnn_superres.UpsampleESPCN(src, 5)
	if err != nil {
		panic(err)
	}
	fmt.Printf("%dx%d, center=%d\n", out.Rows, out.Cols, out.At(12, 12, 0))
	// Output: 25x25, center=80
}

// ExampleSSIM measures structural similarity between identical images.
func ExampleSSIM() {
	a := cv.NewMat(16, 16, 1)
	a.SetTo(120)
	s, err := dnn_superres.SSIM(a, a.Clone())
	if err != nil {
		panic(err)
	}
	fmt.Printf("%.1f\n", s)
	// Output: 1.0
}

// ExampleBenchmark compares the default method set on a reconstruction task and
// prints the best performer.
func ExampleBenchmark() {
	hi := cv.NewMat(24, 24, 3)
	for y := 0; y < hi.Rows; y++ {
		for x := 0; x < hi.Cols; x++ {
			for c := 0; c < hi.Channels; c++ {
				hi.Set(y, x, c, uint8((x*10+y*5)%256))
			}
		}
	}
	results, err := dnn_superres.Benchmark(hi, 2, nil)
	if err != nil {
		panic(err)
	}
	fmt.Printf("methods=%d named=%t\n", len(results), results[0].Name != "")
	// Output: methods=11 named=true
}
