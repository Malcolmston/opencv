package gapi_test

import (
	"fmt"

	cv "github.com/malcolmston/opencv"
	"github.com/malcolmston/opencv/gapi"
)

// ExampleNewComputation builds and runs the classic edge-detection pipeline
// Canny(GaussianBlur(RGB2Gray(in))) as a compiled graph.
func ExampleNewComputation() {
	// Describe the graph symbolically; nothing runs yet.
	in := gapi.NewMat()
	edges := gapi.Canny(gapi.GaussianBlur(gapi.RGB2Gray(in), 5, 1.4), 50, 100)

	// Compile once, then apply to a concrete image.
	cc := gapi.NewComputation(in, edges).Compile()

	img := cv.NewMat(16, 16, 3)
	for i := range img.Data {
		img.Data[i] = uint8(i % 256)
	}
	out := cc.Apply(img)[0]

	fmt.Printf("%dx%dx%d\n", out.Rows, out.Cols, out.Channels)
	// Output: 16x16x1
}

// ExampleComputationT defines a reusable single-in single-out pipeline with the
// typed helper.
func ExampleComputationT() {
	pipe := gapi.NewComputationT(func(in gapi.GMat) gapi.GMat {
		return gapi.EqualizeHist(gapi.RGB2Gray(in))
	})

	img := cv.NewMat(8, 8, 3)
	for i := range img.Data {
		img.Data[i] = uint8((i * 3) % 256)
	}
	out := pipe.Apply(img)
	fmt.Printf("%dx%dx%d\n", out.Rows, out.Cols, out.Channels)
	// Output: 8x8x1
}

// ExampleGCompiled_Run drives a two-input graph and reads a single output.
func ExampleGCompiled_Run() {
	a := gapi.NewMat()
	b := gapi.NewMat()
	blended := gapi.AddWeighted(a, 0.5, b, 0.5, 0)

	cc := gapi.NewComputationMulti([]gapi.GMat{a, b}, []gapi.GMat{blended}).Compile()

	ma := cv.NewMat(2, 2, 1)
	ma.SetTo(100)
	mb := cv.NewMat(2, 2, 1)
	mb.SetTo(200)

	outs, _ := cc.Run(gapi.Inputs{Mats: []*cv.Mat{ma, mb}})
	fmt.Println(outs[0].Data[0])
	// Output: 150
}

// ExampleKernels overrides an operation with a custom kernel at compile time.
func ExampleKernels() {
	a := gapi.NewMat()
	b := gapi.NewMat()
	comp := gapi.NewComputationMulti([]gapi.GMat{a, b}, []gapi.GMat{gapi.Add(a, b)})

	// Replace "add" with a saturating max.
	pkg := gapi.Kernels(gapi.GKernel{
		Op: gapi.OpAdd,
		Eval: func(ctx gapi.KernelContext) *cv.Mat {
			return cv.Max(ctx.Mats[0], ctx.Mats[1])
		},
	})

	ma := cv.NewMat(1, 1, 1)
	ma.SetTo(30)
	mb := cv.NewMat(1, 1, 1)
	mb.SetTo(200)

	out := comp.CompileWith(pkg).Apply(ma, mb)[0]
	fmt.Println(out.Data[0])
	// Output: 200
}
