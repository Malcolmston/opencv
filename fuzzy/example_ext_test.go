package fuzzy_test

import (
	"fmt"

	cv "github.com/malcolmston/opencv"
	"github.com/malcolmston/opencv/fuzzy"
)

func ExampleFT12DProcess() {
	// The degree-1 F-transform reconstructs a linear ramp essentially exactly,
	// where the degree-0 transform would round the local slope to a constant.
	img := cv.NewMat(16, 16, 1)
	for y := 0; y < 16; y++ {
		for x := 0; x < 16; x++ {
			img.Data[y*16+x] = uint8(10 + 3*x)
		}
	}
	out := fuzzy.FT12DProcess(img, fuzzy.CreateKernel(fuzzy.LinearBasis, 3), nil)
	fmt.Println(out.At(8, 4, 0), out.At(8, 5, 0), out.At(8, 6, 0))
	// Output: 22 25 28
}

func ExampleFT12DPolynomial() {
	// The fitted per-node coefficients recover the ramp's slope directly: c10 is
	// the horizontal gradient (+3 per pixel here) at every interior node.
	img := cv.NewMat(20, 20, 1)
	for y := 0; y < 20; y++ {
		for x := 0; x < 20; x++ {
			img.Data[y*20+x] = uint8(5 + 3*x)
		}
	}
	comps, _, c10, _ := fuzzy.FT12DPolynomial(img, fuzzy.CreateKernel(fuzzy.LinearBasis, 3), nil, 0)
	fmt.Printf("%.1f", c10.At(comps.Bn/2, comps.An/2))
	// Output: 3.0
}

func ExampleCreateKernelAB() {
	// An anisotropic kernel from two different 1-D profiles: element (y,x) is the
	// outer product b[y]*a[x], so the centre is 1 and the corners are 0.
	a := fuzzy.CreateKernel1D(fuzzy.LinearBasis, 2) // width 5
	b := fuzzy.CreateKernel1D(fuzzy.SinusBasis, 3)  // height 7
	k := fuzzy.CreateKernelAB(a, b)
	fmt.Printf("%dx%d centre=%.1f\n", k.Rows, k.Cols, k.At(3, 2))
	// Output: 7x5 centre=1.0
}

func ExampleFT02DFLProcess() {
	// The fast separable variant produces the same partition-of-unity result as
	// the dense process on a flat field.
	img := cv.NewMat(12, 12, 1)
	img.SetTo(64)
	out := fuzzy.FT02DFLProcess(img, fuzzy.LinearBasis, 3)
	fmt.Println(out.At(6, 6, 0))
	// Output: 64
}

func ExampleInpaintMultiStep() {
	// Multi-step fills a hole wider than the kernel that one-step cannot complete.
	img := cv.NewMat(24, 24, 1)
	img.SetTo(150)
	mask := cv.NewMat(24, 24, 1)
	for y := 8; y < 16; y++ {
		for x := 8; x < 16; x++ {
			img.Data[y*24+x] = 0
			mask.Data[y*24+x] = 255
		}
	}
	out := fuzzy.InpaintMultiStep(img, mask, 3, fuzzy.LinearBasis)
	fmt.Println(out.At(12, 12, 0))
	// Output: 150
}

func ExampleTransformError() {
	// Quality reporting: a perfect reconstruction has zero error and infinite PSNR.
	img := cv.NewMat(8, 8, 1)
	img.SetTo(200)
	out := fuzzy.Filter(img, fuzzy.LinearBasis, 2)
	fmt.Println(fuzzy.TransformError(img, out).String())
	// Output: MAE=0.000 RMSE=0.000 Max=0 PSNR=infdB
}
