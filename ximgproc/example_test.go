package ximgproc_test

import (
	"fmt"

	cv "github.com/malcolmston/opencv"
	"github.com/malcolmston/opencv/ximgproc"
)

// ExampleGuidedFilter smooths an image while preserving edges by guiding the
// filter with the image itself (self-guided mode).
func ExampleGuidedFilter() {
	src := cv.NewMat(16, 16, 1)
	for i := range src.Data {
		src.Data[i] = 100
	}
	// Add a single bright outlier.
	src.Data[8*16+8] = 240

	out := ximgproc.GuidedFilter(src, src, 3, 500)
	fmt.Println(out.Rows, out.Cols, out.Channels)
	// The outlier is averaged down toward its neighbourhood.
	fmt.Println(out.Data[8*16+8] < 240)
	// Output:
	// 16 16 1
	// true
}

// ExampleThinning reduces a thick binary bar to a one-pixel skeleton.
func ExampleThinning() {
	m := cv.NewMat(9, 20, 1)
	for y := 3; y < 6; y++ {
		for x := 2; x < 18; x++ {
			m.Data[y*20+x] = 255
		}
	}
	sk := ximgproc.Thinning(m)

	blocks := 0
	for y := 0; y < 8; y++ {
		for x := 0; x < 19; x++ {
			if sk.Data[y*20+x] != 0 && sk.Data[y*20+x+1] != 0 &&
				sk.Data[(y+1)*20+x] != 0 && sk.Data[(y+1)*20+x+1] != 0 {
				blocks++
			}
		}
	}
	fmt.Println("solid 2x2 blocks:", blocks)
	// Output:
	// solid 2x2 blocks: 0
}

// ExampleNiBlackThreshold binarizes with the Sauvola local-threshold rule.
func ExampleNiBlackThreshold() {
	img := cv.NewMat(12, 12, 1)
	for y := 0; y < 12; y++ {
		for x := 0; x < 12; x++ {
			v := 50 + 12*x // horizontal illumination ramp
			if x%4 == 0 {
				v -= 40 // dark stroke
			}
			if v < 0 {
				v = 0
			}
			if v > 255 {
				v = 255
			}
			img.Data[y*12+x] = uint8(v)
		}
	}
	out := ximgproc.NiBlackThreshold(img, 0.2, 7, int(ximgproc.NiBlackSauvola))
	fmt.Println(out.Channels)
	// Output:
	// 1
}

// ExampleSuperpixelSLIC segments a colour image into compact superpixels.
func ExampleSuperpixelSLIC() {
	img := cv.NewMat(32, 32, 3)
	for y := 0; y < 32; y++ {
		for x := 0; x < 32; x++ {
			i := (y*32 + x) * 3
			img.Data[i+0] = uint8((x / 8) * 60)
			img.Data[i+1] = uint8((y / 8) * 60)
			img.Data[i+2] = 128
		}
	}
	labels, n := ximgproc.SuperpixelSLIC(img, 8, 20)
	fmt.Println(labels.Rows, labels.Cols)
	fmt.Println(n > 0)
	// Output:
	// 32 32
	// true
}

// ExampleAnisotropicDiffusion applies a few Perona–Malik iterations.
func ExampleAnisotropicDiffusion() {
	src := cv.NewMat(10, 10, 1)
	for i := range src.Data {
		src.Data[i] = 128
	}
	out := ximgproc.AnisotropicDiffusion(src, 0.2, 20, 5)
	fmt.Println(out.Data[55]) // a flat image stays flat
	// Output:
	// 128
}
