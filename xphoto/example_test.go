package xphoto_test

import (
	"fmt"

	cv "github.com/malcolmston/opencv"
	"github.com/malcolmston/opencv/xphoto"
)

// solid returns a 3-channel image filled with the given RGB colour.
func solid(rows, cols int, r, g, b uint8) *cv.Mat {
	m := cv.NewMat(rows, cols, 3)
	for p := 0; p < m.Total(); p++ {
		m.Data[p*3+0] = r
		m.Data[p*3+1] = g
		m.Data[p*3+2] = b
	}
	return m
}

func ExampleApplyChannelGains() {
	src := solid(1, 1, 100, 100, 100)
	out := xphoto.ApplyChannelGains(src, 1.2, 1.0, 0.5)
	px := out.AtPixel(0, 0)
	fmt.Println(px[0], px[1], px[2])
	// Output: 120 100 50
}

func ExampleSimpleWB() {
	// A flat, slightly warm patch stretched back toward neutral.
	src := solid(2, 2, 200, 180, 150)
	wb := xphoto.NewSimpleWB()
	out := wb.BalanceWhite(src)
	fmt.Println(out.Channels, out.Rows, out.Cols)
	// Output: 3 2 2
}

func ExampleGrayworldWB() {
	src := solid(4, 4, 160, 128, 96) // warm cast
	wb := xphoto.NewGrayworldWB()
	out := wb.BalanceWhite(src)
	// Gray-world equalises the channel means; on a flat patch every pixel
	// becomes the same neutral grey.
	px := out.AtPixel(0, 0)
	fmt.Println(px[0] == px[1] && px[1] == px[2])
	// Output: true
}

func ExampleLearningBasedWB() {
	src := solid(8, 8, 150, 120, 90)
	var wb xphoto.WhiteBalancer = xphoto.NewLearningBasedWB()
	out := wb.BalanceWhite(src)
	fmt.Println(out.Rows, out.Cols)
	// Output: 8 8
}

func ExampleOilpainting() {
	src := cv.NewMat(8, 8, 3)
	for p := 0; p < src.Total(); p++ {
		src.Data[p*3+0] = uint8((p * 11) % 256)
		src.Data[p*3+1] = uint8((p * 7) % 256)
		src.Data[p*3+2] = uint8((p * 5) % 256)
	}
	out := xphoto.Oilpainting(src, 2, 32)
	fmt.Println(out.Rows, out.Cols, out.Channels)
	// Output: 8 8 3
}

func ExampleBm3dDenoising() {
	src := cv.NewMat(16, 16, 1)
	for i := range src.Data {
		src.Data[i] = 128
	}
	out := xphoto.Bm3dDenoising(src, 10)
	fmt.Println(out.At(8, 8, 0))
	// Output: 128
}

func ExampleInpaint() {
	src := solid(8, 8, 100, 100, 100)
	mask := cv.NewMat(8, 8, 1)
	mask.Set(4, 4, 0, 255) // one pixel to fill
	src.SetPixel(4, 4, []uint8{0, 0, 0})
	out := xphoto.Inpaint(src, mask)
	fmt.Println(out.At(4, 4, 0)) // filled from the surrounding flat region
	// Output: 100
}
