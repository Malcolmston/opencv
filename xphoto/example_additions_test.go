package xphoto_test

import (
	"fmt"

	cv "github.com/malcolmston/opencv"
	"github.com/malcolmston/opencv/xphoto"
)

func ExampleGammaCorrection() {
	src := solid(1, 1, 0, 128, 255)
	// gamma > 1 brightens interior values while fixing 0 and 255.
	out := xphoto.GammaCorrection(src, 2.2)
	px := out.AtPixel(0, 0)
	fmt.Println(px[0], px[1] > 128, px[2])
	// Output: 0 true 255
}

func ExampleShadesOfGray() {
	// A warm-cast flat patch pulled back toward neutral (p = 1 is gray-world).
	src := solid(4, 4, 160, 128, 96)
	out := xphoto.ShadesOfGray(src, 1)
	px := out.AtPixel(0, 0)
	fmt.Println(px[0] == px[1] && px[1] == px[2])
	// Output: true
}

func ExampleWhitePatchWB() {
	src := solid(4, 4, 200, 160, 120)
	out := xphoto.WhitePatchWB(src)
	fmt.Println(out.Rows, out.Cols, out.Channels)
	// Output: 4 4 3
}

func ExampleCreateSimpleWB() {
	wb := xphoto.CreateSimpleWB()
	wb.SetP(1.0)
	src := solid(2, 2, 200, 180, 150)
	out := wb.BalanceWhite(src)
	fmt.Println(wb.GetP(), out.Channels)
	// Output: 1 3
}

func ExampleDctDenoising() {
	src := cv.NewMat(16, 16, 1)
	for i := range src.Data {
		src.Data[i] = 128
	}
	out := xphoto.DctDenoising(src, 10, 8)
	fmt.Println(out.At(8, 8, 0))
	// Output: 128
}

func ExampleBm3dDenoisingTwoStep() {
	src := cv.NewMat(16, 16, 1)
	for i := range src.Data {
		src.Data[i] = 100
	}
	out := xphoto.Bm3dDenoisingTwoStep(src, 10)
	fmt.Println(out.At(8, 8, 0))
	// Output: 100
}

func ExampleDehaze() {
	// A flat patch has no scene structure to recover; dehazing keeps its shape.
	src := solid(8, 8, 200, 190, 180)
	out := xphoto.Dehaze(src)
	fmt.Println(out.Rows, out.Cols, out.Channels)
	// Output: 8 8 3
}

func ExampleOilpaintingColorSpace() {
	src := cv.NewMat(8, 8, 3)
	for p := 0; p < src.Total(); p++ {
		src.Data[p*3+0] = uint8((p * 11) % 256)
		src.Data[p*3+1] = uint8((p * 7) % 256)
		src.Data[p*3+2] = uint8((p * 5) % 256)
	}
	out := xphoto.OilpaintingColorSpace(src, 2, 32, xphoto.OilIntensityValue)
	fmt.Println(out.Rows, out.Cols, out.Channels)
	// Output: 8 8 3
}

func ExampleInpaintFSR() {
	src := solid(16, 16, 100, 120, 140)
	mask := cv.NewMat(16, 16, 1)
	mask.Set(8, 8, 0, 255) // one unknown pixel
	src.SetPixel(8, 8, []uint8{0, 0, 0})
	out := xphoto.InpaintFSR(src, mask, xphoto.FSRBest)
	px := out.AtPixel(8, 8)
	// Reconstructed from the surrounding flat region.
	fmt.Println(px[0], px[1], px[2])
	// Output: 100 120 140
}

func ExampleTonemapDurand() {
	src := solid(8, 8, 128, 128, 128)
	tm := xphoto.NewTonemapDurand()
	out := tm.Process(src)
	fmt.Println(out.Rows, out.Cols, out.Channels)
	// Output: 8 8 3
}
