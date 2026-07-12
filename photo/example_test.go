package photo

import (
	"fmt"
	"image"

	cv "github.com/malcolmston/opencv"
)

func ExampleFastNlMeansDenoising() {
	// A perfectly flat image has identical patches everywhere, so the weighted
	// average leaves it unchanged.
	img := cv.NewMat(5, 5, 1)
	img.SetTo(100)
	out := FastNlMeansDenoising(img, 10, 3, 5)
	fmt.Println(out.At(2, 2, 0))
	// Output: 100
}

func ExampleInpaint() {
	// Fill a single corrupted pixel surrounded by a uniform field.
	img := cv.NewMat(3, 3, 1)
	img.SetTo(50)
	img.Set(1, 1, 0, 200)
	mask := cv.NewMat(3, 3, 1)
	mask.Set(1, 1, 0, 255)
	out := Inpaint(img, mask, 1, InpaintNS)
	fmt.Println(out.At(1, 1, 0))
	// Output: 50
}

func ExampleDecolor() {
	// A constant colour image decolorizes to a constant single-channel image.
	img := cv.NewMat(2, 2, 3)
	for i := range img.Data {
		img.Data[i] = 90
	}
	gray, _ := Decolor(img)
	fmt.Println(gray.Channels, gray.At(0, 0, 0))
	// Output: 1 90
}

func ExampleSeamlessClone() {
	// Cloning a flat patch onto a uniform background reproduces the background.
	dst := cv.NewMat(11, 11, 3)
	dst.SetTo(100)
	src := cv.NewMat(7, 7, 3)
	src.SetTo(40)
	mask := cv.NewMat(7, 7, 1)
	for y := 2; y < 5; y++ {
		for x := 2; x < 5; x++ {
			mask.Set(y, x, 0, 255)
		}
	}
	out := SeamlessClone(src, dst, mask, image.Point{X: 5, Y: 5}, NormalClone)
	fmt.Println(out.At(5, 5, 0))
	// Output: 100
}
