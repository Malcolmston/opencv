package bioinspired_test

import (
	"fmt"

	cv "github.com/malcolmston/opencv"
	"github.com/malcolmston/opencv/bioinspired"
)

// ExampleMosaicBayer shows that demosaicing a Bayer mosaic recovers a smooth
// image almost exactly.
func ExampleMosaicBayer() {
	rows, cols := 16, 16
	img := cv.NewMat(rows, cols, 3)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			img.Set(y, x, 0, uint8(40+x))
			img.Set(y, x, 1, uint8(50+y))
			img.Set(y, x, 2, uint8(60+(x+y)/2))
		}
	}
	mosaic := bioinspired.MosaicBayer(img, bioinspired.BayerRGGB)
	back := bioinspired.DemosaicBayer(mosaic, bioinspired.BayerRGGB)

	fmt.Printf("mosaic channels: %d\n", mosaic.Channels)
	fmt.Printf("recovered red near-exact: %v\n", abs8(img.At(8, 8, 0), back.At(8, 8, 0)) <= 1)
	// Output:
	// mosaic channels: 1
	// recovered red near-exact: true
}

// ExampleTransientAreasSegmentationModule flags a compact moving region while
// leaving a static scene unsegmented.
func ExampleTransientAreasSegmentationModule() {
	rows, cols := 32, 32
	seg := bioinspired.NewTransientAreasSegmentationModule(rows, cols)

	blob := cv.NewFloatMat(rows, cols)
	for y := 14; y < 18; y++ {
		for x := 14; x < 18; x++ {
			blob.Data[y*cols+x] = 200 // transient motion energy
		}
	}
	for i := 0; i < 8; i++ {
		seg.RunFloat(blob)
	}
	pic := seg.GetSegmentationPicture()

	fmt.Printf("blob centre segmented: %v\n", pic.At(16, 16, 0) == 255)
	fmt.Printf("static corner segmented: %v\n", pic.At(0, 0, 0) == 255)
	// Output:
	// blob centre segmented: true
	// static corner segmented: false
}

// ExampleWriteRetinaParameters round-trips a parameter set through its text form.
func ExampleWriteRetinaParameters() {
	p := bioinspired.DefaultRetinaParameters()
	p.IplMagno.MagnoGain = 5.5

	text := bioinspired.WriteRetinaParameters(p)
	back, err := bioinspired.ReadRetinaParameters(text)

	fmt.Printf("parse ok: %v\n", err == nil)
	fmt.Printf("gain preserved: %v\n", back.IplMagno.MagnoGain == 5.5)
	// Output:
	// parse ok: true
	// gain preserved: true
}

// ExampleRetinaProcessor shows the channel-activation toggles: disabling the
// moving-contours channel makes the magno output all zero.
func ExampleRetinaProcessor() {
	rows, cols := 16, 16
	img := cv.NewMat(rows, cols, 1)
	img.SetTo(120)

	rp := bioinspired.NewRetinaProcessor(rows, cols)
	rp.Run(img)
	rp.ActivateMovingContoursProcessing(false)

	magno := rp.GetMagno()
	allZero := true
	for _, v := range magno.Data {
		if v != 0 {
			allZero = false
			break
		}
	}
	fmt.Printf("magno disabled -> all zero: %v\n", allZero)
	// Output:
	// magno disabled -> all zero: true
}

func abs8(a, b uint8) int {
	d := int(a) - int(b)
	if d < 0 {
		return -d
	}
	return d
}
