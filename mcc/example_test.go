package mcc_test

import (
	"fmt"

	cv "github.com/malcolmston/opencv"
	"github.com/malcolmston/opencv/mcc"
)

// ExampleRenderChart renders the classic 24-patch chart and reports its size.
func ExampleRenderChart() {
	img := mcc.RenderChart(mcc.Macbeth24, 20, 6)
	fmt.Printf("%dx%d, %d channels\n", img.Cols, img.Rows, img.Channels)
	// Output: 162x110, 3 channels
}

// ExampleCCheckerDetector_DetectWithHint samples a chart from its known outer
// corners and reports the mean color error against the reference.
func ExampleCCheckerDetector_DetectWithHint() {
	patch, gap := 30, 8
	img := mcc.RenderChart(mcc.Macbeth24, patch, gap)
	quad := mcc.ChartOuterQuad(mcc.Macbeth24, patch, gap)

	d := mcc.NewCCheckerDetector(mcc.Macbeth24)
	cc, ok := d.DetectWithHint(img, quad)
	fmt.Printf("detected=%v patches=%d meanDeltaE<1=%v\n", ok, len(cc.MeasuredRGB()), cc.MeanError() < 1)
	// Output: detected=true patches=24 meanDeltaE<1=true
}

// ExampleTrainCCM fits a color-correction matrix that maps a camera's measured
// patch colors back to the chart reference and confirms it reduces error.
func ExampleTrainCCM() {
	ref := mcc.ReferenceRGB(mcc.Macbeth24)
	// Pretend the camera darkened blue and lifted black slightly.
	measured := make([][3]float64, len(ref))
	for i, c := range ref {
		measured[i] = [3]float64{c[0]*0.95 + 5, c[1]*0.97 + 4, c[2]*0.9 + 3}
	}
	before := mcc.MeanDeltaE(measured, ref)

	model, err := mcc.TrainCCM(measured, ref, mcc.CCMConfig{Type: mcc.CCMAffine3x4})
	if err != nil {
		panic(err)
	}
	after := model.MeanError(measured, ref)
	fmt.Printf("improved=%v\n", after < before)
	// Output: improved=true
}

// ExampleDeltaERGB shows the CIE76 color difference between two sRGB colors.
func ExampleDeltaERGB() {
	same := mcc.DeltaERGB([3]uint8{120, 130, 140}, [3]uint8{120, 130, 140})
	diff := mcc.DeltaERGB([3]uint8{120, 130, 140}, [3]uint8{130, 130, 140})
	fmt.Printf("%.0f %v\n", same, diff > 0)
	// Output: 0 true
}

// ExampleCCM_Apply color-corrects an image with a fitted model.
func ExampleCCM_Apply() {
	ref := mcc.ReferenceRGB(mcc.Macbeth24)
	measured := make([][3]float64, len(ref))
	for i, c := range ref {
		measured[i] = [3]float64{c[0] * 0.9, c[1] * 0.95, c[2] * 0.85}
	}
	model, _ := mcc.TrainCCM(measured, ref, mcc.CCMConfig{Type: mcc.CCMLinear3x3})

	img := cv.NewMat(1, 1, 3)
	img.SetPixel(0, 0, []uint8{uint8(measured[6][0]), uint8(measured[6][1]), uint8(measured[6][2])})
	out := model.Apply(img)
	fmt.Printf("channels=%d\n", out.Channels)
	// Output: channels=3
}
