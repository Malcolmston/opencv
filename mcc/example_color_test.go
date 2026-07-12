package mcc_test

import (
	"fmt"

	"github.com/malcolmston/opencv/mcc"
)

// ExampleDeltaE2000 evaluates the CIEDE2000 difference on a Sharma reference
// pair and confirms it matches the published value.
func ExampleDeltaE2000() {
	d := mcc.DeltaE2000(
		[3]float64{50, 2.6772, -79.7751},
		[3]float64{50, 0, -82.7485},
	)
	fmt.Printf("%.4f\n", d)
	// Output: 2.0425
}

// ExampleLabToLCh converts a Lab color to cylindrical LCh coordinates.
func ExampleLabToLCh() {
	lch := mcc.LabToLCh([3]float64{50, 0, 30})
	fmt.Printf("L=%.0f C=%.0f h=%.0f\n", lch[0], lch[1], lch[2])
	// Output: L=50 C=30 h=90
}

// ExampleChromaticAdaptation adapts the D65 white point to illuminant A and
// recovers the destination white point exactly.
func ExampleChromaticAdaptation() {
	got := mcc.ChromaticAdaptation(mcc.WhiteD65, mcc.WhiteD65, mcc.WhiteA, mcc.Bradford)
	fmt.Printf("%.2f %.2f %.2f\n", got[0], got[1], got[2])
	// Output: 1.10 1.00 0.36
}

// ExampleXYZToxyY reports the chromaticity of the D65 white point.
func ExampleXYZToxyY() {
	xyY := mcc.XYZToxyY(mcc.WhiteD65)
	fmt.Printf("x=%.4f y=%.4f\n", xyY[0], xyY[1])
	// Output: x=0.3127 y=0.3290
}

// ExampleTrainColorCorrection fits a root-polynomial model and confirms it
// reduces the perceptual (CIEDE2000) error of a simulated camera.
func ExampleTrainColorCorrection() {
	ref := mcc.ReferenceRGB(mcc.Macbeth24)
	measured := make([][3]float64, len(ref))
	for i, c := range ref {
		measured[i] = [3]float64{c[0]*0.9 + 4, c[1]*0.95 + 3, c[2]*0.88 + 5}
	}
	before := 0.0
	for i := range ref {
		before += mcc.DeltaE2000(
			mcc.RGBToLab(uint8(measured[i][0]), uint8(measured[i][1]), uint8(measured[i][2])),
			mcc.RGBToLab(uint8(ref[i][0]), uint8(ref[i][1]), uint8(ref[i][2])),
		)
	}
	before /= float64(len(ref))

	model, err := mcc.TrainColorCorrection(measured, ref, mcc.ColorCorrectionConfig{Model: mcc.ModelRootPoly2})
	if err != nil {
		panic(err)
	}
	fmt.Printf("improved=%v\n", model.MeanDeltaE2000(measured, ref) < before)
	// Output: improved=true
}

// ExampleDigitalSGReference reports the size of the 140-patch ColorChecker
// Digital SG reference chart.
func ExampleDigitalSGReference() {
	ref := mcc.DigitalSGReference()
	fmt.Printf("%d patches, %dx%d\n", len(ref), mcc.DigitalSGRows(), mcc.DigitalSGCols())
	// Output: 140 patches, 10x14
}
