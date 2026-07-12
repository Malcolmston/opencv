package hdr_test

import (
	"bytes"
	"fmt"
	"math"

	cv "github.com/malcolmston/opencv"
	"github.com/malcolmston/opencv/hdr"
)

// ExampleAlignMTB recovers a known translation between two frames with the
// median-threshold-bitmap aligner.
func ExampleAlignMTB() {
	// Build a small textured scene and two windows into it offset by (2, 1).
	scene := cv.NewMat(24, 24, 1)
	for y := 0; y < 24; y++ {
		for x := 0; x < 24; x++ {
			scene.Data[y*24+x] = uint8((x*13 + y*7) % 256)
		}
	}
	crop := func(oy, ox int) *cv.Mat {
		m := cv.NewMat(16, 16, 1)
		for y := 0; y < 16; y++ {
			for x := 0; x < 16; x++ {
				m.Data[y*16+x] = scene.Data[(oy+y)*24+(ox+x)]
			}
		}
		return m
	}
	ref := crop(4, 4)
	src := crop(5, 6) // offset by dx=2, dy=1

	a := hdr.NewAlignMTB(3)
	dx, dy := a.CalculateShift(ref, src)
	fmt.Printf("recovered shift dx=%d dy=%d\n", dx, dy)
	// Output: recovered shift dx=2 dy=1
}

// ExampleTonemapDurand tone maps a wide-range radiance map with the bilateral
// Durand operator.
func ExampleTonemapDurand() {
	imgs, times := makeBracket()
	resp, _ := hdr.CalibrateDebevec(imgs, times, 0, 0)
	radiance, _ := hdr.MergeDebevec(imgs, times, resp)
	ldr := hdr.NewTonemapDurand().Process(radiance)
	fmt.Printf("durand ldr %dx%dx%d\n", ldr.Rows, ldr.Cols, ldr.Channels)
	// Output: durand ldr 8x16x3
}

// ExampleMergeRobertson merges a bracket with Robertson's estimator paired with
// a Robertson-calibrated response.
func ExampleMergeRobertson() {
	imgs, times := makeBracket()
	resp, _ := hdr.CalibrateRobertson(imgs, times, 0, 0)
	radiance, _ := hdr.MergeRobertson(imgs, times, resp)
	fmt.Printf("robertson radiance %dx%dx%d\n", radiance.Rows, radiance.Cols, radiance.Channels)
	// Output: robertson radiance 8x16x3
}

// ExampleApplyColorMap false-colours the luminance of a radiance map for
// inspection.
func ExampleApplyColorMap() {
	imgs, times := makeBracket()
	resp, _ := hdr.CalibrateDebevec(imgs, times, 0, 0)
	radiance, _ := hdr.MergeDebevec(imgs, times, resp)
	lum := radiance.LuminanceFloatMat()
	viz := hdr.ApplyColorMap(lum, hdr.ColorMapJet)
	fmt.Printf("false-colour %dx%dx%d\n", viz.Rows, viz.Cols, viz.Channels)
	// Output: false-colour 8x16x3
}

// ExampleWritePFM round-trips a radiance map through the PFM float-image format.
func ExampleWritePFM() {
	r := hdr.NewRadiance(2, 2, 3)
	for i := range r.Data {
		r.Data[i] = float64(i) * 0.25
	}
	var buf bytes.Buffer
	if err := hdr.WritePFM(&buf, r); err != nil {
		panic(err)
	}
	back, err := hdr.ReadPFM(&buf)
	if err != nil {
		panic(err)
	}
	fmt.Printf("round-trip %dx%dx%d equal=%v\n",
		back.Rows, back.Cols, back.Channels, math.Abs(back.At(1, 1, 2)-r.At(1, 1, 2)) < 1e-6)
	// Output: round-trip 2x2x3 equal=true
}

// ExampleDetailEnhance sharpens local detail on a tonemapped image.
func ExampleDetailEnhance() {
	imgs, _ := makeBracket()
	fused, _ := hdr.MergeMertens(imgs, hdr.NewMergeMertensParams())
	enhanced := hdr.DetailEnhance(fused, 3, 0.15, 2)
	fmt.Printf("enhanced %dx%dx%d\n", enhanced.Rows, enhanced.Cols, enhanced.Channels)
	// Output: enhanced 8x16x3
}
