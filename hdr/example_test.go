package hdr_test

import (
	"fmt"
	"math"

	cv "github.com/malcolmston/opencv"
	"github.com/malcolmston/opencv/hdr"
)

// makeBracket renders a tiny 3-exposure bracket of a column ramp scene through
// a gamma response, for use by the examples.
func makeBracket() ([]*cv.Mat, []float64) {
	const rows, cols = 8, 16
	times := []float64{0.25, 1, 4}
	imgs := make([]*cv.Mat, len(times))
	for j, t := range times {
		m := cv.NewMat(rows, cols, 3)
		for y := 0; y < rows; y++ {
			for x := 0; x < cols; x++ {
				e := 0.1 * math.Pow(30, float64(x)/float64(cols-1))
				xn := math.Min((e*t)/6.0, 1)
				z := uint8(math.Round(math.Pow(xn, 1.0/2.2) * 255))
				i := (y*cols + x) * 3
				m.Data[i], m.Data[i+1], m.Data[i+2] = z, z, z
			}
		}
		imgs[j] = m
	}
	return imgs, times
}

// ExampleCalibrateDebevec recovers the camera response and reports that it is
// monotonic.
func ExampleCalibrateDebevec() {
	imgs, times := makeBracket()
	resp, err := hdr.CalibrateDebevec(imgs, times, 0, 0)
	if err != nil {
		panic(err)
	}
	monotone := true
	for z := 40; z <= 210; z++ {
		if resp.Curve[0][z] < resp.Curve[0][z-1]-1e-6 {
			monotone = false
			break
		}
	}
	fmt.Printf("channels=%d entries=%d monotonic=%v\n", resp.Channels, len(resp.Curve[0]), monotone)
	// Output: channels=3 entries=256 monotonic=true
}

// ExampleMergeDebevec merges a bracket into a linear radiance map and tonemaps
// it with the Reinhard operator.
func ExampleMergeDebevec() {
	imgs, times := makeBracket()
	resp, _ := hdr.CalibrateDebevec(imgs, times, 0, 0)
	radiance, _ := hdr.MergeDebevec(imgs, times, resp)
	ldr := hdr.NewTonemapReinhard().Process(radiance)
	fmt.Printf("radiance %dx%dx%d -> ldr %dx%dx%d\n",
		radiance.Rows, radiance.Cols, radiance.Channels, ldr.Rows, ldr.Cols, ldr.Channels)
	// Output: radiance 8x16x3 -> ldr 8x16x3
}

// ExampleMergeMertens fuses a bracket directly into a displayable image with no
// exposure times or camera response.
func ExampleMergeMertens() {
	imgs, _ := makeBracket()
	fused, err := hdr.MergeMertens(imgs, hdr.NewMergeMertensParams())
	if err != nil {
		panic(err)
	}
	fmt.Printf("fused %dx%dx%d\n", fused.Rows, fused.Cols, fused.Channels)
	// Output: fused 8x16x3
}
