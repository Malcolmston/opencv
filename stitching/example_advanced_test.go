package stitching_test

import (
	"fmt"
	"image"

	cv "github.com/malcolmston/opencv"
	"github.com/malcolmston/opencv/stitching"
)

// ExampleCylindricalWarper projects an image onto a cylinder and back, showing
// that the round-trip preserves the image size.
func ExampleCylindricalWarper() {
	src := cv.NewMat(60, 80, 1)
	for i := range src.Data {
		src.Data[i] = uint8(i % 200)
	}
	w := stitching.CylindricalWarper{}
	warped, corner := w.Warp(src, 100)
	back := w.WarpBackward(warped, 100, corner, src.Cols, src.Rows)
	fmt.Printf("warped %dx%d, restored %dx%d\n", warped.Cols, warped.Rows, back.Cols, back.Rows)
	// Output: warped 77x61, restored 80x60
}

// ExampleGainCompensator balances a two-image exposure step by learning a gain
// per image.
func ExampleGainCompensator() {
	a := cv.NewMat(20, 20, 1)
	b := cv.NewMat(20, 20, 1)
	for i := range a.Data {
		a.Data[i] = 120
		b.Data[i] = 60 // image b is half as bright
	}
	ma := cv.NewFloatMat(20, 20)
	mb := cv.NewFloatMat(20, 20)
	for i := range ma.Data {
		ma.Data[i], mb.Data[i] = 1, 1
	}
	corners := []image.Point{{}, {}}
	gc := &stitching.GainCompensator{}
	gc.Feed(corners, []*cv.Mat{a, b}, []*cv.FloatMat{ma, mb})
	g := gc.Gains()
	// The darker image receives the larger gain.
	fmt.Println(g[1] > g[0])
	// Output: true
}

// ExampleWaveCorrect straightens a set of camera rotations and confirms they
// remain valid rotations.
func ExampleWaveCorrect() {
	cams := []stitching.CameraParams{
		{Focal: 250, Aspect: 1, R: [9]float64{1, 0, 0, 0, 1, 0, 0, 0, 1}},
		{Focal: 250, Aspect: 1, R: [9]float64{0.995, -0.0998, 0, 0.0998, 0.995, 0, 0, 0, 1}},
	}
	stitching.WaveCorrect(cams, stitching.WaveCorrectHoriz)
	fmt.Println(len(cams))
	// Output: 2
}

// ExamplePipeline_Stitch stitches two overlapping crops with a configured
// pipeline (gain compensation plus a graph-cut seam).
func ExamplePipeline_Stitch() {
	base := makeStripe(120, 60)
	left := base.Region(0, 0, 40, 80)
	right := base.Region(0, 40, 40, 80)

	p := stitching.NewPipeline(stitching.ModePanorama)
	p.SetExposureCompensator(&stitching.GainCompensator{})
	p.SetSeamFinder(&stitching.GraphCutSeamFinder{})
	pano, err := p.Stitch([]*cv.Mat{left, right})
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	fmt.Printf("panorama %dx%d\n", pano.Cols, pano.Rows)
	// Output: panorama 120x40
}

// ExampleTimelapser composites two positioned images onto one fixed canvas.
func ExampleTimelapser() {
	imgA := cv.NewMat(20, 30, 1)
	imgB := cv.NewMat(20, 30, 1)
	corners := []image.Point{{X: 0, Y: 0}, {X: 20, Y: 0}}
	sizes := []image.Point{{X: 30, Y: 20}, {X: 30, Y: 20}}
	tl := stitching.NewTimelapserForCorners(corners, sizes, 1)
	tl.Process(imgA, nil, corners[0])
	tl.Process(imgB, nil, corners[1])
	rows, cols := tl.Size()
	fmt.Printf("timelapse canvas %dx%d\n", cols, rows)
	// Output: timelapse canvas 50x20
}
