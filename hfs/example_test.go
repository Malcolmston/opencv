package hfs_test

import (
	"fmt"

	cv "github.com/malcolmston/opencv"
	"github.com/malcolmston/opencv/hfs"
)

// solidQuadrants builds a size x size RGB image of four distinct solid colours.
func solidQuadrants(size int) *cv.Mat {
	m := cv.NewMat(size, size, 3)
	half := size / 2
	colors := [4][3]uint8{
		{220, 30, 30}, {30, 200, 30}, {30, 30, 220}, {230, 220, 40},
	}
	for y := 0; y < size; y++ {
		for x := 0; x < size; x++ {
			q := 0
			if x >= half {
				q++
			}
			if y >= half {
				q += 2
			}
			m.SetPixel(y, x, colors[q][:])
		}
	}
	return m
}

// ExampleHfsSegment_PerformSegmentCpu segments a four-colour image and reports
// how many regions the pipeline recovered.
func ExampleHfsSegment_PerformSegmentCpu() {
	img := solidQuadrants(64)
	seg := hfs.CreateWithDefaults(img.Rows, img.Cols)
	drawn := seg.PerformSegmentCpu(img, true)
	fmt.Printf("regions=%d channels=%d\n", seg.NumSegments(), drawn.Channels)
	// Output: regions=4 channels=3
}

// ExampleHfsSegment_DrawSegmentation renders the segmentation with average
// region colours and prints the colour recovered for the top-left quadrant.
func ExampleHfsSegment_DrawSegmentation() {
	img := solidQuadrants(64)
	seg := hfs.CreateWithDefaults(img.Rows, img.Cols)
	seg.PerformSegmentCpu(img, false)
	avg := seg.DrawSegmentation(hfs.DrawAverageColor)
	fmt.Printf("%d %d %d\n", avg.At(16, 16, 0), avg.At(16, 16, 1), avg.At(16, 16, 2))
	// Output: 220 30 30
}

// ExampleHfsSegment_Labels shows how to read the dense region labelling.
func ExampleHfsSegment_Labels() {
	img := solidQuadrants(48)
	seg := hfs.CreateWithDefaults(img.Rows, img.Cols)
	seg.SetSlicSpixelSize(8)
	seg.PerformSegmentCpu(img, false)
	labels, rows, cols := seg.Labels()
	// The four quadrant centres carry four different labels.
	tl := labels[12*cols+12]
	br := labels[36*cols+36]
	fmt.Printf("%dx%d distinct=%v corners_differ=%v\n", rows, cols, seg.NumSegments(), tl != br)
	// Output: 48x48 distinct=4 corners_differ=true
}
