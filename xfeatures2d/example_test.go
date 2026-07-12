package xfeatures2d_test

import (
	"fmt"

	cv "github.com/malcolmston/opencv"
	"github.com/malcolmston/opencv/xfeatures2d"
)

// ExampleSimpleBlobDetector detects filled circles drawn on a white background.
func ExampleSimpleBlobDetector() {
	img := cv.NewMat(100, 100, 1)
	img.SetTo(255)
	for _, c := range []cv.Point{{X: 30, Y: 30}, {X: 70, Y: 70}} {
		cv.Circle(img, c, 11, cv.NewScalar(0), cv.Filled)
	}

	det := xfeatures2d.NewSimpleBlobDetector()
	blobs := det.Detect(img)

	fmt.Println("blobs:", len(blobs))
	// Output:
	// blobs: 2
}

// ExampleAGAST finds corners with the adaptive-threshold FAST segment test.
func ExampleAGAST() {
	// A white square on a black background: its four convex corners are strong
	// FAST/AGAST features.
	img := cv.NewMat(60, 60, 1)
	for y := 20; y < 40; y++ {
		for x := 20; x < 40; x++ {
			img.Data[y*60+x] = 255
		}
	}

	kps := xfeatures2d.NewAGAST(30).Detect(img)
	fmt.Println("found corners:", len(kps) > 0)
	// Output:
	// found corners: true
}

// ExampleGFTTDetector wraps the Shi–Tomasi corner detector.
func ExampleGFTTDetector() {
	img := cv.NewMat(80, 80, 1)
	for y := 0; y < 80; y++ {
		for x := 0; x < 80; x++ {
			if (x/20+y/20)%2 == 0 {
				img.Data[y*80+x] = 255
			}
		}
	}

	det := xfeatures2d.NewGFTTDetector(10)
	kps := det.Detect(img)
	fmt.Println("at most 10 corners:", len(kps) <= 10 && len(kps) > 0)
	// Output:
	// at most 10 corners: true
}

// ExampleBRISK_Compute shows that two identical neighbourhoods produce the same
// binary descriptor (Hamming distance 0).
func ExampleBRISK_Compute() {
	img := cv.NewMat(120, 120, 1)
	for y := 0; y < 120; y++ {
		for x := 0; x < 120; x++ {
			img.Data[y*120+x] = uint8((x*3 + y) % 200)
		}
	}
	// Stamp an identical 40×40 patch at two locations.
	for _, c := range []cv.Point{{X: 30, Y: 60}, {X: 90, Y: 60}} {
		for i := 0; i < 40; i++ {
			for j := 0; j < 40; j++ {
				img.Data[(c.Y-20+i)*120+(c.X-20+j)] = uint8((i*9 + j*5) % 256)
			}
		}
	}

	b := xfeatures2d.NewBRISK(20)
	_, descs := b.Compute(img, []xfeatures2d.KeyPoint{
		{Pt: cv.Point{X: 30, Y: 60}},
		{Pt: cv.Point{X: 90, Y: 60}},
	})
	fmt.Println("hamming:", xfeatures2d.HammingDistance(descs[0], descs[1]))
	// Output:
	// hamming: 0
}
