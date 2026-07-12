package stitching_test

import (
	"fmt"

	cv "github.com/malcolmston/opencv"
	"github.com/malcolmston/opencv/stitching"
)

// makeStripe builds a small grayscale image textured with contrasting blocks of
// position-dependent intensity, so every neighbourhood is distinct and the
// example has unambiguous corners to detect and match.
func makeStripe(cols int, shade uint8) *cv.Mat {
	m := cv.NewMat(40, cols, 1)
	for i := range m.Data {
		m.Data[i] = shade
	}
	bi := 0
	for by := 4; by < 36; by += 6 {
		for bx := 3; bx < cols-4; bx += 6 {
			bi++
			v := uint8(30 + (bi*53)%200)
			for y := by; y < by+3 && y < 40; y++ {
				for x := bx; x < bx+3 && x < cols; x++ {
					m.Data[y*cols+x] = v
				}
			}
		}
	}
	return m
}

// ExampleStitcher_Stitch stitches two overlapping crops of one image back into a
// panorama and reports its size.
func ExampleStitcher_Stitch() {
	base := makeStripe(120, 60)
	left := base.Region(0, 0, 40, 80)   // columns [0,80)
	right := base.Region(0, 40, 40, 80) // columns [40,120)

	s := stitching.NewStitcher()
	pano, err := s.Stitch([]*cv.Mat{left, right})
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	fmt.Printf("panorama %dx%d\n", pano.Cols, pano.Rows)
	// Output: panorama 120x40
}

// ExampleStitcher_EstimateTransform recovers the homography relating two crops;
// here a pure horizontal translation.
func ExampleStitcher_EstimateTransform() {
	base := makeStripe(120, 50)
	left := base.Region(0, 0, 40, 80)
	right := base.Region(0, 40, 40, 80)

	s := stitching.NewStitcher()
	h, err := s.EstimateTransform(left, right)
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	// h maps the right image into the left image's frame: a +40 px shift in x.
	fmt.Printf("dx=%.0f dy=%.0f\n", h[2], h[5])
	// Output: dx=40 dy=0
}

// ExampleMultiBandBlend selects the multi-band blender instead of the default
// feather blender.
func ExampleMultiBandBlend() {
	base := makeStripe(120, 70)
	left := base.Region(0, 0, 40, 80)
	right := base.Region(0, 40, 40, 80)

	s := stitching.NewStitcher()
	s.Blender = stitching.MultiBandBlend{Bands: 3}
	pano, err := s.Stitch([]*cv.Mat{left, right})
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	fmt.Printf("panorama %dx%d\n", pano.Cols, pano.Rows)
	// Output: panorama 120x40
}
