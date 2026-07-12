package segmentation_test

import (
	"fmt"

	cv "github.com/malcolmston/opencv"
	"github.com/malcolmston/opencv/segmentation"
)

// quad builds a small four-quadrant colour image for the examples.
func quad() *cv.Mat {
	img := cv.NewMat(20, 20, 3)
	colors := [4][3]uint8{{220, 20, 20}, {20, 220, 20}, {20, 20, 220}, {220, 220, 20}}
	for y := 0; y < 20; y++ {
		for x := 0; x < 20; x++ {
			q := 0
			if x >= 10 {
				q |= 1
			}
			if y >= 10 {
				q |= 2
			}
			img.SetPixel(y, x, colors[q][:])
		}
	}
	return img
}

func ExampleEfficientGraphSegmentation() {
	lm := segmentation.EfficientGraphSegmentation(quad(), 0, 300, 1)
	fmt.Println(lm.Count)
	// Output: 4
}

func ExampleMultiOtsu() {
	// Three intensity bands at 30/130/220.
	img := cv.NewMat(30, 3, 1)
	for y := 0; y < 30; y++ {
		v := uint8(30)
		switch {
		case y >= 20:
			v = 220
		case y >= 10:
			v = 130
		}
		for x := 0; x < 3; x++ {
			img.Set(y, x, 0, v)
		}
	}
	thresholds := segmentation.MultiOtsu(img, 3)
	fmt.Println(len(thresholds))
	// Output: 2
}

func ExampleDistanceTransformWatershed() {
	// Two overlapping discs forming one connected blob with a thin waist.
	m := cv.NewMat(24, 30, 1)
	disc := func(cx, cy, r int) {
		for y := cy - r; y <= cy+r; y++ {
			for x := cx - r; x <= cx+r; x++ {
				if x >= 0 && x < 30 && y >= 0 && y < 24 && (x-cx)*(x-cx)+(y-cy)*(y-cy) <= r*r {
					m.Set(y, x, 0, 255)
				}
			}
		}
	}
	disc(8, 12, 6)
	disc(18, 12, 6)
	lm := segmentation.DistanceTransformWatershed(m, 0.7)
	// Background plus the two separated blobs.
	fmt.Println(lm.Count, lm.At(8, 12) != lm.At(18, 12))
	// Output: 3 true
}

func ExampleKMeansSegmentation() {
	// Three solid colour blocks.
	img := cv.NewMat(6, 6, 3)
	colors := [3][3]uint8{{200, 10, 10}, {10, 200, 10}, {10, 10, 200}}
	for y := 0; y < 6; y++ {
		for x := 0; x < 6; x++ {
			img.SetPixel(y, x, colors[x/2][:])
		}
	}
	lm, centers := segmentation.KMeansSegmentation(img, 3, 10)
	fmt.Println(lm.Count, len(centers))
	// Output: 3 3
}

func ExampleIntelligentScissors() {
	// A strong vertical edge at column 5.
	img := cv.NewMat(10, 10, 3)
	for y := 0; y < 10; y++ {
		for x := 0; x < 10; x++ {
			v := uint8(40)
			if x >= 5 {
				v = 210
			}
			img.SetPixel(y, x, []uint8{v, v, v})
		}
	}
	sc := segmentation.NewIntelligentScissors(img)
	sc.BuildMap(cv.Point{X: 5, Y: 0})
	path := sc.Trace(cv.Point{X: 5, Y: 9})
	fmt.Println(path[0], path[len(path)-1])
	// Output: {5 0} {5 9}
}

func ExampleRAG_MergeByColor() {
	// Two near-identical stripes and one distinct stripe.
	img := cv.NewMat(9, 9, 3)
	cols := [3][3]uint8{{200, 10, 10}, {188, 20, 18}, {10, 10, 210}}
	lm := &segmentation.LabelMap{Rows: 9, Cols: 9, Count: 3, Labels: make([]int, 81)}
	for y := 0; y < 9; y++ {
		for x := 0; x < 9; x++ {
			s := x / 3
			lm.Labels[y*9+x] = s
			img.SetPixel(y, x, cols[s][:])
		}
	}
	merged := segmentation.BuildRAG(lm, img).MergeByColor(40)
	fmt.Println(merged.Count)
	// Output: 2
}
