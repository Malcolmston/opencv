package stereo_test

import (
	"fmt"

	cv "github.com/malcolmston/opencv"
	"github.com/malcolmston/opencv/stereo"
)

// shiftedPair builds a rectified pair whose right half is shifted right by disp,
// so matchers should recover that disparity there.
func shiftedPair(w, h, disp int) (left, right *cv.Mat) {
	tex := func(x, y int) uint8 { return uint8((x*167 + y*83 + (x*x)%91) % 256) }
	right = cv.NewMat(h, w, 1)
	left = cv.NewMat(h, w, 1)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			right.Data[y*w+x] = tex(x, y)
			sx := x
			if x >= w/2 {
				sx = x - disp
			}
			if sx < 0 {
				sx = 0
			}
			left.Data[y*w+x] = tex(sx, y)
		}
	}
	return left, right
}

// ExampleStereoSGM recovers a disparity with full eight-path aggregation and a
// left-right consistency check.
func ExampleStereoSGM() {
	left, right := shiftedPair(64, 24, 8)
	sg := stereo.StereoSGM{NumDisparities: 16, BlockSize: 5, Mode: stereo.ModeHH, Disp12MaxDiff: 2}
	d := sg.Compute(left, right)
	fmt.Println(d.Data[12*d.Cols+50])
	// Output: 8
}

// ExampleMatchingCostVolume builds a cost volume and decodes it with
// winner-take-all.
func ExampleMatchingCostVolume() {
	left, right := shiftedPair(64, 24, 8)
	vol := stereo.MatchingCostVolume(left, right, 0, 16, 5, stereo.CostSAD)
	d := vol.WinnerTakeAll()
	fmt.Println(d.Data[12*d.Cols+50])
	// Output: 8
}

// ExampleCensusTransform shows that a lone bright pixel has all census
// comparison bits set relative to its darker neighbours.
func ExampleCensusTransform() {
	m := cv.NewMat(3, 3, 1)
	for i := range m.Data {
		m.Data[i] = 5
	}
	m.Data[4] = 250 // centre pixel
	codes, _, _ := stereo.CensusTransform(m, 3, 3)
	fmt.Println(stereo.HammingDistance64(codes[4], 0))
	// Output: 8
}

// ExampleSubpixelParabola interpolates the vertex of a symmetric cost triple.
func ExampleSubpixelParabola() {
	// A minimum skewed toward the lower neighbour.
	fmt.Printf("%.2f\n", stereo.SubpixelParabola(2, 0, 6))
	// Output: -0.25
}

// ExampleGetValidDisparityROI trims the overlap of two rectification ROIs.
func ExampleGetValidDisparityROI() {
	r := stereo.GetValidDisparityROI(
		stereo.Rect{X: 0, Y: 0, Width: 100, Height: 100},
		stereo.Rect{X: 0, Y: 0, Width: 100, Height: 100},
		0, 16, 5)
	fmt.Printf("%d %d %d %d\n", r.X, r.Y, r.Width, r.Height)
	// Output: 17 2 81 96
}

// ExampleValidateDisparity rejects a left pixel whose right-view disparity
// disagrees.
func ExampleValidateDisparity() {
	dl := cv.NewMat(1, 8, 1)
	dr := cv.NewMat(1, 8, 1)
	dl.Data[6] = 4 // left x=6 claims disparity 4 -> right x=2
	dr.Data[2] = 9 // right disagrees badly
	out := stereo.ValidateDisparity(dl, dr, 1, stereo.InvalidDisparity)
	fmt.Println(out.Data[6])
	// Output: 0
}

// ExampleDisparityWLSFilter fills an invalid hole from its surroundings.
func ExampleDisparityWLSFilter() {
	const w, h = 12, 12
	disp := cv.NewMat(h, w, 1)
	guide := cv.NewMat(h, w, 1)
	for i := range disp.Data {
		disp.Data[i] = 10
		guide.Data[i] = 128
	}
	disp.Data[6*w+6] = stereo.InvalidDisparity // a hole
	out := stereo.DisparityWLSFilter{Lambda: 2, SigmaColor: 10, Iterations: 60}.Filter(disp, guide)
	fmt.Println(out.Data[6*w+6])
	// Output: 10
}

// ExampleBlockMatcher runs the full block-matching pipeline (pre-filter,
// uniqueness, speckle removal, left-right check).
func ExampleBlockMatcher() {
	left, right := shiftedPair(64, 24, 8)
	bm := stereo.BlockMatcher{
		NumDisparities: 16,
		BlockSize:      7,
		PreFilterType:  stereo.PrefilterXSobel,
		Disp12MaxDiff:  2,
	}
	d := bm.Compute(left, right)
	fmt.Println(d.Data[12*d.Cols+50])
	// Output: 8
}

// ExampleQuasiDenseStereo grows dense matches from correlation seeds.
func ExampleQuasiDenseStereo() {
	left, right := shiftedPair(64, 24, 8)
	q := stereo.QuasiDenseStereo{NumDisparities: 16, CorrWinSize: 5, CorrThreshold: 0.6, DisparityGradient: 2}
	d := q.Process(left, right)
	fmt.Println(d.Data[12*d.Cols+50])
	// Output: 8
}
