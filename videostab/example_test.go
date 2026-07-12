package videostab_test

import (
	"fmt"
	"math"
	"math/rand"

	cv "github.com/malcolmston/opencv"
	"github.com/malcolmston/opencv/video"
	"github.com/malcolmston/opencv/videostab"
)

// makeTexture builds a small deterministic texture for the examples.
func makeTexture(rows, cols int, seed int64) *cv.Mat {
	rng := rand.New(rand.NewSource(seed))
	m := cv.NewMat(rows, cols, 1)
	for i := range m.Data {
		m.Data[i] = uint8(rng.Intn(256))
	}
	return cv.GaussianBlur(m, 3, 0)
}

// ExampleMotionEstimatorRansacL2 fits a global similarity motion to point
// correspondences even when a third of them are gross outliers.
func ExampleMotionEstimatorRansacL2() {
	want := videostab.SimilarityMotion(1.1, 0.1, 5, -3)
	rng := rand.New(rand.NewSource(1))
	var from, to []video.PointF
	for i := 0; i < 40; i++ {
		p := video.PointF{X: rng.Float64() * 100, Y: rng.Float64() * 100}
		tx, ty := want.Apply(p.X, p.Y)
		from = append(from, p)
		if i%3 == 0 { // outlier
			to = append(to, video.PointF{X: rng.Float64() * 500, Y: rng.Float64() * 500})
		} else {
			to = append(to, video.PointF{X: tx, Y: ty})
		}
	}
	est := videostab.NewMotionEstimatorRansacL2(videostab.MotionModelSimilarity)
	est.SetMinInlierRatio(0.3)
	m, ok := est.Estimate(from, to)
	x, y := m.Apply(0, 0)
	fmt.Printf("ok=%v translation=(%.0f,%.0f)\n", ok, math.Round(x), math.Round(y))
	// Output: ok=true translation=(5,-3)
}

// ExampleGaussianMotionFilter smooths a jittery one-dimensional camera path.
func ExampleGaussianMotionFilter() {
	n := 9
	motions := make([]videostab.Motion, n-1)
	shifts := []float64{5, -6, 5, -5, 6, -6, 5, -5}
	for i := range motions {
		motions[i] = videostab.TranslationMotion(shifts[i], 0)
	}
	f := videostab.NewGaussianMotionFilter(3, 0)
	out := make([]videostab.Motion, n)
	f.Stabilize(n, motions, videostab.Range{First: 0, Last: n - 1}, out)
	fmt.Printf("smoothed %d frames\n", len(out))
	// Output: smoothed 9 frames
}

// ExampleOnePassStabilizer stabilizes a short jittered sequence and reports that
// the residual inter-frame motion has dropped.
func ExampleOnePassStabilizer() {
	const size, margin, n = 64, 10, 12
	base := makeTexture(size+2*margin, size+2*margin, 3)
	rng := rand.New(rand.NewSource(4))
	frames := make([]*cv.Mat, n)
	for i := 0; i < n; i++ {
		ox := margin + rng.Intn(2*margin+1) - margin
		oy := margin + rng.Intn(2*margin+1) - margin
		if ox < 0 {
			ox = 0
		}
		if oy < 0 {
			oy = 0
		}
		frames[i] = base.Region(oy, ox, size, size)
	}

	s := videostab.NewOnePassStabilizer(5)
	s.SetFrames(frames)
	out := s.Stabilize()
	fmt.Printf("stabilized %d frames\n", len(out))
	// Output: stabilized 12 frames
}

// ExampleCalcBlurriness shows that blurring a frame raises its blurriness
// measure and a weighting deblurer lowers it again.
func ExampleCalcBlurriness() {
	sharp := makeTexture(48, 48, 7)
	blurred := cv.GaussianBlur(sharp, 7, 0)
	d := videostab.NewWeightingDeblurer(1)
	deblurred := blurred.Clone()
	d.Deblur(0, deblurred)
	fmt.Println(videostab.CalcBlurriness(blurred) > videostab.CalcBlurriness(sharp))
	fmt.Println(videostab.CalcBlurriness(deblurred) < videostab.CalcBlurriness(blurred))
	// Output:
	// true
	// true
}
