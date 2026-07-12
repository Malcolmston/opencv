package imgprocx_test

import (
	"fmt"

	cv "github.com/malcolmston/opencv"
	"github.com/malcolmston/opencv/imgprocx"
)

// ExampleGetAffineTransform derives the affine transform from three point
// correspondences and applies it with cv.WarpAffine.
func ExampleGetAffineTransform() {
	src := [3]cv.Point{{X: 0, Y: 0}, {X: 4, Y: 0}, {X: 0, Y: 4}}
	dst := [3]cv.Point{{X: 1, Y: 1}, {X: 9, Y: 1}, {X: 1, Y: 9}}
	m := imgprocx.GetAffineTransform(src, dst)
	// m maps the first source point exactly onto the first destination.
	p := imgprocx.ApplyAffine(imgprocx.FromAffineMatrix(m), src[1])
	fmt.Printf("(%.0f, %.0f)\n", p.X, p.Y)
	// Output: (9, 1)
}

// ExampleIntegralImage reads the sum over a rectangle from the summed-area
// table in constant time.
func ExampleIntegralImage() {
	img := cv.NewMat(4, 4, 1)
	img.SetTo(2) // every pixel intensity is 2
	sum, _ := imgprocx.IntegralImage(img)
	// Sum over the whole 4x4 image: 16 pixels * 2 = 32.
	fmt.Println(imgprocx.RectSum(sum, 0, 0, 4, 4))
	// Output: 32
}

// ExamplePhaseCorrelate recovers the translation between an image and a shifted
// copy of it.
func ExamplePhaseCorrelate() {
	rows, cols := 16, 16
	a := cv.NewMat(rows, cols, 1)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			a.Set(y, x, 0, uint8((x*13+y*7)%256))
		}
	}
	dx, dy := 3, 2
	b := cv.NewMat(rows, cols, 1)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			sy := ((y-dy)%rows + rows) % rows
			sx := ((x-dx)%cols + cols) % cols
			b.Set(y, x, 0, a.At(sy, sx, 0))
		}
	}
	shift, _ := imgprocx.PhaseCorrelate(a, b)
	fmt.Printf("shift = (%.0f, %.0f)\n", shift.X, shift.Y)
	// Output: shift = (3, 2)
}

// ExampleGetGaussianKernel returns the exact small-aperture Gaussian OpenCV
// uses when no sigma is supplied.
func ExampleGetGaussianKernel() {
	k := imgprocx.GetGaussianKernel(3, 0)
	fmt.Println(k)
	// Output: [0.25 0.5 0.25]
}

// ExampleGetDerivKernels shows the separable Sobel factors for a first
// derivative in x with a 3x3 aperture.
func ExampleGetDerivKernels() {
	kx, ky := imgprocx.GetDerivKernels(1, 0, 3, false)
	fmt.Println(kx, ky)
	// Output: [-1 0 1] [1 2 1]
}

// ExampleEMD computes the Earth Mover's Distance between two 1-D distributions
// with a |i-j| ground metric.
func ExampleEMD() {
	d := imgprocx.EMD(
		[]float64{0.4, 0.6},
		[]float64{0.5, 0.5},
		[][]float64{{0, 1}, {1, 0}},
	)
	fmt.Printf("%.1f\n", d)
	// Output: 0.1
}

// ExampleFloodFill fills the connected same-valued region around a seed and
// reports how many pixels it painted.
func ExampleFloodFill() {
	m := cv.NewMat(2, 4, 1)
	for x := 0; x < 4; x++ {
		v := uint8(10)
		if x >= 2 {
			v = 200
		}
		m.Set(0, x, 0, v)
		m.Set(1, x, 0, v)
	}
	area, rect := imgprocx.FloodFill(m, cv.Point{X: 0, Y: 0}, cv.Scalar{99},
		&imgprocx.FloodFillOptions{FixedRange: true})
	fmt.Printf("area=%d rect=%dx%d\n", area, rect.Width, rect.Height)
	// Output: area=4 rect=2x2
}

// ExampleCreateHanningWindow builds a small raised-cosine window; its border is
// zero and its centre is one.
func ExampleCreateHanningWindow() {
	w := imgprocx.CreateHanningWindow(5, 5)
	fmt.Printf("corner=%.0f centre=%.0f\n", w.At(0, 0), w.At(2, 2))
	// Output: corner=0 centre=1
}
