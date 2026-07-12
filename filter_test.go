package cv

import "testing"

func grayFromValues(rows, cols int, vals []uint8) *Mat {
	m := NewMat(rows, cols, 1)
	copy(m.Data, vals)
	return m
}

func TestFilter2DHandComputed(t *testing.T) {
	// 3x3 image:
	//  10 20 30
	//  40 50 60
	//  70 80 90
	m := grayFromValues(3, 3, []uint8{10, 20, 30, 40, 50, 60, 70, 80, 90})
	// Vertical-edge kernel:
	//   1 0 -1
	//   1 0 -1
	//   1 0 -1
	k := NewKernel(3, 3, []float64{1, 0, -1, 1, 0, -1, 1, 0, -1})
	// Center pixel (1,1): (10-30)+(40-60)+(70-90) = -60, +delta 128 -> 68.
	out := Filter2D(m, k, 128)
	if got := out.At(1, 1, 0); got != 68 {
		t.Errorf("Filter2D center = %d, want 68", got)
	}
}

func TestBlurMeanCenter(t *testing.T) {
	m := grayFromValues(3, 3, []uint8{10, 20, 30, 40, 50, 60, 70, 80, 90})
	// 3x3 mean of all values = 450/9 = 50.
	out := Blur(m, 3)
	if got := out.At(1, 1, 0); got != 50 {
		t.Errorf("Blur center = %d, want 50", got)
	}
}

func TestBoxFilterUnnormalizedSaturates(t *testing.T) {
	m := grayFromValues(3, 3, []uint8{10, 20, 30, 40, 50, 60, 70, 80, 90})
	out := BoxFilter(m, 3, false) // sum = 450 -> clamps to 255
	if out.At(1, 1, 0) != 255 {
		t.Errorf("unnormalized box center = %d, want 255", out.At(1, 1, 0))
	}
}

func TestGaussianKernelNormalised(t *testing.T) {
	k := GaussianKernel1D(5, 1.0)
	var sum float64
	for _, v := range k {
		sum += v
	}
	if sum < 0.999 || sum > 1.001 {
		t.Errorf("gaussian kernel sum = %v, want ~1", sum)
	}
	// Symmetric.
	if k[0] != k[4] || k[1] != k[3] {
		t.Error("gaussian kernel not symmetric")
	}
	// Peaked in the middle.
	if k[2] <= k[1] {
		t.Error("gaussian kernel not peaked at centre")
	}
}

func TestGaussianBlurPreservesConstant(t *testing.T) {
	m := NewMat(5, 5, 1)
	m.SetTo(100)
	out := GaussianBlur(m, 3, 0)
	for _, v := range out.Data {
		if v != 100 {
			t.Fatalf("gaussian blur of constant changed value to %d", v)
		}
	}
}

func TestMedianBlurRemovesSpike(t *testing.T) {
	m := NewMat(3, 3, 1)
	m.SetTo(50)
	m.Set(1, 1, 0, 255) // salt spike
	out := MedianBlur(m, 3)
	if out.At(1, 1, 0) != 50 {
		t.Errorf("median did not remove spike, got %d", out.At(1, 1, 0))
	}
}

func TestSobelVerticalEdge(t *testing.T) {
	// Left half dark, right half bright -> strong horizontal (dx) gradient.
	m := NewMat(5, 6, 1)
	for y := 0; y < 5; y++ {
		for x := 0; x < 6; x++ {
			if x >= 3 {
				m.Set(y, x, 0, 255)
			}
		}
	}
	gx := SobelFloat(m, 1, 0, 3)[0]
	// A column straddling the edge should have a large positive response.
	maxAbs := 0.0
	for _, v := range gx {
		if v > maxAbs {
			maxAbs = v
		}
	}
	if maxAbs < 500 {
		t.Errorf("expected strong sobel dx response, max = %v", maxAbs)
	}
}

func TestLaplacianConstantIsZero(t *testing.T) {
	m := NewMat(4, 4, 1)
	m.SetTo(80)
	out := Laplacian(m, 1, 1, 0)
	for _, v := range out.Data {
		if v != 0 {
			t.Fatalf("laplacian of constant = %d, want 0", v)
		}
	}
}
