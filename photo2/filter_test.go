package photo2

import (
	"math"
	"testing"

	cv "github.com/malcolmston/opencv"
)

func TestBilateralFilterConstant(t *testing.T) {
	img := constRGB(7, 7, 100, 120, 140)
	out := BilateralFilter(img, 5, 20, 3)
	for i := range img.Data {
		if absDiff(img.Data[i], out.Data[i]) > 1 {
			t.Fatalf("bilateral of constant changed pixel")
		}
	}
}

func TestBilateralPreservesEdge(t *testing.T) {
	// A sharp vertical step edge should survive bilateral filtering better than
	// a plain box blur (the mean across the edge stays near the two levels).
	rows, cols := 9, 8
	img := cv.NewMat(rows, cols, 1)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			if x < cols/2 {
				img.Data[y*cols+x] = 40
			} else {
				img.Data[y*cols+x] = 220
			}
		}
	}
	out := BilateralFilter(img, 5, 15, 3)
	// Column far from the edge stays near its level.
	if absDiff(out.Data[4*cols+0], 40) > 5 {
		t.Fatalf("bilateral shifted flat dark region: %d", out.Data[4*cols+0])
	}
	if absDiff(out.Data[4*cols+cols-1], 220) > 5 {
		t.Fatalf("bilateral shifted flat bright region: %d", out.Data[4*cols+cols-1])
	}
}

func TestDomainTransformConstant(t *testing.T) {
	img := constRGB(8, 8, 70, 80, 90)
	for _, mode := range []EdgeFilterMode{RecursiveFilter, NormalizedConvolution} {
		out := DomainTransformFilter(img, mode, 30, 20)
		for i := range img.Data {
			if absDiff(img.Data[i], out.Data[i]) > 1 {
				t.Fatalf("domain transform of constant changed pixel (mode %d)", mode)
			}
		}
	}
}

func TestEdgePreservingSmooths(t *testing.T) {
	// A noisy flat region should become flatter (lower variance) after filtering.
	rows, cols := 12, 12
	img := cv.NewMat(rows, cols, 3)
	for i := 0; i < rows*cols; i++ {
		v := uint8(120)
		if i%2 == 0 {
			v = 130
		}
		img.Data[i*3+0], img.Data[i*3+1], img.Data[i*3+2] = v, v, v
	}
	out := EdgePreservingFilter(img, 40, 30)
	variance := func(m *cv.Mat) float64 {
		var mean, v float64
		n := len(m.Data)
		for _, s := range m.Data {
			mean += float64(s)
		}
		mean /= float64(n)
		for _, s := range m.Data {
			d := float64(s) - mean
			v += d * d
		}
		return v / float64(n)
	}
	if variance(out) >= variance(img) {
		t.Fatalf("edge-preserving filter did not reduce variance")
	}
}

func TestGuidedFilterConstant(t *testing.T) {
	f := cv.NewFloatMat(6, 6)
	for i := range f.Data {
		f.Data[i] = 0.5
	}
	out := GuidedFilter(f, f, 2, 0.01)
	for i := range f.Data {
		if math.Abs(out.Data[i]-0.5) > 1e-6 {
			t.Fatalf("guided filter of constant changed value: %v", out.Data[i])
		}
	}
}

func TestGuidedFilterSelfEdge(t *testing.T) {
	// Self-guided guided filter with tiny eps should nearly reproduce a step.
	rows, cols := 8, 8
	f := cv.NewFloatMat(rows, cols)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			if x < cols/2 {
				f.Data[y*cols+x] = 0.2
			} else {
				f.Data[y*cols+x] = 0.9
			}
		}
	}
	out := GuidedFilter(f, f, 2, 1e-6)
	// Interior of each region preserved.
	if math.Abs(out.At(4, 0)-0.2) > 0.02 {
		t.Fatalf("guided self-filter shifted dark region: %v", out.At(4, 0))
	}
	if math.Abs(out.At(4, cols-1)-0.9) > 0.02 {
		t.Fatalf("guided self-filter shifted bright region: %v", out.At(4, cols-1))
	}
}

func TestStylizeShapes(t *testing.T) {
	img := cv.NewMat(10, 10, 3)
	for i := range img.Data {
		img.Data[i] = uint8((i * 3) % 256)
	}
	if got := DetailEnhance(img, 30, 20, 3); got.Rows != 10 || got.Cols != 10 || got.Channels != 3 {
		t.Fatalf("DetailEnhance shape wrong")
	}
	if got := Stylization(img, 40, 30); got.Channels != 3 {
		t.Fatalf("Stylization shape wrong")
	}
	if got := Cartoon(img, 40, 30); got.Channels != 3 {
		t.Fatalf("Cartoon shape wrong")
	}
	gray, color := PencilSketch(img, 3, 0.07, 0.02)
	if gray.Channels != 1 || gray.Rows != 10 {
		t.Fatalf("PencilSketch gray shape wrong")
	}
	if color.Channels != 3 || color.Rows != 10 {
		t.Fatalf("PencilSketch color shape wrong")
	}
}

func TestDetailEnhanceConstant(t *testing.T) {
	img := constRGB(8, 8, 100, 110, 120)
	out := DetailEnhance(img, 30, 20, 3)
	for i := range img.Data {
		if absDiff(img.Data[i], out.Data[i]) > 1 {
			t.Fatalf("detail enhance of constant changed pixel")
		}
	}
}

func BenchmarkMertensFusion(b *testing.B) {
	rows, cols := 64, 64
	mk := func(off int) *cv.Mat {
		m := cv.NewMat(rows, cols, 3)
		for i := range m.Data {
			m.Data[i] = uint8((i + off) % 256)
		}
		return m
	}
	imgs := []*cv.Mat{mk(0), mk(64), mk(128)}
	params := DefaultMertensParams()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = MertensFusion(imgs, params)
	}
}
