package stitch

import (
	"math"
	"testing"

	cv "github.com/malcolmston/opencv"
)

func TestFeatherWeightMap(t *testing.T) {
	m := FeatherWeightMap(7, 5, 1)
	// Centre weight is the maximum and strictly positive everywhere.
	center := m.At(2, 3)
	for _, v := range m.Data {
		if v <= 0 {
			t.Fatal("feather weight must be strictly positive")
		}
		if v > center+1e-12 {
			t.Fatalf("centre weight %g is not the maximum (found %g)", center, v)
		}
	}
	// Symmetric about the centre column.
	if math.Abs(m.At(2, 2)-m.At(2, 4)) > 1e-12 {
		t.Fatal("feather map should be horizontally symmetric")
	}
}

func TestLaplacianPyramidReconstruction(t *testing.T) {
	src := cv.NewFloatMat(16, 20)
	for y := 0; y < 16; y++ {
		for x := 0; x < 20; x++ {
			src.Data[y*20+x] = float64((x*13 + y*7) % 100)
		}
	}
	lap := BuildLaplacianPyramid(src, 3)
	rec := CollapseLaplacianPyramid(lap)
	if rec.Rows != src.Rows || rec.Cols != src.Cols {
		t.Fatalf("reconstructed size %dx%d", rec.Rows, rec.Cols)
	}
	var maxErr float64
	for i := range src.Data {
		if d := math.Abs(src.Data[i] - rec.Data[i]); d > maxErr {
			maxErr = d
		}
	}
	if maxErr > 1e-9 {
		t.Fatalf("reconstruction error %g too large", maxErr)
	}
}

func TestPyrDownSize(t *testing.T) {
	src := cv.NewFloatMat(9, 7)
	d := PyrDownFloat(src)
	if d.Rows != 5 || d.Cols != 4 {
		t.Fatalf("pyrDown size = %dx%d, want 5x4", d.Rows, d.Cols)
	}
}

func fullLayer(img *cv.Mat) Layer {
	w := FeatherWeightMap(img.Cols, img.Rows, 1)
	return Layer{Image: img, Weight: w}
}

func gradientImage(rows, cols, ch int) *cv.Mat {
	img := cv.NewMat(rows, cols, ch)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			base := (y*cols + x) * ch
			for c := 0; c < ch; c++ {
				img.Data[base+c] = uint8((x*4 + y*3 + c*20) % 256)
			}
		}
	}
	return img
}

func TestFeatherBlendSingleLayer(t *testing.T) {
	img := gradientImage(8, 8, 3)
	out, err := FeatherBlender{}.Blend([]Layer{fullLayer(img)})
	if err != nil {
		t.Fatal(err)
	}
	for i := range img.Data {
		if d := int(out.Data[i]) - int(img.Data[i]); d < -1 || d > 1 {
			t.Fatalf("single-layer feather changed pixel %d: %d vs %d", i, out.Data[i], img.Data[i])
		}
	}
}

func TestFeatherBlendEmpty(t *testing.T) {
	if _, err := (FeatherBlender{}).Blend(nil); err == nil {
		t.Fatal("expected error for empty layers")
	}
}

func TestMultiBandBlendIdentical(t *testing.T) {
	img := gradientImage(16, 16, 3)
	l1 := fullLayer(img)
	l2 := fullLayer(img.Clone())
	out, err := MultiBandBlender{NumBands: 3}.Blend([]Layer{l1, l2})
	if err != nil {
		t.Fatal(err)
	}
	var maxErr int
	for i := range img.Data {
		d := int(out.Data[i]) - int(img.Data[i])
		if d < 0 {
			d = -d
		}
		if d > maxErr {
			maxErr = d
		}
	}
	if maxErr > 2 {
		t.Fatalf("multi-band blend of identical images differs by %d", maxErr)
	}
}

func TestMultiBandBlendSeam(t *testing.T) {
	// Two solid half-covering layers of different colours must blend to a valid
	// image with no out-of-range values and colours between the two inputs.
	rows, cols := 16, 16
	left := cv.NewMat(rows, cols, 1)
	right := cv.NewMat(rows, cols, 1)
	wl := cv.NewFloatMat(rows, cols)
	wr := cv.NewFloatMat(rows, cols)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			p := y*cols + x
			if x < cols/2 {
				left.Data[p] = 50
				wl.Data[p] = 1
			}
			if x >= cols/2 {
				right.Data[p] = 200
				wr.Data[p] = 1
			}
		}
	}
	out, err := MultiBandBlender{NumBands: 2}.Blend([]Layer{
		{Image: left, Weight: wl}, {Image: right, Weight: wr},
	})
	if err != nil {
		t.Fatal(err)
	}
	// Far edges are dominated by their own layer (pyramid boundary smoothing
	// permits a couple of levels of slack); the blend must stay in range and
	// keep the left side darker than the right.
	le := int(out.At(8, 0, 0))
	re := int(out.At(8, cols-1, 0))
	if le < 45 || le > 60 || re < 190 || re > 205 {
		t.Fatalf("edges = %d,%d want ≈50,≈200", le, re)
	}
	if le >= re {
		t.Fatalf("left edge %d should be darker than right edge %d", le, re)
	}
}

func BenchmarkMultiBandBlend(b *testing.B) {
	rows, cols := 128, 128
	imgA := gradientImage(rows, cols, 3)
	imgB := gradientImage(rows, cols, 3)
	for i := range imgB.Data {
		imgB.Data[i] = 255 - imgB.Data[i]
	}
	layers := []Layer{fullLayer(imgA), fullLayer(imgB)}
	blender := MultiBandBlender{NumBands: 4}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := blender.Blend(layers); err != nil {
			b.Fatal(err)
		}
	}
}
