package cudaimgproc

import (
	"fmt"

	cv "github.com/malcolmston/opencv"
)

// BilateralFilter applies an edge-preserving bilateral filter, mirroring
// cuda::bilateralFilter. d is the diameter of the pixel neighbourhood;
// sigmaColor weights by intensity similarity and sigmaSpace by spatial
// distance. The trailing Stream argument is accepted and ignored. It panics on
// an empty source.
func BilateralFilter(src GpuMat, d int, sigmaColor, sigmaSpace float64, streams ...Stream) GpuMat {
	_ = firstStream(streams)
	m := src.requireHost("BilateralFilter")
	return wrap(cv.BilateralFilter(m, d, sigmaColor, sigmaSpace))
}

// Blend returns the constant-weight linear combination
// weight1·img1 + weight2·img2, saturating to [0,255], mirroring the scalar form
// of cuda::blendLinear. Both inputs must have matching dimensions and channel
// counts. The trailing Stream argument is accepted and ignored.
func Blend(img1, img2 GpuMat, weight1, weight2 float64, streams ...Stream) GpuMat {
	_ = firstStream(streams)
	a := img1.requireHost("Blend")
	b := img2.requireHost("Blend")
	if a.Rows != b.Rows || a.Cols != b.Cols || a.Channels != b.Channels {
		panic("cudaimgproc: Blend requires images of identical shape")
	}
	return wrap(cv.AddWeighted(a, weight1, b, weight2, 0))
}

// BlendLinear blends two images using per-pixel weight maps, mirroring
// cuda::blendLinear: for each pixel the output is
// (w1·img1 + w2·img2)/(w1 + w2), where w1 and w2 are read from the
// single-channel weight images (their 8-bit values are used directly as
// weights). Where both weights are zero the output pixel is zero. All four
// inputs must share the same dimensions; img1 and img2 must have equal channel
// counts and the weight maps must be single-channel. The trailing Stream
// argument is accepted and ignored.
func BlendLinear(img1, img2, weights1, weights2 GpuMat, streams ...Stream) GpuMat {
	_ = firstStream(streams)
	a := img1.requireHost("BlendLinear")
	b := img2.requireHost("BlendLinear")
	w1 := weights1.requireHost("BlendLinear")
	w2 := weights2.requireHost("BlendLinear")
	if a.Rows != b.Rows || a.Cols != b.Cols || a.Channels != b.Channels {
		panic("cudaimgproc: BlendLinear image shape mismatch")
	}
	if w1.Channels != 1 || w2.Channels != 1 {
		panic("cudaimgproc: BlendLinear weight maps must be single-channel")
	}
	if w1.Rows != a.Rows || w1.Cols != a.Cols || w2.Rows != a.Rows || w2.Cols != a.Cols {
		panic(fmt.Sprintf("cudaimgproc: BlendLinear weight size mismatch (%dx%d)", a.Rows, a.Cols))
	}
	dst := cv.NewMat(a.Rows, a.Cols, a.Channels)
	n := a.Total()
	for p := 0; p < n; p++ {
		f1 := float64(w1.Data[p])
		f2 := float64(w2.Data[p])
		denom := f1 + f2
		base := p * a.Channels
		if denom == 0 {
			continue
		}
		for c := 0; c < a.Channels; c++ {
			v := (f1*float64(a.Data[base+c]) + f2*float64(b.Data[base+c])) / denom
			dst.Data[base+c] = clampU8(v + 0.5)
		}
	}
	return wrap(dst)
}
