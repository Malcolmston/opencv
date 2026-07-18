package saliency2_test

import (
	"testing"

	cv "github.com/malcolmston/opencv"
	"github.com/malcolmston/opencv/saliency2"
)

// benchScene builds a deterministic multi-object grayscale scene for benchmarks.
func benchScene(size int) *cv.Mat {
	m := cv.NewMat(size, size, 1)
	m.SetTo(40)
	for y := 0; y < size; y++ {
		for x := 0; x < size; x++ {
			cx1, cy1 := size/3, size/3
			cx2, cy2 := 2*size/3, 2*size/3
			d1 := (x-cx1)*(x-cx1) + (y-cy1)*(y-cy1)
			d2 := (x-cx2)*(x-cx2) + (y-cy2)*(y-cy2)
			if d1 <= (size/8)*(size/8) {
				m.Set(y, x, 0, 210)
			} else if d2 <= (size/10)*(size/10) {
				m.Set(y, x, 0, 150)
			}
		}
	}
	return m
}

// BenchmarkIttiKoch benchmarks the heaviest detector, the Itti-Koch center-
// surround model, on a 128x128 scene.
func BenchmarkIttiKoch(b *testing.B) {
	img := benchScene(128)
	det := saliency2.NewStaticSaliencyIttiKoch()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = det.ComputeSaliency(img)
	}
}
