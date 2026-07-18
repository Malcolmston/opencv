package transforms2

import (
	"testing"

	cv "github.com/malcolmston/opencv"
)

// BenchmarkThinPlateSplineWarp exercises the heaviest routine: a thin-plate
// spline evaluated at every output pixel with several control points.
func BenchmarkThinPlateSplineWarp(b *testing.B) {
	src := grad(96, 96)
	from := []cv.Point2f{
		{X: 0, Y: 0}, {X: 95, Y: 0}, {X: 95, Y: 95}, {X: 0, Y: 95},
		{X: 48, Y: 48}, {X: 24, Y: 70}, {X: 70, Y: 24},
	}
	to := []cv.Point2f{
		{X: 2, Y: 1}, {X: 94, Y: 3}, {X: 93, Y: 92}, {X: 1, Y: 90},
		{X: 50, Y: 46}, {X: 22, Y: 72}, {X: 72, Y: 26},
	}
	tps := NewThinPlateSpline(to, from, 0)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = tps.Warp(src, 96, 96, InterpBilinear, BorderReplicate, 0)
	}
}
