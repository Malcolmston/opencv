package tracking

import "testing"

// BenchmarkHornSchunck exercises the heaviest routine in the package: a dense
// variational flow solve over a moderately sized image with many Jacobi sweeps.
func BenchmarkHornSchunck(b *testing.B) {
	prev := synthTexture(96, 96, 0, 0)
	next := synthTexture(96, 96, 1, 0)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = CalcOpticalFlowHornSchunck(prev, next, 5, 50)
	}
}

// BenchmarkFarneback benchmarks the dense block-matching estimator, the other
// compute-heavy dense routine.
func BenchmarkFarneback(b *testing.B) {
	prev := synthTexture(96, 96, 0, 0)
	next := synthTexture(96, 96, 2, 1)
	params := DefaultFarnebackParams()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = CalcOpticalFlowFarneback(prev, next, params)
	}
}
