package ml2

import "testing"

// BenchmarkRandomForestFit measures training the heaviest routine in the
// package: growing a bootstrap-aggregated ensemble of randomised CART trees on
// a modest synthetic vision-style dataset.
func BenchmarkRandomForestFit(b *testing.B) {
	x, y := genBlobs([][]float64{{0, 0}, {5, 5}, {5, 0}, {0, 5}}, 60, 0.8, 99)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		m := NewRandomForest(50, 8, 0, 7)
		if err := m.Fit(x, y); err != nil {
			b.Fatal(err)
		}
	}
}
