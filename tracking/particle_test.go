package tracking

import (
	"math"
	"testing"
)

// gaussianLikelihood returns a likelihood peaked at target with the given
// standard deviation, for a 1-D state.
func gaussianLikelihood(target, std float64) func([]float64) float64 {
	return func(s []float64) float64 {
		d := s[0] - target
		return math.Exp(-(d * d) / (2 * std * std))
	}
}

func runParticleFilter(seed int64) float64 {
	pf := NewParticleFilter(500, 1, seed)
	pf.Init([]float64{0}, []float64{5})
	pf.SetProcessNoise([]float64{1.0})
	like := gaussianLikelihood(10, 2.0)
	var est []float64
	for i := 0; i < 20; i++ {
		pf.Predict(nil)
		pf.UpdateWeights(like)
		pf.Resample()
		est = pf.Estimate()
	}
	return est[0]
}

func TestParticleFilterConverges(t *testing.T) {
	got := runParticleFilter(42)
	requireTrue(t, approx(got, 10, 1.0), "estimate = %v, want ~10", got)
}

func TestParticleFilterDeterministic(t *testing.T) {
	a := runParticleFilter(7)
	b := runParticleFilter(7)
	if a != b {
		t.Fatalf("same seed produced different results: %v vs %v", a, b)
	}
}

func TestParticleFilterEffectiveSampleSize(t *testing.T) {
	pf := NewParticleFilter(100, 1, 1)
	// Uniform weights => effective sample size equals the particle count.
	if !approx(pf.EffectiveSampleSize(), 100, 1e-6) {
		t.Fatalf("uniform ESS = %v, want 100", pf.EffectiveSampleSize())
	}
	if pf.Count() != 100 {
		t.Fatalf("Count = %d, want 100", pf.Count())
	}
	if len(pf.Particles()) != 100 {
		t.Fatalf("Particles snapshot wrong length")
	}
}
