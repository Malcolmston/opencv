package tracking

import (
	"math/rand"
)

// Particle is a single hypothesis in a [ParticleFilter]: a state vector and its
// importance weight.
type Particle struct {
	// State is the hypothesised state vector.
	State []float64
	// Weight is the (normalised) importance weight.
	Weight float64
}

// ParticleFilter is a bootstrap (sequential importance resampling) particle
// filter for non-linear/non-Gaussian tracking. It maintains a weighted set of
// state hypotheses that are propagated by a motion model, reweighted by a
// measurement likelihood and periodically resampled. All randomness is drawn
// from an internal, seeded generator, so a filter constructed with a fixed seed
// and driven with the same inputs is fully deterministic.
type ParticleFilter struct {
	particles  [][]float64
	weights    []float64
	processStd []float64
	stateDim   int
	rng        *rand.Rand
}

// NewParticleFilter creates a filter with numParticles hypotheses of dimension
// stateDim, seeded for deterministic behaviour. All particles start at the
// origin with uniform weights; call [ParticleFilter.Init] to scatter them around
// an initial estimate and [ParticleFilter.SetProcessNoise] to set the motion
// noise. It panics if numParticles or stateDim is not positive.
func NewParticleFilter(numParticles, stateDim int, seed int64) *ParticleFilter {
	if numParticles <= 0 || stateDim <= 0 {
		panic("tracking: NewParticleFilter requires positive numParticles and stateDim")
	}
	pf := &ParticleFilter{
		particles:  make([][]float64, numParticles),
		weights:    make([]float64, numParticles),
		processStd: make([]float64, stateDim),
		stateDim:   stateDim,
		rng:        rand.New(rand.NewSource(seed)),
	}
	w := 1.0 / float64(numParticles)
	for i := range pf.particles {
		pf.particles[i] = make([]float64, stateDim)
		pf.weights[i] = w
	}
	return pf
}

// Count returns the number of particles.
func (pf *ParticleFilter) Count() int { return len(pf.particles) }

// SetProcessNoise sets the per-dimension standard deviation of the Gaussian
// process noise added on each [ParticleFilter.Predict]. std must have length
// stateDim.
func (pf *ParticleFilter) SetProcessNoise(std []float64) {
	if len(std) != pf.stateDim {
		panic("tracking: SetProcessNoise std has wrong length")
	}
	copy(pf.processStd, std)
}

// Init scatters the particles around mean with independent Gaussian spread std
// (both of length stateDim) and resets the weights to uniform.
func (pf *ParticleFilter) Init(mean, std []float64) {
	if len(mean) != pf.stateDim || len(std) != pf.stateDim {
		panic("tracking: Init mean/std have wrong length")
	}
	w := 1.0 / float64(len(pf.particles))
	for i := range pf.particles {
		for d := 0; d < pf.stateDim; d++ {
			pf.particles[i][d] = mean[d] + pf.rng.NormFloat64()*std[d]
		}
		pf.weights[i] = w
	}
}

// Predict advances every particle by the deterministic motion vector (which may
// be nil for a stationary model) and adds the configured Gaussian process noise.
// motion, when non-nil, must have length stateDim.
func (pf *ParticleFilter) Predict(motion []float64) {
	if motion != nil && len(motion) != pf.stateDim {
		panic("tracking: Predict motion has wrong length")
	}
	for i := range pf.particles {
		for d := 0; d < pf.stateDim; d++ {
			if motion != nil {
				pf.particles[i][d] += motion[d]
			}
			if pf.processStd[d] > 0 {
				pf.particles[i][d] += pf.rng.NormFloat64() * pf.processStd[d]
			}
		}
	}
}

// UpdateWeights multiplies each particle's weight by the likelihood that the
// current measurement was generated from its state, as returned by likelihood,
// then renormalises the weights to sum to one. If every likelihood is zero the
// weights are reset to uniform to keep the filter alive.
func (pf *ParticleFilter) UpdateWeights(likelihood func(state []float64) float64) {
	var sum float64
	for i := range pf.particles {
		pf.weights[i] *= likelihood(pf.particles[i])
		sum += pf.weights[i]
	}
	if sum <= 0 {
		w := 1.0 / float64(len(pf.particles))
		for i := range pf.weights {
			pf.weights[i] = w
		}
		return
	}
	for i := range pf.weights {
		pf.weights[i] /= sum
	}
}

// Estimate returns the weighted-mean state over all particles.
func (pf *ParticleFilter) Estimate() []float64 {
	out := make([]float64, pf.stateDim)
	for i := range pf.particles {
		w := pf.weights[i]
		for d := 0; d < pf.stateDim; d++ {
			out[d] += pf.particles[i][d] * w
		}
	}
	return out
}

// EffectiveSampleSize returns the effective number of particles
// 1/Σ wᵢ², a measure of weight degeneracy. It ranges from 1 (all weight on one
// particle) to the particle count (uniform weights); resample when it drops well
// below the count.
func (pf *ParticleFilter) EffectiveSampleSize() float64 {
	var s float64
	for _, w := range pf.weights {
		s += w * w
	}
	if s <= 0 {
		return 0
	}
	return 1 / s
}

// Resample performs low-variance systematic resampling, replacing the particle
// set with a new one drawn in proportion to the current weights and resetting
// all weights to uniform. The single random offset is drawn from the internal
// seeded generator, so resampling is deterministic.
func (pf *ParticleFilter) Resample() {
	n := len(pf.particles)
	positions := make([]float64, n)
	r0 := pf.rng.Float64() / float64(n)
	for i := 0; i < n; i++ {
		positions[i] = r0 + float64(i)/float64(n)
	}
	cumulative := make([]float64, n)
	var c float64
	for i, w := range pf.weights {
		c += w
		cumulative[i] = c
	}
	newParticles := make([][]float64, n)
	i, j := 0, 0
	for i < n {
		if positions[i] < cumulative[j] {
			p := make([]float64, pf.stateDim)
			copy(p, pf.particles[j])
			newParticles[i] = p
			i++
		} else {
			j++
			if j >= n {
				j = n - 1
			}
		}
	}
	pf.particles = newParticles
	w := 1.0 / float64(n)
	for k := range pf.weights {
		pf.weights[k] = w
	}
}

// Particles returns a snapshot copy of the current particle set with their
// weights.
func (pf *ParticleFilter) Particles() []Particle {
	out := make([]Particle, len(pf.particles))
	for i := range pf.particles {
		s := make([]float64, pf.stateDim)
		copy(s, pf.particles[i])
		out[i] = Particle{State: s, Weight: pf.weights[i]}
	}
	return out
}
