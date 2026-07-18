package core

import "math"

// RNG is a fast pseudo-random number generator using the multiply-with-carry
// scheme of OpenCV's cv::RNG. Given the same seed it produces the same
// deterministic stream on every platform.
type RNG struct {
	state uint64
	// hasCachedGauss and cachedGauss implement the Box–Muller pair cache.
	hasCachedGauss bool
	cachedGauss    float64
}

// NewRNG creates an RNG seeded with the given value. A zero seed is replaced by
// the OpenCV default so the generator never degenerates.
func NewRNG(seed uint64) *RNG {
	if seed == 0 {
		seed = 0xffffffff
	}
	return &RNG{state: seed}
}

// Next returns the next 32-bit pseudo-random value and advances the state.
func (r *RNG) Next() uint32 {
	r.state = uint64(uint32(r.state))*4164903690 + (r.state >> 32)
	return uint32(r.state)
}

// Uniformi returns a uniformly distributed integer in the half-open range
// [a, b). It panics if b is not greater than a.
func (r *RNG) Uniformi(a, b int) int {
	if b <= a {
		panic("core: RNG.Uniformi requires b > a")
	}
	return a + int(r.Next()%uint32(b-a))
}

// Uniformf returns a uniformly distributed float32 in the half-open range
// [a, b).
func (r *RNG) Uniformf(a, b float32) float32 {
	return a + float32(float64(r.Next())*2.3283064365386963e-10)*(b-a)
}

// Uniformd returns a uniformly distributed float64 in the half-open range
// [a, b).
func (r *RNG) Uniformd(a, b float64) float64 {
	return a + float64(r.Next())*2.3283064365386963e-10*(b-a)
}

// NextBool returns a pseudo-random boolean.
func (r *RNG) NextBool() bool { return r.Next()&1 == 1 }

// Gaussian returns a normally distributed value with mean 0 and the given
// standard deviation, using the Box–Muller transform.
func (r *RNG) Gaussian(sigma float64) float64 {
	if r.hasCachedGauss {
		r.hasCachedGauss = false
		return r.cachedGauss * sigma
	}
	var u1, u2 float64
	for u1 <= 1e-15 {
		u1 = r.Uniformd(0, 1)
	}
	u2 = r.Uniformd(0, 1)
	mag := math.Sqrt(-2 * math.Log(u1))
	r.cachedGauss = mag * math.Sin(2*math.Pi*u2)
	r.hasCachedGauss = true
	return mag * math.Cos(2*math.Pi*u2) * sigma
}

// FillUniformd fills dst with independent uniform samples in [a, b).
func (r *RNG) FillUniformd(dst []float64, a, b float64) {
	for i := range dst {
		dst[i] = r.Uniformd(a, b)
	}
}

// FillGaussian fills dst with independent normal samples of the given standard
// deviation and zero mean.
func (r *RNG) FillGaussian(dst []float64, sigma float64) {
	for i := range dst {
		dst[i] = r.Gaussian(sigma)
	}
}

// RNGMT19937 is a Mersenne Twister generator matching OpenCV's
// cv::RNG_MT19937, offering a longer period than [RNG].
type RNGMT19937 struct {
	mt  [624]uint32
	idx int
}

// NewRNGMT19937 creates a Mersenne Twister seeded with the given value.
func NewRNGMT19937(seed uint32) *RNGMT19937 {
	r := &RNGMT19937{}
	r.mt[0] = seed
	for i := 1; i < 624; i++ {
		r.mt[i] = 1812433253*(r.mt[i-1]^(r.mt[i-1]>>30)) + uint32(i)
	}
	r.idx = 624
	return r
}

// Next returns the next 32-bit pseudo-random value.
func (r *RNGMT19937) Next() uint32 {
	if r.idx >= 624 {
		for i := 0; i < 624; i++ {
			y := (r.mt[i] & 0x80000000) | (r.mt[(i+1)%624] & 0x7fffffff)
			next := r.mt[(i+397)%624] ^ (y >> 1)
			if y&1 != 0 {
				next ^= 2567483615
			}
			r.mt[i] = next
		}
		r.idx = 0
	}
	y := r.mt[r.idx]
	r.idx++
	y ^= y >> 11
	y ^= (y << 7) & 2636928640
	y ^= (y << 15) & 4022730752
	y ^= y >> 18
	return y
}

// Uniformi returns a uniformly distributed integer in the half-open range
// [a, b). It panics if b is not greater than a.
func (r *RNGMT19937) Uniformi(a, b int) int {
	if b <= a {
		panic("core: RNGMT19937.Uniformi requires b > a")
	}
	return a + int(r.Next()%uint32(b-a))
}

// Uniformd returns a uniformly distributed float64 in the half-open range
// [a, b).
func (r *RNGMT19937) Uniformd(a, b float64) float64 {
	return a + float64(r.Next())*2.3283064365386963e-10*(b-a)
}
