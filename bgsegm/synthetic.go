package bgsegm

import (
	"math"
	"math/rand"

	cv "github.com/malcolmston/opencv"
)

// SyntheticSequenceGenerator reproduces OpenCV's bgsegm helper of the same name:
// it manufactures an endless test video by sliding a small foreground object
// over a static background image while adding Gaussian sensor noise, and it
// hands back the exact ground-truth foreground mask for every frame. It is the
// standard way to exercise and score a [BackgroundSubtractor] without real
// footage.
//
// The object follows a smooth path: it drifts horizontally at Objspeed columns
// per frame (wrapping around the width) while oscillating vertically by
// Amplitude rows with period Wavelength frames, the whole path advancing in time
// at Wavespeed. Each generated frame is the background plus zero-mean Gaussian
// noise of standard deviation NoiseStdDev, with the object's non-zero pixels
// composited on top. All randomness is drawn from a seeded generator, so the
// same construction parameters always yield byte-identical sequences.
//
// Construct one with [NewSyntheticSequenceGenerator]; the zero value is not
// usable.
type SyntheticSequenceGenerator struct {
	// Amplitude is the vertical oscillation amplitude of the object, in rows.
	Amplitude float64
	// Wavelength is the oscillation period, in frames.
	Wavelength float64
	// Wavespeed scales how fast simulated time advances per frame.
	Wavespeed float64
	// Objspeed is the horizontal drift of the object, in columns per frame.
	Objspeed float64
	// NoiseStdDev is the standard deviation of the additive Gaussian noise.
	NoiseStdDev float64

	background *cv.Mat
	object     *cv.Mat
	rng        *rand.Rand
	t          float64
	baseX      float64
	baseY      float64
}

// NewSyntheticSequenceGenerator creates a generator that slides object over
// background. background must be a single-channel image and object a
// single-channel patch smaller than it; the object starts centred. amplitude,
// wavelength, wavespeed and objspeed shape the motion (see the field docs) and
// fall back to sensible defaults (2, 20, 1, 1) when non-positive; noiseStdDev is
// the additive-noise level; seed makes the sequence reproducible. It panics if
// either image is nil/empty or multi-channel, or if object does not fit inside
// background.
func NewSyntheticSequenceGenerator(background, object *cv.Mat, amplitude, wavelength, wavespeed, objspeed, noiseStdDev float64, seed int64) *SyntheticSequenceGenerator {
	if background == nil || background.Empty() || object == nil || object.Empty() {
		panic("bgsegm: SyntheticSequenceGenerator requires non-empty background and object")
	}
	if background.Channels != 1 || object.Channels != 1 {
		panic("bgsegm: SyntheticSequenceGenerator requires single-channel background and object")
	}
	if object.Rows > background.Rows || object.Cols > background.Cols {
		panic("bgsegm: SyntheticSequenceGenerator object does not fit inside background")
	}
	if amplitude <= 0 {
		amplitude = 2
	}
	if wavelength <= 0 {
		wavelength = 20
	}
	if wavespeed <= 0 {
		wavespeed = 1
	}
	if objspeed <= 0 {
		objspeed = 1
	}
	return &SyntheticSequenceGenerator{
		Amplitude:   amplitude,
		Wavelength:  wavelength,
		Wavespeed:   wavespeed,
		Objspeed:    objspeed,
		NoiseStdDev: noiseStdDev,
		background:  background.Clone(),
		object:      object.Clone(),
		rng:         rand.New(rand.NewSource(seed)),
		baseX:       float64(background.Cols-object.Cols) / 2,
		baseY:       float64(background.Rows-object.Rows) / 2,
	}
}

// Next produces the next frame of the sequence and its ground-truth foreground
// mask. The frame is a fresh single-channel [cv.Mat] the size of the background,
// and gtMask is the same size with [ForegroundValue] at every pixel the object
// currently covers and [BackgroundValue] elsewhere. Each call advances the
// internal clock, so successive calls trace the object along its path.
func (g *SyntheticSequenceGenerator) Next() (frame, gtMask *cv.Mat) {
	rows, cols := g.background.Rows, g.background.Cols
	frame = g.background.Clone()

	// Additive zero-mean Gaussian noise, saturated to the byte range.
	if g.NoiseStdDev > 0 {
		for i := range frame.Data {
			v := float64(frame.Data[i]) + g.rng.NormFloat64()*g.NoiseStdDev
			frame.Data[i] = clampUint8(v)
		}
	}

	// Object position along its sinusoidal path.
	x := g.baseX + g.Objspeed*g.t
	if span := float64(cols - g.object.Cols + 1); span > 0 {
		x = math.Mod(x, span)
		if x < 0 {
			x += span
		}
	}
	y := g.baseY + g.Amplitude*math.Sin(2*math.Pi*g.t*g.Wavespeed/g.Wavelength)
	ox := int(math.Round(x))
	oy := int(math.Round(y))
	if ox < 0 {
		ox = 0
	}
	if ox > cols-g.object.Cols {
		ox = cols - g.object.Cols
	}
	if oy < 0 {
		oy = 0
	}
	if oy > rows-g.object.Rows {
		oy = rows - g.object.Rows
	}

	gtMask = newMask(rows, cols)
	for ry := 0; ry < g.object.Rows; ry++ {
		for rx := 0; rx < g.object.Cols; rx++ {
			v := g.object.Data[ry*g.object.Cols+rx]
			if v == 0 {
				continue // transparent object pixel
			}
			frame.Data[(oy+ry)*cols+(ox+rx)] = v
			gtMask.Data[(oy+ry)*cols+(ox+rx)] = ForegroundValue
		}
	}

	g.t++
	return frame, gtMask
}
