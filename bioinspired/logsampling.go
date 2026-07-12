package bioinspired

import (
	"fmt"
	"math"

	cv "github.com/malcolmston/opencv"
)

// RetinaLogSampler models the non-uniform, log-polar photoreceptor sampling of
// the retina: dense sampling near the fovea (image centre) and progressively
// sparser sampling towards the periphery. It maps a Cartesian [cv.Mat] to a
// log-polar image whose rows are rings at logarithmically-spaced radii and whose
// columns are angular sectors, and back again. This is the "configurable
// photoreceptor log-sampling" that OpenCV's retina applies before its filter
// bank; here it is exposed as a standalone, reversible transform.
//
// Because the periphery is undersampled, [RetinaLogSampler.InverseSample] is a
// lossy inverse of [RetinaLogSampler.Sample]; it reconstructs smooth and central
// content well and blurs fine peripheral detail, matching foveal magnification.
type RetinaLogSampler struct {
	rows, cols     int
	rings, sectors int

	cx, cy     float64
	rMin, rMax float64
	logBase    float64 // ln(rMax/rMin)
}

// NewRetinaLogSampler creates a sampler for Cartesian images of size rows x cols
// that produces log-polar images of size rings x sectors. All four dimensions
// must be positive. The fovea is placed at the image centre and the outermost
// ring reaches the farthest image corner, so the whole frame is covered.
func NewRetinaLogSampler(rows, cols, rings, sectors int) *RetinaLogSampler {
	if rows <= 0 || cols <= 0 || rings <= 0 || sectors <= 0 {
		panic(fmt.Sprintf("bioinspired: NewRetinaLogSampler requires positive dimensions, got %dx%d -> %dx%d",
			rows, cols, rings, sectors))
	}
	cx := float64(cols-1) / 2
	cy := float64(rows-1) / 2
	rMax := math.Hypot(math.Max(cx, float64(cols-1)-cx), math.Max(cy, float64(rows-1)-cy))
	if rMax < 1 {
		rMax = 1
	}
	rMin := 1.0
	return &RetinaLogSampler{
		rows: rows, cols: cols, rings: rings, sectors: sectors,
		cx: cx, cy: cy, rMin: rMin, rMax: rMax,
		logBase: math.Log(rMax / rMin),
	}
}

// InputSize returns the Cartesian frame size, as (rows, cols).
func (s *RetinaLogSampler) InputSize() (rows, cols int) { return s.rows, s.cols }

// OutputSize returns the log-polar image size, as (rings, sectors).
func (s *RetinaLogSampler) OutputSize() (rings, sectors int) { return s.rings, s.sectors }

// ringRadius returns the radius (in Cartesian pixels) of ring i in [0, rings).
func (s *RetinaLogSampler) ringRadius(i int) float64 {
	t := float64(i) / float64(s.rings-1)
	if s.rings == 1 {
		t = 0
	}
	return s.rMin * math.Exp(t*s.logBase)
}

// Sample transforms a Cartesian image into its log-polar representation. The
// output is a rings x sectors [cv.Mat] with the same channel count as the input;
// each output sample is read from the input by bilinear interpolation at the
// corresponding (radius, angle). It panics on a size or emptiness mismatch.
func (s *RetinaLogSampler) Sample(input *cv.Mat) *cv.Mat {
	if input.Empty() {
		panic("bioinspired: RetinaLogSampler.Sample given an empty Mat")
	}
	if input.Rows != s.rows || input.Cols != s.cols {
		panic(fmt.Sprintf("bioinspired: RetinaLogSampler.Sample size mismatch: got %dx%d want %dx%d",
			input.Rows, input.Cols, s.rows, s.cols))
	}
	nc := input.Channels
	out := cv.NewMat(s.rings, s.sectors, nc)
	for i := 0; i < s.rings; i++ {
		r := s.ringRadius(i)
		for j := 0; j < s.sectors; j++ {
			theta := 2 * math.Pi * float64(j) / float64(s.sectors)
			x := s.cx + r*math.Cos(theta)
			y := s.cy + r*math.Sin(theta)
			dst := (i*s.sectors + j) * nc
			for c := 0; c < nc; c++ {
				out.Data[dst+c] = clampRound(bilinearSample(input, y, x, c))
			}
		}
	}
	return out
}

// InverseSample reconstructs a Cartesian image from a log-polar image produced
// by [RetinaLogSampler.Sample]. For each Cartesian pixel it computes the ring and
// sector coordinates of its (radius, angle) and reads the log-polar image by
// bilinear interpolation, wrapping around in the angular dimension. Peripheral
// detail is smoothed because it was undersampled. It panics if logPolar is not
// the sampler's rings x sectors size.
func (s *RetinaLogSampler) InverseSample(logPolar *cv.Mat) *cv.Mat {
	if logPolar.Empty() {
		panic("bioinspired: RetinaLogSampler.InverseSample given an empty Mat")
	}
	if logPolar.Rows != s.rings || logPolar.Cols != s.sectors {
		panic(fmt.Sprintf("bioinspired: RetinaLogSampler.InverseSample size mismatch: got %dx%d want %dx%d",
			logPolar.Rows, logPolar.Cols, s.rings, s.sectors))
	}
	nc := logPolar.Channels
	out := cv.NewMat(s.rows, s.cols, nc)
	for y := 0; y < s.rows; y++ {
		for x := 0; x < s.cols; x++ {
			dx := float64(x) - s.cx
			dy := float64(y) - s.cy
			r := math.Hypot(dx, dy)
			if r < s.rMin {
				r = s.rMin
			}
			// Ring coordinate from the inverse of the log radius mapping.
			var ringPos float64
			if s.rings > 1 && s.logBase > 0 {
				ringPos = math.Log(r/s.rMin) / s.logBase * float64(s.rings-1)
			}
			theta := math.Atan2(dy, dx)
			if theta < 0 {
				theta += 2 * math.Pi
			}
			sectorPos := theta / (2 * math.Pi) * float64(s.sectors)
			dst := (y*s.cols + x) * nc
			for c := 0; c < nc; c++ {
				out.Data[dst+c] = clampRound(bilinearSampleWrap(logPolar, ringPos, sectorPos, c))
			}
		}
	}
	return out
}

// bilinearSample reads channel c of m at fractional (y, x) using bilinear
// interpolation with edge replication for out-of-range coordinates.
func bilinearSample(m *cv.Mat, y, x float64, c int) float64 {
	x0 := int(math.Floor(x))
	y0 := int(math.Floor(y))
	fx := x - float64(x0)
	fy := y - float64(y0)
	v00 := replicateSample(m, y0, x0, c)
	v01 := replicateSample(m, y0, x0+1, c)
	v10 := replicateSample(m, y0+1, x0, c)
	v11 := replicateSample(m, y0+1, x0+1, c)
	top := v00*(1-fx) + v01*fx
	bot := v10*(1-fx) + v11*fx
	return top*(1-fy) + bot*fy
}

// bilinearSampleWrap reads channel c of a log-polar Mat at fractional
// (ring, sector); the ring axis replicates at the ends and the sector axis wraps
// around modulo the number of columns.
func bilinearSampleWrap(m *cv.Mat, ring, sector float64, c int) float64 {
	r0 := int(math.Floor(ring))
	s0 := int(math.Floor(sector))
	fr := ring - float64(r0)
	fs := sector - float64(s0)
	clampRing := func(r int) int {
		if r < 0 {
			return 0
		}
		if r >= m.Rows {
			return m.Rows - 1
		}
		return r
	}
	wrapSec := func(sv int) int {
		sv %= m.Cols
		if sv < 0 {
			sv += m.Cols
		}
		return sv
	}
	get := func(r, sv int) float64 {
		return float64(m.Data[(clampRing(r)*m.Cols+wrapSec(sv))*m.Channels+c])
	}
	top := get(r0, s0)*(1-fs) + get(r0, s0+1)*fs
	bot := get(r0+1, s0)*(1-fs) + get(r0+1, s0+1)*fs
	return top*(1-fr) + bot*fr
}

// replicateSample reads channel c of m at integer (y, x), clamping out-of-range
// coordinates to the nearest edge.
func replicateSample(m *cv.Mat, y, x, c int) float64 {
	if y < 0 {
		y = 0
	} else if y >= m.Rows {
		y = m.Rows - 1
	}
	if x < 0 {
		x = 0
	} else if x >= m.Cols {
		x = m.Cols - 1
	}
	return float64(m.Data[(y*m.Cols+x)*m.Channels+c])
}
