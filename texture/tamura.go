package texture

import (
	"fmt"
	"math"

	cv "github.com/malcolmston/opencv"
)

// TamuraFeatureSet bundles the three principal Tamura perceptual texture
// features together with the derived roughness. These features were designed to
// correspond to human visual perception of texture.
type TamuraFeatureSet struct {
	// Coarseness measures the scale of the dominant texture elements; larger
	// values indicate coarser texture. See [TamuraCoarseness].
	Coarseness float64
	// Contrast measures the dynamic range and polarisation of intensities. See
	// [TamuraContrast].
	Contrast float64
	// Directionality measures how strongly texture orientation is concentrated
	// in one direction, in [0,1]. See [TamuraDirectionality].
	Directionality float64
	// Roughness is Coarseness + Contrast, Tamura's combined roughness measure.
	Roughness float64
}

// textureIntegral builds a (rows+1)x(cols+1) summed-area table of a row-major
// plane, so that the sum over any axis-aligned rectangle is a constant-time
// lookup.
func textureIntegral(src []float64, rows, cols int) []float64 {
	w := cols + 1
	ii := make([]float64, (rows+1)*w)
	for y := 0; y < rows; y++ {
		var rowsum float64
		for x := 0; x < cols; x++ {
			rowsum += src[y*cols+x]
			ii[(y+1)*w+(x+1)] = ii[y*w+(x+1)] + rowsum
		}
	}
	return ii
}

// rectMean returns the mean of src over the inclusive rectangle [x0,x1]x[y0,y1],
// clamped to the image, using the integral image ii of width cols+1.
func rectMean(ii []float64, rows, cols, x0, y0, x1, y1 int) float64 {
	// Replicate-border clamping: a window that falls partly or wholly outside
	// the image is clamped to the nearest valid rows/columns, so a flat image
	// yields zero cross-window difference at every scale.
	clamp := func(v, hi int) int {
		if v < 0 {
			return 0
		}
		if v > hi {
			return hi
		}
		return v
	}
	x0 = clamp(x0, cols-1)
	x1 = clamp(x1, cols-1)
	y0 = clamp(y0, rows-1)
	y1 = clamp(y1, rows-1)
	if x1 < x0 || y1 < y0 {
		return 0
	}
	w := cols + 1
	s := ii[(y1+1)*w+(x1+1)] - ii[y0*w+(x1+1)] - ii[(y1+1)*w+x0] + ii[y0*w+x0]
	n := float64((x1 - x0 + 1) * (y1 - y0 + 1))
	return s / n
}

// TamuraCoarseness returns the Tamura coarseness of img. For each pixel it finds
// the largest neighbourhood size 2^k (k in [0, kmax]) at which the average
// intensity changes most between opposite sides, then averages that best size
// over the image. Coarser textures — larger, more uniform elements — yield
// larger values. kmax must be >= 1; a typical value is 5 (sizes up to 32x32).
func TamuraCoarseness(img *cv.Mat, kmax int) float64 {
	textureRequire(img, "TamuraCoarseness")
	if kmax < 1 {
		panic(fmt.Sprintf("texture: TamuraCoarseness requires kmax >= 1, got %d", kmax))
	}
	rows, cols := img.Rows, img.Cols
	luma := textureLumaFloat(img)
	ii := textureIntegral(luma, rows, cols)

	var total float64
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			bestS := 1.0
			var bestE float64
			for k := 1; k <= kmax; k++ {
				shift := 1 << uint(k-1) // 2^(k-1)
				w := 1 << uint(k)       // 2^k
				half := w / 2
				// Averages of windows centred shift to each side.
				aRight := rectMean(ii, rows, cols, x+shift-half, y-half, x+shift+half-1, y+half-1)
				aLeft := rectMean(ii, rows, cols, x-shift-half, y-half, x-shift+half-1, y+half-1)
				aDown := rectMean(ii, rows, cols, x-half, y+shift-half, x+half-1, y+shift+half-1)
				aUp := rectMean(ii, rows, cols, x-half, y-shift-half, x+half-1, y-shift+half-1)
				eh := math.Abs(aRight - aLeft)
				ev := math.Abs(aDown - aUp)
				e := math.Max(eh, ev)
				if e > bestE {
					bestE = e
					bestS = float64(w)
				}
			}
			total += bestS
		}
	}
	return total / float64(rows*cols)
}

// TamuraContrast returns the Tamura contrast of img, defined as the standard
// deviation divided by the fourth root of the kurtosis (sigma / alpha4^0.25,
// with alpha4 = mu4/sigma^4). It captures both the spread of intensities and
// how sharply the histogram is polarised. A perfectly flat image returns 0.
func TamuraContrast(img *cv.Mat) float64 {
	textureRequire(img, "TamuraContrast")
	luma := textureLumaFloat(img)
	n := float64(len(luma))
	var mean float64
	for _, v := range luma {
		mean += v
	}
	mean /= n
	var m2, m4 float64
	for _, v := range luma {
		d := v - mean
		d2 := d * d
		m2 += d2
		m4 += d2 * d2
	}
	m2 /= n
	m4 /= n
	if m2 == 0 {
		return 0
	}
	sigma := math.Sqrt(m2)
	alpha4 := m4 / (m2 * m2)
	return sigma / math.Pow(alpha4, 0.25)
}

// TamuraDirectionality returns the Tamura directionality of img in [0,1]. It
// builds a magnitude-weighted histogram of local edge orientations (from Prewitt
// gradients), then measures how concentrated that histogram is about its
// dominant orientation: a strongly oriented texture (stripes) approaches 1,
// while an isotropic texture approaches 0. Only gradients whose magnitude
// exceeds threshold contribute; nbins controls the angular resolution
// (16 is typical). nbins must be >= 2 and threshold >= 0.
func TamuraDirectionality(img *cv.Mat, nbins int, threshold float64) float64 {
	textureRequire(img, "TamuraDirectionality")
	if nbins < 2 {
		panic(fmt.Sprintf("texture: TamuraDirectionality requires nbins >= 2, got %d", nbins))
	}
	if threshold < 0 {
		panic("texture: TamuraDirectionality requires threshold >= 0")
	}
	rows, cols := img.Rows, img.Cols
	luma := textureLumaFloat(img)
	hist := make([]float64, nbins)
	var total float64
	for y := 1; y < rows-1; y++ {
		for x := 1; x < cols-1; x++ {
			// Prewitt gradients.
			gx := (luma[(y-1)*cols+x+1] + luma[y*cols+x+1] + luma[(y+1)*cols+x+1]) -
				(luma[(y-1)*cols+x-1] + luma[y*cols+x-1] + luma[(y+1)*cols+x-1])
			gy := (luma[(y+1)*cols+x-1] + luma[(y+1)*cols+x] + luma[(y+1)*cols+x+1]) -
				(luma[(y-1)*cols+x-1] + luma[(y-1)*cols+x] + luma[(y-1)*cols+x+1])
			mag := (math.Abs(gx) + math.Abs(gy)) / 2
			if mag < threshold {
				continue
			}
			// Orientation of the edge (perpendicular to the gradient), mapped
			// to [0, pi).
			theta := math.Atan2(gy, gx) + math.Pi/2
			for theta < 0 {
				theta += math.Pi
			}
			for theta >= math.Pi {
				theta -= math.Pi
			}
			b := int(theta / math.Pi * float64(nbins))
			if b >= nbins {
				b = nbins - 1
			}
			hist[b] += mag
			total += mag
		}
	}
	if total == 0 {
		return 0
	}
	// Normalise, find the dominant bin, and measure angular concentration.
	peak := 0
	for i := 1; i < nbins; i++ {
		if hist[i] > hist[peak] {
			peak = i
		}
	}
	binW := math.Pi / float64(nbins)
	peakAngle := (float64(peak) + 0.5) * binW
	var spread float64
	for i := 0; i < nbins; i++ {
		a := (float64(i) + 0.5) * binW
		d := math.Abs(a - peakAngle)
		if d > math.Pi-d { // circular distance on [0,pi)
			d = math.Pi - d
		}
		spread += d * d * (hist[i] / total)
	}
	// Maximum possible spread is (pi/2)^2; normalise so Fdir in [0,1].
	maxSpread := (math.Pi / 2) * (math.Pi / 2)
	fdir := 1 - spread/maxSpread
	if fdir < 0 {
		fdir = 0
	}
	if fdir > 1 {
		fdir = 1
	}
	return fdir
}

// TamuraFeatures computes all Tamura features of img with default parameters
// (coarseness kmax=5, directionality nbins=16 threshold=12) and returns them as
// a [TamuraFeatureSet]. Roughness is the sum of coarseness and contrast.
func TamuraFeatures(img *cv.Mat) TamuraFeatureSet {
	crs := TamuraCoarseness(img, 5)
	con := TamuraContrast(img)
	dir := TamuraDirectionality(img, 16, 12)
	return TamuraFeatureSet{
		Coarseness:     crs,
		Contrast:       con,
		Directionality: dir,
		Roughness:      crs + con,
	}
}
