package saliency2

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// GaussianSmooth returns a Gaussian-blurred copy of m. The kernel size is
// derived from sigma (about 6*sigma, forced odd) and the border sample is
// replicated. Smoothing suppresses pixel noise so nearby salient responses
// merge into coherent blobs.
func GaussianSmooth(m *SaliencyMap, sigma float64) *SaliencyMap {
	if sigma <= 0 {
		return m.Clone()
	}
	ksize := int(math.Ceil(sigma*3))*2 + 1
	return saliency2GaussianBlurMap(m, ksize, sigma)
}

// BoxSmooth returns a box-averaged copy of m with the given window radius,
// computed in linear time with an integral image.
func BoxSmooth(m *SaliencyMap, radius int) *SaliencyMap {
	return saliency2BoxBlurMap(m, radius)
}

// NormalizeRange returns a copy of m linearly rescaled to the closed interval
// [lo, hi]. It is the range-targeted companion to [SaliencyMap.Normalize],
// which always targets [0,1].
func NormalizeRange(m *SaliencyMap, lo, hi float64) *SaliencyMap {
	min, max := m.MinMax()
	out := NewSaliencyMap(m.Rows, m.Cols)
	rng := max - min
	if rng <= 0 {
		for i := range out.Data {
			out.Data[i] = lo
		}
		return out
	}
	for i, v := range m.Data {
		out.Data[i] = lo + (v-min)/rng*(hi-lo)
	}
	return out
}

// IttiNormalize applies Itti, Koch and Niebur's normalisation operator N(.) to
// m: the map is scaled to [0,1], then multiplied by (1 - mean)^2 where mean is
// the average of its local maxima (excluding the single global maximum). A map
// with one dominant peak is promoted; a map with many comparable peaks is
// suppressed. This is the operator used to combine feature maps in the
// Itti-Koch model.
func IttiNormalize(m *SaliencyMap) *SaliencyMap {
	n := m.Normalize()
	rows, cols := n.Rows, n.Cols
	var sum float64
	var count int
	globalMax := 0.0
	for _, v := range n.Data {
		if v > globalMax {
			globalMax = v
		}
	}
	seenGlobal := false
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			v := n.Data[y*cols+x]
			isMax := true
			for dy := -1; dy <= 1 && isMax; dy++ {
				for dx := -1; dx <= 1; dx++ {
					if dx == 0 && dy == 0 {
						continue
					}
					yy := y + dy
					xx := x + dx
					if yy < 0 || yy >= rows || xx < 0 || xx >= cols {
						continue
					}
					if n.Data[yy*cols+xx] > v {
						isMax = false
						break
					}
				}
			}
			if !isMax {
				continue
			}
			if !seenGlobal && v >= globalMax {
				seenGlobal = true
				continue
			}
			sum += v
			count++
		}
	}
	factor := 1.0
	if count > 0 {
		mean := sum / float64(count)
		factor = (globalMax - mean) * (globalMax - mean)
	}
	return n.Scale(factor)
}

// CenterPrior multiplies m by an isotropic Gaussian centre prior, damping
// responses toward the image border. sigmaFrac sets the standard deviation as a
// fraction of the half-diagonal (0.5 is a mild prior). Human fixations cluster
// near the image centre, so this bias improves most saliency maps.
func CenterPrior(m *SaliencyMap, sigmaFrac float64) *SaliencyMap {
	if sigmaFrac <= 0 {
		return m.Clone()
	}
	rows, cols := m.Rows, m.Cols
	cy := float64(rows-1) / 2
	cx := float64(cols-1) / 2
	half := math.Hypot(cx, cy)
	sigma := sigmaFrac * half
	if sigma <= 0 {
		return m.Clone()
	}
	inv := 1 / (2 * sigma * sigma)
	out := NewSaliencyMap(rows, cols)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			dy := float64(y) - cy
			dx := float64(x) - cx
			g := math.Exp(-(dx*dx + dy*dy) * inv)
			out.Data[y*cols+x] = m.Data[y*cols+x] * g
		}
	}
	return out
}

// GammaCorrect returns a copy of m with each value raised to the power gamma
// after normalisation to [0,1]. gamma > 1 sharpens the map toward its peaks;
// gamma < 1 lifts weak responses.
func GammaCorrect(m *SaliencyMap, gamma float64) *SaliencyMap {
	n := m.Normalize()
	for i, v := range n.Data {
		n.Data[i] = math.Pow(v, gamma)
	}
	return n
}

// LogScale returns a copy of m compressed by log(1+v) after shifting so its
// minimum is zero. It tames maps with a few very large outliers.
func LogScale(m *SaliencyMap) *SaliencyMap {
	min, _ := m.MinMax()
	out := NewSaliencyMap(m.Rows, m.Cols)
	for i, v := range m.Data {
		out.Data[i] = math.Log1p(v - min)
	}
	return out
}

// CombineMaps averages several saliency maps into one, after normalising each
// to [0,1] so they contribute equally. All maps must share the same
// dimensions. It panics if the slice is empty or the sizes differ.
func CombineMaps(maps ...*SaliencyMap) *SaliencyMap {
	if len(maps) == 0 {
		panic("saliency2: CombineMaps requires at least one map")
	}
	rows, cols := maps[0].Rows, maps[0].Cols
	out := NewSaliencyMap(rows, cols)
	for _, mp := range maps {
		if mp.Rows != rows || mp.Cols != cols {
			panic("saliency2: CombineMaps size mismatch")
		}
		n := mp.Normalize()
		for i := range out.Data {
			out.Data[i] += n.Data[i]
		}
	}
	inv := 1 / float64(len(maps))
	for i := range out.Data {
		out.Data[i] *= inv
	}
	return out
}

// SalientMask returns the adaptive binary segmentation of m using the
// mean-based rule of Achanta et al.: a pixel is foreground when its value is at
// least twice the map mean. It is a convenience wrapper over
// [SaliencyMap.MeanThreshold] with k=2.
func SalientMask(m *SaliencyMap) *cv.Mat {
	mask, _ := m.MeanThreshold(2.0)
	return mask
}

// HeatmapOverlay blends a colourised saliency map over a copy of img and
// returns the three-channel result. alpha (in [0,1]) is the saliency weight:
// 0 returns the image unchanged, 1 shows the pure heat colours. Salient regions
// are tinted from blue (low) through green to red (high). The saliency map is
// resized to the image size if necessary.
func HeatmapOverlay(img *cv.Mat, m *SaliencyMap, alpha float64) *cv.Mat {
	alpha = saliency2ClampFloat(alpha, 0, 1)
	base := img
	if img.Channels != 3 {
		base = cv.CvtColor(img, cv.ColorGray2RGB)
	}
	sm := m
	if m.Rows != base.Rows || m.Cols != base.Cols {
		sm = saliency2ResizeMap(m, base.Rows, base.Cols)
	}
	norm := sm.Normalize()
	out := cv.NewMat(base.Rows, base.Cols, 3)
	n := base.Rows * base.Cols
	for i := 0; i < n; i++ {
		hr, hg, hb := saliency2HeatColor(norm.Data[i])
		for c, hv := range [3]float64{hr, hg, hb} {
			orig := float64(base.Data[i*3+c])
			v := (1-alpha)*orig + alpha*hv
			out.Data[i*3+c] = uint8(math.Round(saliency2ClampFloat(v, 0, 255)))
		}
	}
	return out
}

// saliency2HeatColor maps a value in [0,1] to an RGB "jet"-like colour.
func saliency2HeatColor(v float64) (r, g, b float64) {
	v = saliency2ClampFloat(v, 0, 1)
	switch {
	case v < 0.25:
		return 0, 255 * (v / 0.25), 255
	case v < 0.5:
		return 0, 255, 255 * (1 - (v-0.25)/0.25)
	case v < 0.75:
		return 255 * ((v - 0.5) / 0.25), 255, 0
	default:
		return 255, 255 * (1 - (v-0.75)/0.25), 0
	}
}
