package saliency2

import (
	"image"
	"math"
	"sort"

	cv "github.com/malcolmston/opencv"
)

// SaliencyMap is a single-channel grid of float64 saliency values. It is the
// common working and result type of the package: detectors compute it, the
// post-processing operators transform it, and it converts to a displayable
// 8-bit [cv.Mat] with [SaliencyMap.ToMat].
//
// Values are stored row-major in Data: the value at row y, column x is at
// Data[y*Cols+x]. Higher values mark more salient locations, but the range is
// not fixed — raw detector output can span any scale until it is normalised.
type SaliencyMap struct {
	// Rows is the map height.
	Rows int
	// Cols is the map width.
	Cols int
	// Data holds Rows*Cols values in row-major order.
	Data []float64
}

// NewSaliencyMap allocates a zero-filled map of the given size. It panics if
// either dimension is not positive.
func NewSaliencyMap(rows, cols int) *SaliencyMap {
	if rows <= 0 || cols <= 0 {
		panic("saliency2: NewSaliencyMap requires positive dimensions")
	}
	return &SaliencyMap{Rows: rows, Cols: cols, Data: make([]float64, rows*cols)}
}

// SaliencyMapFromMat builds a map from the luminance of img, with values in
// [0,255]. Multi-channel input is converted to grayscale first.
func SaliencyMapFromMat(img *cv.Mat) *SaliencyMap {
	return saliency2GrayFloat(img)
}

// SaliencyMapFromFloatMat copies the parent package's [cv.FloatMat] into a
// SaliencyMap of the same size.
func SaliencyMapFromFloatMat(f *cv.FloatMat) *SaliencyMap {
	if f == nil {
		panic("saliency2: nil FloatMat")
	}
	out := NewSaliencyMap(f.Rows, f.Cols)
	copy(out.Data, f.Data)
	return out
}

// At returns the value at row y, column x. It panics on out-of-range access.
func (m *SaliencyMap) At(y, x int) float64 {
	if y < 0 || y >= m.Rows || x < 0 || x >= m.Cols {
		panic("saliency2: SaliencyMap.At out of range")
	}
	return m.Data[y*m.Cols+x]
}

// Set stores value at row y, column x. It panics on out-of-range access.
func (m *SaliencyMap) Set(y, x int, value float64) {
	if y < 0 || y >= m.Rows || x < 0 || x >= m.Cols {
		panic("saliency2: SaliencyMap.Set out of range")
	}
	m.Data[y*m.Cols+x] = value
}

// Size returns the map dimensions as (rows, cols).
func (m *SaliencyMap) Size() (rows, cols int) {
	return m.Rows, m.Cols
}

// Clone returns a deep copy of the map with its own backing storage.
func (m *SaliencyMap) Clone() *SaliencyMap {
	out := &SaliencyMap{Rows: m.Rows, Cols: m.Cols, Data: make([]float64, len(m.Data))}
	copy(out.Data, m.Data)
	return out
}

// MinMax returns the smallest and largest values in the map.
func (m *SaliencyMap) MinMax() (min, max float64) {
	min, max = math.Inf(1), math.Inf(-1)
	for _, v := range m.Data {
		if v < min {
			min = v
		}
		if v > max {
			max = v
		}
	}
	return min, max
}

// Sum returns the total of all values in the map.
func (m *SaliencyMap) Sum() float64 {
	var s float64
	for _, v := range m.Data {
		s += v
	}
	return s
}

// Mean returns the arithmetic mean of all values in the map.
func (m *SaliencyMap) Mean() float64 {
	if len(m.Data) == 0 {
		return 0
	}
	return m.Sum() / float64(len(m.Data))
}

// StdDev returns the population standard deviation of the map's values.
func (m *SaliencyMap) StdDev() float64 {
	if len(m.Data) == 0 {
		return 0
	}
	mean := m.Mean()
	var acc float64
	for _, v := range m.Data {
		d := v - mean
		acc += d * d
	}
	return math.Sqrt(acc / float64(len(m.Data)))
}

// Normalize returns a copy of the map linearly rescaled so its minimum maps to
// 0 and its maximum to 1. A constant map yields all zeros.
func (m *SaliencyMap) Normalize() *SaliencyMap {
	min, max := m.MinMax()
	out := NewSaliencyMap(m.Rows, m.Cols)
	rng := max - min
	if rng <= 0 {
		return out
	}
	for i, v := range m.Data {
		out.Data[i] = (v - min) / rng
	}
	return out
}

// Scale returns a copy of the map with every value multiplied by factor.
func (m *SaliencyMap) Scale(factor float64) *SaliencyMap {
	out := NewSaliencyMap(m.Rows, m.Cols)
	for i, v := range m.Data {
		out.Data[i] = v * factor
	}
	return out
}

// AddMap returns the elementwise sum of m and other, which must have the same
// dimensions.
func (m *SaliencyMap) AddMap(other *SaliencyMap) *SaliencyMap {
	m.requireSameSize(other, "AddMap")
	out := NewSaliencyMap(m.Rows, m.Cols)
	for i := range m.Data {
		out.Data[i] = m.Data[i] + other.Data[i]
	}
	return out
}

// MultiplyMap returns the elementwise product of m and other, which must have
// the same dimensions.
func (m *SaliencyMap) MultiplyMap(other *SaliencyMap) *SaliencyMap {
	m.requireSameSize(other, "MultiplyMap")
	out := NewSaliencyMap(m.Rows, m.Cols)
	for i := range m.Data {
		out.Data[i] = m.Data[i] * other.Data[i]
	}
	return out
}

func (m *SaliencyMap) requireSameSize(other *SaliencyMap, name string) {
	if other == nil || m.Rows != other.Rows || m.Cols != other.Cols {
		panic("saliency2: SaliencyMap." + name + " size mismatch")
	}
}

// ToMat returns an 8-bit single-channel [cv.Mat] rendering of the map, min-max
// normalised so the smallest value becomes 0 and the largest 255. This is the
// standard displayable saliency map.
func (m *SaliencyMap) ToMat() *cv.Mat {
	min, max := m.MinMax()
	out := cv.NewMat(m.Rows, m.Cols, 1)
	rng := max - min
	if rng <= 0 {
		return out
	}
	for i, v := range m.Data {
		s := (v - min) / rng * 255
		out.Data[i] = uint8(math.Round(saliency2ClampFloat(s, 0, 255)))
	}
	return out
}

// ToFloatMat returns a copy of the map as the parent package's [cv.FloatMat].
func (m *SaliencyMap) ToFloatMat() *cv.FloatMat {
	out := cv.NewFloatMat(m.Rows, m.Cols)
	copy(out.Data, m.Data)
	return out
}

func saliency2ClampFloat(v, lo, hi float64) float64 {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

// Threshold returns a binary [cv.Mat] in which pixels whose value is greater
// than or equal to t are 255 and all others 0.
func (m *SaliencyMap) Threshold(t float64) *cv.Mat {
	out := cv.NewMat(m.Rows, m.Cols, 1)
	for i, v := range m.Data {
		if v >= t {
			out.Data[i] = 255
		}
	}
	return out
}

// MeanThreshold returns the binary mask produced by thresholding the map at k
// times its mean value — the adaptive rule of Achanta et al., which uses k=2.
// It returns both the mask and the threshold that was applied.
func (m *SaliencyMap) MeanThreshold(k float64) (mask *cv.Mat, threshold float64) {
	threshold = k * m.Mean()
	return m.Threshold(threshold), threshold
}

// PercentileThreshold returns the binary mask keeping the fraction p (in
// [0,1]) of the highest-valued pixels, together with the threshold value used.
// p=0.1 keeps roughly the top 10 percent.
func (m *SaliencyMap) PercentileThreshold(p float64) (mask *cv.Mat, threshold float64) {
	p = saliency2ClampFloat(p, 0, 1)
	sorted := make([]float64, len(m.Data))
	copy(sorted, m.Data)
	sort.Float64s(sorted)
	idx := int(math.Floor((1 - p) * float64(len(sorted)-1)))
	idx = saliency2ClampInt(idx, 0, len(sorted)-1)
	threshold = sorted[idx]
	return m.Threshold(threshold), threshold
}

// OtsuThreshold binarises the map using Otsu's method, which chooses the
// threshold that maximises between-class variance over a 256-bin histogram of
// the value range. It returns the mask and the threshold in the map's own
// units.
func (m *SaliencyMap) OtsuThreshold() (mask *cv.Mat, threshold float64) {
	min, max := m.MinMax()
	rng := max - min
	if rng <= 0 {
		return cv.NewMat(m.Rows, m.Cols, 1), min
	}
	const bins = 256
	hist := make([]float64, bins)
	for _, v := range m.Data {
		b := int((v - min) / rng * (bins - 1))
		b = saliency2ClampInt(b, 0, bins-1)
		hist[b]++
	}
	total := float64(len(m.Data))
	var sumAll float64
	for i, h := range hist {
		sumAll += float64(i) * h
	}
	var sumB, wB, best float64
	bestBin := 0
	for i := 0; i < bins; i++ {
		wB += hist[i]
		if wB == 0 {
			continue
		}
		wF := total - wB
		if wF == 0 {
			break
		}
		sumB += float64(i) * hist[i]
		mB := sumB / wB
		mF := (sumAll - sumB) / wF
		between := wB * wF * (mB - mF) * (mB - mF)
		if between > best {
			best = between
			bestBin = i
		}
	}
	threshold = min + (float64(bestBin)+0.5)/(bins-1)*rng
	return m.Threshold(threshold), threshold
}

// CenterOfMass returns the value-weighted centroid (x, y) of the map, treating
// values below zero as zero. For an all-zero map it returns the geometric
// centre.
func (m *SaliencyMap) CenterOfMass() (x, y float64) {
	var sw, sx, sy float64
	for r := 0; r < m.Rows; r++ {
		for c := 0; c < m.Cols; c++ {
			w := m.Data[r*m.Cols+c]
			if w < 0 {
				w = 0
			}
			sw += w
			sx += w * float64(c)
			sy += w * float64(r)
		}
	}
	if sw == 0 {
		return float64(m.Cols-1) / 2, float64(m.Rows-1) / 2
	}
	return sx / sw, sy / sw
}

// BoundingBox returns the smallest axis-aligned rectangle enclosing every pixel
// whose value is greater than or equal to t. The ok result is false when no
// pixel meets the threshold, in which case the rectangle is empty.
func (m *SaliencyMap) BoundingBox(t float64) (rect image.Rectangle, ok bool) {
	minX, minY := m.Cols, m.Rows
	maxX, maxY := -1, -1
	for y := 0; y < m.Rows; y++ {
		for x := 0; x < m.Cols; x++ {
			if m.Data[y*m.Cols+x] >= t {
				if x < minX {
					minX = x
				}
				if x > maxX {
					maxX = x
				}
				if y < minY {
					minY = y
				}
				if y > maxY {
					maxY = y
				}
			}
		}
	}
	if maxX < 0 {
		return image.Rectangle{}, false
	}
	return image.Rect(minX, minY, maxX+1, maxY+1), true
}
