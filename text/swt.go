package text

import (
	"math"
	"sort"

	cv "github.com/malcolmston/opencv"
)

// SWTParams controls the Stroke Width Transform detector (see
// [StrokeWidthTransform] and [TextDetectorSWT]).
type SWTParams struct {
	// DarkOnLight selects the text polarity. When true the detector walks rays
	// against the intensity gradient (dark strokes on a light background); when
	// false it walks along the gradient (bright strokes on a dark background).
	DarkOnLight bool
	// EdgeThreshold is the gradient-magnitude value above which a pixel is treated
	// as a stroke edge and used as a ray origin.
	EdgeThreshold float64
	// MaxStrokeWidth caps how far, in pixels, a ray searches for the opposite edge.
	// Rays that find no matching edge within this distance contribute nothing.
	MaxStrokeWidth int
	// MinComponentArea and MaxComponentArea bound the pixel area of an accepted
	// stroke component (MaxComponentArea <= 0 disables the upper bound).
	MinComponentArea int
	MaxComponentArea int
	// MaxStrokeVariation bounds the ratio of stroke-width standard deviation to
	// mean within a component; text strokes are near-constant in width.
	MaxStrokeVariation float64
	// MinAspect and MaxAspect bound a component's bounding-box aspect ratio.
	MinAspect float64
	MaxAspect float64
}

// DefaultSWTParams returns parameters tuned for bright strokes on a dark
// background (DarkOnLight false) with strokes up to 20 px wide.
func DefaultSWTParams() SWTParams {
	return SWTParams{
		DarkOnLight:        false,
		EdgeThreshold:      50,
		MaxStrokeWidth:     20,
		MinComponentArea:   6,
		MaxComponentArea:   0,
		MaxStrokeVariation: 0.5,
		MinAspect:          0.05,
		MaxAspect:          10,
	}
}

// StrokeWidthTransform computes the Stroke Width Transform of a single-channel
// image following Epshtein, Ofek and Wexler (CVPR 2010). For every edge pixel it
// casts a ray in (or against) the gradient direction; if the ray meets an edge
// pixel whose gradient points roughly the opposite way, the Euclidean length of
// the ray is the stroke width and is written to every pixel it crossed. A second
// pass replaces each ray pixel that exceeds the ray's median width with that
// median, so corners take the local stroke width rather than an inflated diagonal.
//
// The result is a row-major slice of per-pixel stroke widths; unwritten pixels
// (outside any stroke) are 0. A colour image is reduced to grayscale first.
func StrokeWidthTransform(img *cv.Mat, p SWTParams) []float64 {
	gray := toGray(img)
	rows, cols := gray.Rows, gray.Cols
	n := rows * cols

	gx, gy, mag := sobelGradients(gray)
	sign := 1.0
	if p.DarkOnLight {
		sign = -1.0
	}
	maxW := p.MaxStrokeWidth
	if maxW < 1 {
		maxW = 1
	}

	swt := make([]float64, n)
	for i := range swt {
		swt[i] = math.Inf(1)
	}
	isEdge := func(idx int) bool { return mag[idx] >= p.EdgeThreshold }

	// rays records the pixel indices crossed by each accepted ray for the median
	// refinement pass.
	var rays [][]int

	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			i := y*cols + x
			if !isEdge(i) {
				continue
			}
			m := mag[i]
			if m == 0 {
				continue
			}
			dx := sign * gx[i] / m
			dy := sign * gy[i] / m

			ray := []int{i}
			fx, fy := float64(x)+0.5, float64(y)+0.5
			matched := false
			for step := 1; step <= maxW; step++ {
				cx := int(fx + dx*float64(step))
				cy := int(fy + dy*float64(step))
				if cx < 0 || cx >= cols || cy < 0 || cy >= rows {
					break
				}
				j := cy*cols + cx
				if j == ray[len(ray)-1] {
					continue // same pixel, keep advancing
				}
				ray = append(ray, j)
				if isEdge(j) && mag[j] > 0 {
					// Opposite gradient direction (within ~30 degrees) closes a stroke.
					qdx := sign * gx[j] / mag[j]
					qdy := sign * gy[j] / mag[j]
					if dx*qdx+dy*qdy < -0.5 {
						matched = true
					}
					break
				}
			}
			if !matched || len(ray) < 2 {
				continue
			}
			width := math.Hypot(
				float64(ray[len(ray)-1]%cols-ray[0]%cols),
				float64(ray[len(ray)-1]/cols-ray[0]/cols),
			)
			if width <= 0 {
				continue
			}
			for _, idx := range ray {
				if width < swt[idx] {
					swt[idx] = width
				}
			}
			rays = append(rays, ray)
		}
	}

	// Median refinement: clamp each ray pixel to the ray's median stroke width.
	for _, ray := range rays {
		vals := make([]float64, 0, len(ray))
		for _, idx := range ray {
			if !math.IsInf(swt[idx], 1) {
				vals = append(vals, swt[idx])
			}
		}
		if len(vals) == 0 {
			continue
		}
		sort.Float64s(vals)
		med := vals[len(vals)/2]
		for _, idx := range ray {
			if swt[idx] > med {
				swt[idx] = med
			}
		}
	}

	for i := range swt {
		if math.IsInf(swt[i], 1) {
			swt[i] = 0
		}
	}
	return swt
}

// SWTComponent is one stroke-width-connected component found by
// [TextDetectorSWT.DetectComponents]: its bounding box and the mean and standard
// deviation of the stroke width of its pixels.
type SWTComponent struct {
	Rect            cv.Rect
	Area            int
	StrokeWidthMean float64
	StrokeWidthStd  float64
}

// TextDetectorSWT detects character-like regions with the Stroke Width Transform.
// It is the standard-library analogue of OpenCV's cv::text::TextDetectorCNN slot
// filled instead by the classical SWT detector (cv::text does not ship SWT, but
// the transform underpins the same detect-by-stroke-constancy idea).
type TextDetectorSWT struct {
	Params SWTParams
}

// NewTextDetectorSWT returns a detector using the given parameters.
func NewTextDetectorSWT(p SWTParams) *TextDetectorSWT {
	return &TextDetectorSWT{Params: p}
}

// DetectComponents runs the transform and groups pixels of similar stroke width
// (neighbouring widths within a factor of 3) into components, returning those
// that pass the size, aspect and stroke-constancy gates in [SWTParams]. Results
// are ordered top-to-bottom then left-to-right.
func (d *TextDetectorSWT) DetectComponents(img *cv.Mat) []SWTComponent {
	gray := toGray(img)
	rows, cols := gray.Rows, gray.Cols
	swt := StrokeWidthTransform(gray, d.Params)

	uf := newIntUnionFind(rows * cols)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			i := y*cols + x
			if swt[i] <= 0 {
				continue
			}
			// Link to the right and down neighbours of comparable stroke width.
			if x+1 < cols {
				j := i + 1
				if swt[j] > 0 && strokeRatio(swt[i], swt[j]) <= 3.0 {
					uf.union(i, j)
				}
			}
			if y+1 < rows {
				j := i + cols
				if swt[j] > 0 && strokeRatio(swt[i], swt[j]) <= 3.0 {
					uf.union(i, j)
				}
			}
		}
	}

	type agg struct {
		area                   int
		minX, minY, maxX, maxY int
		sum, sumSq             float64
	}
	comps := map[int]*agg{}
	var roots []int
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			i := y*cols + x
			if swt[i] <= 0 {
				continue
			}
			r := uf.find(i)
			a := comps[r]
			if a == nil {
				a = &agg{minX: x, minY: y, maxX: x, maxY: y}
				comps[r] = a
				roots = append(roots, r)
			}
			a.area++
			a.minX, a.maxX = minInt(a.minX, x), maxInt(a.maxX, x)
			a.minY, a.maxY = minInt(a.minY, y), maxInt(a.maxY, y)
			a.sum += swt[i]
			a.sumSq += swt[i] * swt[i]
		}
	}
	sort.Ints(roots)

	p := d.Params
	var out []SWTComponent
	for _, r := range roots {
		a := comps[r]
		if a.area < p.MinComponentArea {
			continue
		}
		if p.MaxComponentArea > 0 && a.area > p.MaxComponentArea {
			continue
		}
		w := a.maxX - a.minX + 1
		h := a.maxY - a.minY + 1
		aspect := float64(w) / float64(h)
		if aspect < p.MinAspect || aspect > p.MaxAspect {
			continue
		}
		mean := a.sum / float64(a.area)
		variance := a.sumSq/float64(a.area) - mean*mean
		if variance < 0 {
			variance = 0
		}
		std := math.Sqrt(variance)
		if mean > 0 && std/mean > p.MaxStrokeVariation {
			continue
		}
		out = append(out, SWTComponent{
			Rect:            cv.Rect{X: a.minX, Y: a.minY, Width: w, Height: h},
			Area:            a.area,
			StrokeWidthMean: mean,
			StrokeWidthStd:  std,
		})
	}
	sort.SliceStable(out, func(i, j int) bool {
		if out[i].Rect.Y != out[j].Rect.Y {
			return out[i].Rect.Y < out[j].Rect.Y
		}
		return out[i].Rect.X < out[j].Rect.X
	})
	return out
}

// Detect runs [TextDetectorSWT.DetectComponents] and returns just the bounding
// boxes of the stroke components.
func (d *TextDetectorSWT) Detect(img *cv.Mat) []cv.Rect {
	comps := d.DetectComponents(img)
	boxes := make([]cv.Rect, len(comps))
	for i, c := range comps {
		boxes[i] = c.Rect
	}
	return boxes
}

// strokeRatio returns the ratio of the larger to the smaller of two positive
// stroke widths.
func strokeRatio(a, b float64) float64 {
	if a < b {
		a, b = b, a
	}
	if b <= 0 {
		return math.Inf(1)
	}
	return a / b
}

// sobelGradients returns the horizontal and vertical Sobel derivatives and the
// gradient magnitude of a single-channel image, each as a row-major float slice.
// Borders use edge replication.
func sobelGradients(gray *cv.Mat) (gx, gy, mag []float64) {
	rows, cols := gray.Rows, gray.Cols
	n := rows * cols
	gx = make([]float64, n)
	gy = make([]float64, n)
	mag = make([]float64, n)
	at := func(y, x int) float64 {
		if y < 0 {
			y = 0
		} else if y >= rows {
			y = rows - 1
		}
		if x < 0 {
			x = 0
		} else if x >= cols {
			x = cols - 1
		}
		return float64(gray.Data[y*cols+x])
	}
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			i := y*cols + x
			gx[i] = -at(y-1, x-1) - 2*at(y, x-1) - at(y+1, x-1) +
				at(y-1, x+1) + 2*at(y, x+1) + at(y+1, x+1)
			gy[i] = -at(y-1, x-1) - 2*at(y-1, x) - at(y-1, x+1) +
				at(y+1, x-1) + 2*at(y+1, x) + at(y+1, x+1)
			mag[i] = math.Hypot(gx[i], gy[i])
		}
	}
	return gx, gy, mag
}
