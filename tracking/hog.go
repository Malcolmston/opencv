package tracking

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// HOGCells computes a Histogram-of-Oriented-Gradients feature stack for the
// single-channel image m. The image is divided into cellSize×cellSize cells; in
// each cell the gradient magnitude of every pixel is accumulated into one of
// bins unsigned orientation bins (spanning 0–180°) with linear interpolation
// between adjacent bins, and the resulting per-cell histogram is L2-normalised.
//
// It returns bins feature planes, each a row-major slice of length
// cellRows*cellCols (the cell grid), plus that grid's dimensions. HOG gives the
// [TrackerKCFHOG] tracker illumination-robust, edge-oriented features instead of
// raw pixels. m must be single-channel and at least cellSize in each dimension.
func HOGCells(m *cv.Mat, cellSize, bins int) (planes [][]float64, cellRows, cellCols int) {
	if m.Channels != 1 {
		panic("tracking: HOGCells requires a single-channel image")
	}
	if cellSize < 1 || bins < 1 {
		panic("tracking: HOGCells requires cellSize>=1 and bins>=1")
	}
	cellRows = m.Rows / cellSize
	cellCols = m.Cols / cellSize
	if cellRows < 1 || cellCols < 1 {
		panic("tracking: HOGCells image smaller than one cell")
	}
	planes = make([][]float64, bins)
	for b := range planes {
		planes[b] = make([]float64, cellRows*cellCols)
	}
	binW := 180.0 / float64(bins)
	at := func(y, x int) float64 {
		y = clampInt(y, 0, m.Rows-1)
		x = clampInt(x, 0, m.Cols-1)
		return float64(m.Data[y*m.Cols+x])
	}
	for cy := 0; cy < cellRows; cy++ {
		for cx := 0; cx < cellCols; cx++ {
			for py := 0; py < cellSize; py++ {
				for px := 0; px < cellSize; px++ {
					y := cy*cellSize + py
					x := cx*cellSize + px
					gx := at(y, x+1) - at(y, x-1)
					gy := at(y+1, x) - at(y-1, x)
					mag := math.Hypot(gx, gy)
					if mag == 0 {
						continue
					}
					ang := math.Atan2(gy, gx) * 180 / math.Pi
					if ang < 0 {
						ang += 180
					}
					if ang >= 180 {
						ang -= 180
					}
					// Linear interpolation between the two nearest bins.
					pos := ang/binW - 0.5
					b0 := int(math.Floor(pos))
					frac := pos - float64(b0)
					lo := ((b0 % bins) + bins) % bins
					hi := (lo + 1) % bins
					idx := cy*cellCols + cx
					planes[lo][idx] += mag * (1 - frac)
					planes[hi][idx] += mag * frac
				}
			}
		}
	}
	// L2 normalise each cell's histogram across bins.
	for idx := 0; idx < cellRows*cellCols; idx++ {
		var ss float64
		for b := 0; b < bins; b++ {
			ss += planes[b][idx] * planes[b][idx]
		}
		norm := math.Sqrt(ss) + 1e-6
		for b := 0; b < bins; b++ {
			planes[b][idx] /= norm
		}
	}
	return planes, cellRows, cellCols
}

// TrackerKCFHOG is a kernelised correlation filter ([TrackerDCF]'s algorithm)
// driven by multi-channel HOG features ([HOGCells]) rather than raw pixels, with
// a multi-scale search. Learning coefficients and detection use the Gaussian
// kernel summed over all HOG orientation channels, which makes the tracker
// robust to illumination change and gives it the discriminative power of the
// canonical KCF/HOG combination. It tracks translation and scale.
//
// Construct it with [NewTrackerKCFHOG].
type TrackerKCFHOG struct {
	// ModelSize is the power-of-two side length the window is resized to before
	// HOG extraction; ModelSize/CellSize must be a power of two.
	ModelSize int
	// CellSize is the HOG cell size in model pixels.
	CellSize int
	// Bins is the number of unsigned orientation bins (feature channels).
	Bins int
	// Padding scales the search window relative to the object box.
	Padding float64
	// Lambda is the ridge regularisation.
	Lambda float64
	// KernelSigma is the Gaussian-kernel bandwidth.
	KernelSigma float64
	// OutputSigmaFactor sets the Gaussian target width as a fraction of the cell
	// grid side.
	OutputSigmaFactor float64
	// LearnRate blends the new model into the old each frame.
	LearnRate float64
	// Scales are the multiplicative scale hypotheses tried each frame.
	Scales []float64
	// ScalePenalty multiplies non-unit-scale responses.
	ScalePenalty float64
	// MinResponse is the peak response below which Update reports low confidence.
	MinResponse float64

	modelXF []*ComplexMat
	alphaF  *ComplexMat
	hann    []float64
	grid    int // cell grid side (ModelSize/CellSize)
	cx, cy  float64
	w, h    float64
	inited  bool
}

// NewTrackerKCFHOG returns a TrackerKCFHOG with sensible defaults (ModelSize 64,
// CellSize 4, Bins 9, Padding 2.0, multi-scale {0.95,1,1.05}).
func NewTrackerKCFHOG() *TrackerKCFHOG {
	return &TrackerKCFHOG{
		ModelSize:         64,
		CellSize:          4,
		Bins:              9,
		Padding:           2.0,
		Lambda:            1e-4,
		KernelSigma:       0.6,
		OutputSigmaFactor: 1.0 / 12.0,
		LearnRate:         0.075,
		Scales:            []float64{0.95, 1.0, 1.05},
		ScalePenalty:      0.99,
		MinResponse:       0.1,
	}
}

func (t *TrackerKCFHOG) winSize(scale float64) int {
	w := int(math.Round(t.w * t.Padding * scale))
	if w < t.CellSize {
		w = t.CellSize
	}
	return w
}

// features extracts the HOG feature spectra of the window of size win centred on
// (cx, cy), Hann-windowed over the cell grid.
func (t *TrackerKCFHOG) features(gray *cv.Mat, cx, cy float64, win int) []*ComplexMat {
	patch := cropResizeGray(gray, cx, cy, win, t.ModelSize)
	planes, cr, cc := HOGCells(patch, t.CellSize, t.Bins)
	out := make([]*ComplexMat, len(planes))
	for b, p := range planes {
		for i := range p {
			p[i] *= t.hann[i]
		}
		out[b] = FFT2(RealToComplex(p, cr, cc))
	}
	return out
}

func (t *TrackerKCFHOG) train(xf []*ComplexMat, yf *ComplexMat) *ComplexMat {
	kf := gaussianCorrelation(xf, xf, t.KernelSigma)
	alpha := NewComplexMat(t.grid, t.grid)
	lam := complex(t.Lambda, 0)
	for i := range alpha.Data {
		alpha.Data[i] = yf.Data[i] / (kf.Data[i] + lam)
	}
	return alpha
}

// Init trains the initial HOG-KCF from the object inside bbox.
func (t *TrackerKCFHOG) Init(frame *cv.Mat, bbox cv.Rect) {
	if !isPow2(t.ModelSize) {
		t.ModelSize = NextPow2(t.ModelSize)
	}
	t.grid = t.ModelSize / t.CellSize
	if !isPow2(t.grid) {
		panic("tracking: TrackerKCFHOG requires ModelSize/CellSize to be a power of two")
	}
	gray := toGray(frame)
	b := clampRect(bbox, gray.Rows, gray.Cols)
	t.w, t.h = float64(b.Width), float64(b.Height)
	t.cx, t.cy = rectCenter(b)
	t.hann = HannWindow2D(t.grid, t.grid)

	sigma := float64(t.grid) * t.OutputSigmaFactor
	yf := FFT2(RealToComplex(gaussianResponseOrigin(t.grid, t.grid, sigma), t.grid, t.grid))
	xf := t.features(gray, t.cx, t.cy, t.winSize(1.0))
	t.modelXF = xf
	t.alphaF = t.train(xf, yf)
	t.inited = true
}

func (t *TrackerKCFHOG) detect(zf []*ComplexMat) []float64 {
	kzf := gaussianCorrelation(zf, t.modelXF, t.KernelSigma)
	spec := NewComplexMat(t.grid, t.grid)
	for i := range spec.Data {
		spec.Data[i] = t.alphaF.Data[i] * kzf.Data[i]
	}
	return IFFT2(spec).Real()
}

// UpdateConfidence locates the object across scales and returns the new box and
// the peak response as confidence. It panics before Init.
func (t *TrackerKCFHOG) UpdateConfidence(frame *cv.Mat) (cv.Rect, float64) {
	if !t.inited {
		panic("tracking: TrackerKCFHOG.Update called before Init")
	}
	gray := toGray(frame)
	g := t.grid

	bestVal := math.Inf(-1)
	var bestResp []float64
	bestScale := 1.0
	for _, s := range t.Scales {
		zf := t.features(gray, t.cx, t.cy, t.winSize(s))
		resp := t.detect(zf)
		_, _, val := peakLoc(resp, g, g)
		if s != 1.0 {
			val *= t.ScalePenalty
		}
		if val > bestVal {
			bestVal, bestResp, bestScale = val, resp, s
		}
	}

	px, py, _ := peakLoc(bestResp, g, g)
	xl := bestResp[py*g+(px-1+g)%g]
	xr := bestResp[py*g+(px+1)%g]
	yt := bestResp[((py-1+g)%g)*g+px]
	yb := bestResp[((py+1)%g)*g+px]
	dcx := wrapCoord(px, g) + subPixel(xl, bestResp[py*g+px], xr)
	dcy := wrapCoord(py, g) + subPixel(yt, bestResp[py*g+px], yb)

	// One cell shift equals CellSize model pixels; convert to image pixels.
	scaleBack := float64(t.CellSize) * float64(t.winSize(bestScale)) / float64(t.ModelSize)
	t.cx += dcx * scaleBack
	t.cy += dcy * scaleBack
	if bestScale != 1.0 {
		t.w *= 1 + (bestScale-1)*0.75
		t.h *= 1 + (bestScale-1)*0.75
	}
	t.cx = math.Max(0, math.Min(t.cx, float64(gray.Cols-1)))
	t.cy = math.Max(0, math.Min(t.cy, float64(gray.Rows-1)))

	if bestVal >= t.MinResponse {
		sigma := float64(g) * t.OutputSigmaFactor
		yf := FFT2(RealToComplex(gaussianResponseOrigin(g, g, sigma), g, g))
		xf := t.features(gray, t.cx, t.cy, t.winSize(1.0))
		newAlpha := t.train(xf, yf)
		lr := complex(t.LearnRate, 0)
		ilr := complex(1-t.LearnRate, 0)
		for i := range t.alphaF.Data {
			t.alphaF.Data[i] = ilr*t.alphaF.Data[i] + lr*newAlpha.Data[i]
		}
		for c := range t.modelXF {
			for i := range t.modelXF[c].Data {
				t.modelXF[c].Data[i] = ilr*t.modelXF[c].Data[i] + lr*xf[c].Data[i]
			}
		}
	}
	return t.box(gray), bestVal
}

// Update satisfies [Tracker]; the flag is true when the peak reaches MinResponse.
func (t *TrackerKCFHOG) Update(frame *cv.Mat) (cv.Rect, bool) {
	box, conf := t.UpdateConfidence(frame)
	return box, conf >= t.MinResponse
}

func (t *TrackerKCFHOG) box(gray *cv.Mat) cv.Rect {
	r := cv.Rect{
		X:      int(math.Round(t.cx - t.w/2)),
		Y:      int(math.Round(t.cy - t.h/2)),
		Width:  int(math.Round(t.w)),
		Height: int(math.Round(t.h)),
	}
	return clampRect(r, gray.Rows, gray.Cols)
}
