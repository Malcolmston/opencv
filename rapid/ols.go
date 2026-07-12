package rapid

import cv "github.com/malcolmston/opencv"

// OLSTracker is the optimized-local-search RAPID variant. It thresholds edge
// responses with a Sobel-style cutoff and biases the per-line search toward the
// displacements it has accepted so far, learned online in a histogram. This
// stabilises tracking when several competing edges lie within a search line.
type OLSTracker struct {
	*Rapid
}

// NewOLSTracker creates an [OLSTracker] for the given mesh. histBins is the
// number of displacement bins in the learned prior histogram and sobelThresh is
// the minimum edge response accepted. It mirrors cv::rapid::OLSTracker::create.
func NewOLSTracker(mesh *Mesh, histBins int, sobelThresh float64) *OLSTracker {
	if histBins < 1 {
		histBins = 8
	}
	return &OLSTracker{Rapid: &Rapid{
		mesh:        mesh,
		strategy:    &olsSearch{histBins: histBins, thresh: sobelThresh},
		minResponse: sobelThresh,
	}}
}

// olsSearch is the stateful optimized-local-search strategy.
type olsSearch struct {
	histBins int
	thresh   float64
	hist     []float64 // learned displacement prior, one weight per bin
	total    float64
}

func (o *olsSearch) clear() {
	o.hist = nil
	o.total = 0
}

// bin maps a signed displacement in [-half, half] to a histogram bin.
func (o *olsSearch) bin(disp, half int) int {
	if half == 0 {
		return 0
	}
	b := int(float64(disp+half) / float64(2*half+1) * float64(o.histBins))
	if b < 0 {
		b = 0
	}
	if b >= o.histBins {
		b = o.histBins - 1
	}
	return b
}

func (o *olsSearch) search(bundle *cv.Mat) ([]int, []float64) {
	if bundle == nil || bundle.Rows == 0 {
		return nil, nil
	}
	if o.hist == nil {
		o.hist = make([]float64, o.histBins)
	}
	w := bundle.Cols
	half := w / 2
	cols := make([]int, bundle.Rows)
	resp := make([]float64, bundle.Rows)
	// priorScale sizes the learned bias relative to gradient magnitudes (0..255).
	const priorScale = 40.0
	for i := 0; i < bundle.Rows; i++ {
		prof := rowGradient(bundle, i)
		best := -1.0
		bestj := half
		for j := 0; j < w; j++ {
			g := prof[j]
			if g < o.thresh {
				continue
			}
			var prior float64
			if o.total > 0 {
				prior = priorScale * o.hist[o.bin(j-half, half)] / o.total
			}
			if score := g + prior; score > best {
				best = score
				bestj = j
			}
		}
		cols[i] = bestj
		resp[i] = prof[bestj]
		if prof[bestj] >= o.thresh {
			o.hist[o.bin(bestj-half, half)]++
			o.total++
		}
	}
	return cols, resp
}
