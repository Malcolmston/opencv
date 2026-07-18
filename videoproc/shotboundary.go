package videoproc

import (
	cv "github.com/malcolmston/opencv"
)

// videoprocGrayHist returns the 256-bin grayscale intensity histogram of a
// frame, normalised so the bins sum to 1.
func videoprocGrayHist(frame *cv.Mat) [256]float64 {
	g := videoprocToGray(frame)
	var hist [256]float64
	for _, v := range g.Data {
		hist[v]++
	}
	inv := 1.0 / float64(len(g.Data))
	for i := range hist {
		hist[i] *= inv
	}
	return hist
}

// HistogramL1Difference returns the L1 (city-block) distance between the
// normalised grayscale intensity histograms of two frames, a value in [0,2].
// Large values indicate a very different intensity distribution and hence a
// likely shot boundary. The frames need not have equal dimensions. It panics if
// either frame is empty.
func HistogramL1Difference(a, b *cv.Mat) float64 {
	if a == nil || b == nil || a.Empty() || b.Empty() {
		panic("videoproc: HistogramL1Difference requires two non-empty frames")
	}
	ha := videoprocGrayHist(a)
	hb := videoprocGrayHist(b)
	var sum float64
	for i := 0; i < 256; i++ {
		d := ha[i] - hb[i]
		if d < 0 {
			d = -d
		}
		sum += d
	}
	return sum
}

// HistogramChiSquare returns the chi-square distance between the normalised
// grayscale histograms of two frames: sum over bins of (ha-hb)²/(ha+hb), with
// empty bins skipped. It is more sensitive than the L1 distance to shifts
// concentrated in a few intensity bins. It panics if either frame is empty.
func HistogramChiSquare(a, b *cv.Mat) float64 {
	if a == nil || b == nil || a.Empty() || b.Empty() {
		panic("videoproc: HistogramChiSquare requires two non-empty frames")
	}
	ha := videoprocGrayHist(a)
	hb := videoprocGrayHist(b)
	var sum float64
	for i := 0; i < 256; i++ {
		denom := ha[i] + hb[i]
		if denom <= 0 {
			continue
		}
		d := ha[i] - hb[i]
		sum += d * d / denom
	}
	return sum
}

// PixelDifferenceRatio returns the fraction of pixels (in [0,1]) whose absolute
// grayscale change between a and b exceeds threshold. A high ratio signals that
// most of the frame changed at once, the hallmark of a hard cut. The frames must
// share dimensions. It panics on a size mismatch.
func PixelDifferenceRatio(a, b *cv.Mat, threshold uint8) float64 {
	if a == nil || b == nil || a.Empty() || b.Empty() {
		panic("videoproc: PixelDifferenceRatio requires two non-empty frames")
	}
	if a.Rows != b.Rows || a.Cols != b.Cols {
		panic("videoproc: PixelDifferenceRatio frame size mismatch")
	}
	ga := videoprocToGray(a)
	gb := videoprocToGray(b)
	changed := 0
	for i := range ga.Data {
		d := int(ga.Data[i]) - int(gb.Data[i])
		if d < 0 {
			d = -d
		}
		if d > int(threshold) {
			changed++
		}
	}
	return float64(changed) / float64(len(ga.Data))
}

// MeanIntensityDelta returns the absolute difference between the mean grayscale
// intensities of two frames. It is a cheap global cue: a large brightness jump
// often accompanies a cut or a lighting change. The frames need not have equal
// dimensions. It panics if either frame is empty.
func MeanIntensityDelta(a, b *cv.Mat) float64 {
	if a == nil || b == nil || a.Empty() || b.Empty() {
		panic("videoproc: MeanIntensityDelta requires two non-empty frames")
	}
	return absF(meanIntensity(a) - meanIntensity(b))
}

// meanIntensity returns the average grayscale intensity of a frame.
func meanIntensity(frame *cv.Mat) float64 {
	g := videoprocToGray(frame)
	var sum float64
	for _, v := range g.Data {
		sum += float64(v)
	}
	return sum / float64(len(g.Data))
}

// absF returns the absolute value of a float64.
func absF(v float64) float64 {
	if v < 0 {
		return -v
	}
	return v
}

// videoprocEdgeMask returns a binary edge mask (255 where the central-difference
// gradient magnitude exceeds threshold) for a frame.
func videoprocEdgeMask(frame *cv.Mat, threshold float64) *cv.Mat {
	g := videoprocToGray(frame)
	rows, cols := g.Rows, g.Cols
	out := cv.NewMat(rows, cols, 1)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			dx := videoprocGrayAtClamp(g, x+1, y) - videoprocGrayAtClamp(g, x-1, y)
			dy := videoprocGrayAtClamp(g, x, y+1) - videoprocGrayAtClamp(g, x, y-1)
			if dx*dx+dy*dy >= threshold*threshold {
				out.Data[y*cols+x] = 255
			}
		}
	}
	return out
}

// EdgeChangeRatio returns the edge change ratio between two frames of equal
// size: the fraction of edge pixels that appear or disappear from a to b,
// max(entering, exiting) normalised by the edge count. It is robust to global
// brightness changes because it compares structure, not intensity. edgeThreshold
// sets the gradient magnitude for edge detection. The result is in [0,1]; it is
// 0 when neither frame has edges. It panics on a size mismatch.
func EdgeChangeRatio(a, b *cv.Mat, edgeThreshold float64) float64 {
	if a == nil || b == nil || a.Empty() || b.Empty() {
		panic("videoproc: EdgeChangeRatio requires two non-empty frames")
	}
	if a.Rows != b.Rows || a.Cols != b.Cols {
		panic("videoproc: EdgeChangeRatio frame size mismatch")
	}
	ea := videoprocEdgeMask(a, edgeThreshold)
	eb := videoprocEdgeMask(b, edgeThreshold)
	var na, nb, entering, exiting int
	for i := range ea.Data {
		av := ea.Data[i] != 0
		bv := eb.Data[i] != 0
		if av {
			na++
		}
		if bv {
			nb++
		}
		if bv && !av {
			entering++
		}
		if av && !bv {
			exiting++
		}
	}
	rIn := 0.0
	if nb > 0 {
		rIn = float64(entering) / float64(nb)
	}
	rOut := 0.0
	if na > 0 {
		rOut = float64(exiting) / float64(na)
	}
	if rIn > rOut {
		return rIn
	}
	return rOut
}

// ShotBoundaryDetector is an online detector that reports a hard cut whenever the
// chosen dissimilarity between consecutive frames exceeds Threshold. It keeps the
// previous frame internally, so callers simply feed frames one at a time.
type ShotBoundaryDetector struct {
	// Threshold is the cut decision level on the metric returned by Metric.
	Threshold float64
	// Metric computes the dissimilarity of two consecutive frames. If nil,
	// [HistogramL1Difference] is used.
	Metric func(a, b *cv.Mat) float64

	prev *cv.Mat
}

// NewShotBoundaryDetector returns a detector using the given metric and decision
// threshold. If metric is nil the detector defaults to [HistogramL1Difference].
func NewShotBoundaryDetector(threshold float64, metric func(a, b *cv.Mat) float64) *ShotBoundaryDetector {
	return &ShotBoundaryDetector{Threshold: threshold, Metric: metric}
}

// Add feeds the next frame and returns true when a shot boundary is detected
// between the previous frame and this one (metric value strictly greater than
// Threshold). The first frame always returns false. The returned score is the
// raw metric value (0 for the first frame).
func (d *ShotBoundaryDetector) Add(frame *cv.Mat) (isCut bool, score float64) {
	if frame == nil || frame.Empty() {
		panic("videoproc: ShotBoundaryDetector.Add requires a non-empty frame")
	}
	metric := d.Metric
	if metric == nil {
		metric = HistogramL1Difference
	}
	if d.prev == nil {
		d.prev = frame.Clone()
		return false, 0
	}
	score = metric(d.prev, frame)
	d.prev = frame.Clone()
	return score > d.Threshold, score
}

// DetectShotBoundaries runs a [ShotBoundaryDetector] over an ordered slice of
// frames and returns the indices at which a cut begins: an index i in the result
// means the metric between frames[i-1] and frames[i] exceeded threshold, i.e. a
// new shot starts at frame i. If metric is nil [HistogramL1Difference] is used.
// The result is sorted ascending and never contains 0.
func DetectShotBoundaries(frames []*cv.Mat, threshold float64, metric func(a, b *cv.Mat) float64) []int {
	det := NewShotBoundaryDetector(threshold, metric)
	var cuts []int
	for i, f := range frames {
		isCut, _ := det.Add(f)
		if isCut {
			cuts = append(cuts, i)
		}
	}
	return cuts
}
