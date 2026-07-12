package cudaobjdetect

import (
	cv "github.com/malcolmston/opencv"
	"github.com/malcolmston/opencv/objdetect"
)

// Size is a width/height pair in pixels used to configure detector geometry. It
// is the same type as [objdetect.Size], re-exported so callers of this package
// need not import objdetect directly.
type Size = objdetect.Size

// HOG is a CPU-backed analogue of cv::cuda::HOG: a Histogram of Oriented
// Gradients descriptor with a linear-SVM sliding-window detector (Dalal &
// Triggs, 2005). It mirrors the GPU class's construction, parameters and
// detection methods but performs all work on the host by delegating to
// [objdetect.HOGDescriptor]; the [Stream] arguments are accepted for API
// compatibility and ignored.
//
// Construct one with [NewHOG] (custom geometry) or [NewDefaultHOG] (the
// canonical 64×128 person detector). Provide a classifier with
// [HOG.SetSVMDetector] — commonly [HOG.GetDefaultPeopleDetector] — before
// calling [HOG.DetectMultiScale] or [HOG.Detect].
type HOG struct {
	desc *objdetect.HOGDescriptor
	svm  []float64

	hitThreshold    float64
	winStride       Size
	scaleFactor     float64
	numLevels       int
	groupThreshold  int
	gammaCorrection bool
	l2HysThreshold  float64
	winSigma        float64
}

// NewHOG creates a HOG detector with the given geometry, mirroring
// cv::cuda::HOG::create(win_size, block_size, block_stride, cell_size, nbins).
// The five values must be mutually compatible (BlockSize and BlockStride whole
// multiples of CellSize, and WinSize-BlockSize a whole multiple of BlockStride);
// it panics otherwise. The detector starts with OpenCV-like default parameters
// (scale factor 1.05, group threshold 2, gamma correction on, L2-Hys threshold
// 0.2) and no SVM detector set.
func NewHOG(winSize, blockSize, blockStride, cellSize Size, nbins int) *HOG {
	d := &objdetect.HOGDescriptor{
		WinSize:     winSize,
		BlockSize:   blockSize,
		BlockStride: blockStride,
		CellSize:    cellSize,
		NBins:       nbins,
	}
	// DescriptorSize validates the geometry and panics on inconsistency.
	_ = d.DescriptorSize()
	return &HOG{
		desc:            d,
		hitThreshold:    0,
		winStride:       blockStride,
		scaleFactor:     1.05,
		numLevels:       64,
		groupThreshold:  2,
		gammaCorrection: true,
		l2HysThreshold:  0.2,
		winSigma:        -1,
	}
}

// NewDefaultHOG returns a HOG configured with the canonical Dalal–Triggs
// geometry (64×128 window, 16×16 blocks stepping by 8×8, 8×8 cells, 9 bins),
// the same default as cv::cuda::HOG::create() with no arguments. Its descriptor
// size is 3780.
func NewDefaultHOG() *HOG {
	return NewHOG(Size{Width: 64, Height: 128}, Size{Width: 16, Height: 16},
		Size{Width: 8, Height: 8}, Size{Width: 8, Height: 8}, 9)
}

// WinSize returns the detection window in pixels.
func (h *HOG) WinSize() Size { return h.desc.WinSize }

// GetDescriptorSize returns the length of the descriptor vector produced for a
// single window, the analogue of cv::cuda::HOG::getDescriptorSize.
func (h *HOG) GetDescriptorSize() int { return h.desc.DescriptorSize() }

// GetBlockHistogramSize returns the number of values in a single block's
// histogram (cells-per-block × nbins), the analogue of
// cv::cuda::HOG::getBlockHistogramSize.
func (h *HOG) GetBlockHistogramSize() int {
	cpbX := h.desc.BlockSize.Width / h.desc.CellSize.Width
	cpbY := h.desc.BlockSize.Height / h.desc.CellSize.Height
	return cpbX * cpbY * h.desc.NBins
}

// SetSVMDetector sets the linear-SVM coefficients used by [HOG.Detect] and
// [HOG.DetectMultiScale], the analogue of cv::cuda::HOG::setSVMDetector. The
// slice length must equal [HOG.GetDescriptorSize] (no bias) or that plus one
// (the final element is an additive bias); it panics otherwise. The detector
// keeps an independent copy.
func (h *HOG) SetSVMDetector(detector []float64) {
	descLen := h.desc.DescriptorSize()
	if len(detector) != descLen && len(detector) != descLen+1 {
		panic("cudaobjdetect: SetSVMDetector length mismatch")
	}
	h.svm = make([]float64, len(detector))
	copy(h.svm, detector)
}

// GetSVMDetector returns a copy of the currently set SVM coefficients, or nil if
// none has been set.
func (h *HOG) GetSVMDetector() []float64 {
	if h.svm == nil {
		return nil
	}
	out := make([]float64, len(h.svm))
	copy(out, h.svm)
	return out
}

// GetDefaultPeopleDetector returns an approximate linear-SVM weight vector for
// upright-person detection sized to this detector's geometry, the analogue of
// cv::cuda::HOG::getDefaultPeopleDetector. Because OpenCV's exact 3781 INRIA
// coefficients cannot be reproduced without the training data, this synthesises
// a matched-filter classifier from a prototype pedestrian silhouette (see
// [objdetect.HOGDescriptor.DefaultPeopleDetector]). It is directly usable with
// [HOG.SetSVMDetector].
func (h *HOG) GetDefaultPeopleDetector() []float64 {
	return h.desc.DefaultPeopleDetector()
}

// Compute returns the HOG descriptor for a single window taken from the
// top-left corner of img, the analogue of cv::cuda::HOG::compute. OpenCV writes
// the result into a CV_32F GpuMat; because the root [cv.Mat] is 8-bit, the
// full-precision descriptor is returned as a []float64 instead. The stream is
// ignored. It panics if img is too small for the window.
func (h *HOG) Compute(img *GpuMat, stream *Stream) []float64 {
	return h.desc.Compute(mustImage(img))
}

// Detect runs the SVM detector at a single scale and returns the top-left
// corner of every window whose score meets the hit threshold, together with the
// scores, the analogue of cv::cuda::HOG::detect. An SVM detector must have been
// set with [HOG.SetSVMDetector]; it panics otherwise. Windows step by
// BlockStride. The stream is ignored.
func (h *HOG) Detect(img *GpuMat, stream *Stream) (locations []cv.Point, confidences []float64) {
	h.requireSVM()
	m := mustImage(img)
	// A scale factor larger than the image guarantees the pyramid stops after
	// the base level, giving a single-scale scan.
	singleScale := float64(m.Rows+m.Cols) + 2
	rects, scores := h.desc.DetectMultiScaleWeights(m, h.svm, h.hitThreshold, singleScale)
	locations = make([]cv.Point, len(rects))
	for i, r := range rects {
		locations[i] = cv.Point{X: r.X, Y: r.Y}
	}
	return locations, scores
}

// DetectMultiScale slides the detector over a downscaling image pyramid and
// returns the detected object rectangles together with their scores, the
// analogue of cv::cuda::HOG::detectMultiScale. An SVM detector must have been
// set; it panics otherwise. Successive pyramid levels shrink by ScaleFactor
// (see [HOG.SetScaleFactor]); when GroupThreshold is positive, overlapping raw
// hits are clustered with [objdetect.GroupRectanglesWeights] and each surviving
// cluster reports its strongest score. The stream is ignored.
//
// Note that this delegates to the host detector: WinStride is advisory (windows
// step by BlockStride) and NumLevels is not enforced (the pyramid runs until the
// image is smaller than the window).
func (h *HOG) DetectMultiScale(img *GpuMat, stream *Stream) (locations []cv.Rect, confidences []float64) {
	h.requireSVM()
	rects, scores := h.desc.DetectMultiScaleWeights(mustImage(img), h.svm, h.hitThreshold, h.scaleFactor)
	if h.groupThreshold > 0 {
		return objdetect.GroupRectanglesWeights(rects, scores, h.groupThreshold, 0.2)
	}
	return rects, scores
}

func (h *HOG) requireSVM() {
	if h.svm == nil {
		panic("cudaobjdetect: no SVM detector set (call SetSVMDetector)")
	}
}

// --- parameter accessors ------------------------------------------------------

// SetHitThreshold sets the minimum SVM score a window must reach to be reported.
func (h *HOG) SetHitThreshold(t float64) { h.hitThreshold = t }

// GetHitThreshold returns the current hit threshold.
func (h *HOG) GetHitThreshold() float64 { return h.hitThreshold }

// SetWinStride sets the window step (advisory; detection steps by BlockStride).
func (h *HOG) SetWinStride(s Size) { h.winStride = s }

// GetWinStride returns the configured window stride.
func (h *HOG) GetWinStride() Size { return h.winStride }

// SetScaleFactor sets the pyramid ratio between successive levels; it must be
// greater than 1 and panics otherwise.
func (h *HOG) SetScaleFactor(s float64) {
	if s <= 1 {
		panic("cudaobjdetect: scale factor must be > 1")
	}
	h.scaleFactor = s
}

// GetScaleFactor returns the pyramid scale factor.
func (h *HOG) GetScaleFactor() float64 { return h.scaleFactor }

// SetNumLevels sets the maximum number of pyramid levels (advisory).
func (h *HOG) SetNumLevels(n int) { h.numLevels = n }

// GetNumLevels returns the configured maximum number of pyramid levels.
func (h *HOG) GetNumLevels() int { return h.numLevels }

// SetGroupThreshold sets the minimum cluster size kept when grouping overlapping
// detections; 0 disables grouping.
func (h *HOG) SetGroupThreshold(n int) { h.groupThreshold = n }

// GetGroupThreshold returns the grouping threshold.
func (h *HOG) GetGroupThreshold() int { return h.groupThreshold }

// SetGammaCorrection toggles input gamma correction (stored for compatibility).
func (h *HOG) SetGammaCorrection(on bool) { h.gammaCorrection = on }

// GetGammaCorrection reports whether gamma correction is enabled.
func (h *HOG) GetGammaCorrection() bool { return h.gammaCorrection }

// SetL2HysThreshold sets the L2-Hys clipping threshold (stored for compatibility).
func (h *HOG) SetL2HysThreshold(t float64) { h.l2HysThreshold = t }

// GetL2HysThreshold returns the L2-Hys clipping threshold.
func (h *HOG) GetL2HysThreshold() float64 { return h.l2HysThreshold }

// SetWinSigma sets the Gaussian window weighting sigma (stored for compatibility;
// a negative value means "auto").
func (h *HOG) SetWinSigma(s float64) { h.winSigma = s }

// GetWinSigma returns the configured window sigma.
func (h *HOG) GetWinSigma() float64 { return h.winSigma }

// mustImage returns the host image of a GpuMat, panicking if it holds none.
func mustImage(img *GpuMat) *cv.Mat {
	if img == nil || img.mat == nil {
		panic("cudaobjdetect: GpuMat has no image data")
	}
	return img.mat
}
