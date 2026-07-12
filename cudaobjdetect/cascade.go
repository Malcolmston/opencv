package cudaobjdetect

import (
	cv "github.com/malcolmston/opencv"
	"github.com/malcolmston/opencv/objdetect"
)

// CascadeClassifier is a CPU-backed analogue of cv::cuda::CascadeClassifier: a
// Viola–Jones Haar cascade loaded from an OpenCV XML file and evaluated over an
// image. It mirrors the GPU class's construction, parameters and the
// detectMultiScale/convert result-handling protocol, delegating the actual
// evaluation to [objdetect.CascadeClassifier]; the [Stream] argument is accepted
// for API compatibility and ignored.
//
// Detection follows OpenCV's two-call convention: [CascadeClassifier.DetectMultiScale]
// returns a [GpuMat] holding the packed detections and [CascadeClassifier.Convert]
// decodes it into a slice of [cv.Rect].
type CascadeClassifier struct {
	inner *objdetect.CascadeClassifier

	scaleFactor       float64
	minNeighbors      int
	minObjectSize     Size
	maxObjectSize     Size
	findLargestObject bool
}

func newCascade(inner *objdetect.CascadeClassifier) *CascadeClassifier {
	return &CascadeClassifier{
		inner:        inner,
		scaleFactor:  1.1,
		minNeighbors: 4,
	}
}

// NewCascadeClassifier loads and parses an OpenCV Haar cascade XML file from
// path, the analogue of cv::cuda::CascadeClassifier::create(filename). It
// returns an error if the file cannot be read or parsed.
func NewCascadeClassifier(filename string) (*CascadeClassifier, error) {
	inner := &objdetect.CascadeClassifier{}
	if err := inner.Load(filename); err != nil {
		return nil, err
	}
	return newCascade(inner), nil
}

// LoadCascadeFromString parses an OpenCV Haar cascade from an in-memory XML
// string, the analogue of loading a cascade from a cv::FileStorage. It returns
// an error if the XML cannot be parsed into a supported cascade.
func LoadCascadeFromString(xmlData string) (*CascadeClassifier, error) {
	inner := &objdetect.CascadeClassifier{}
	if err := inner.LoadFromString(xmlData); err != nil {
		return nil, err
	}
	return newCascade(inner), nil
}

// GetClassifierSize returns the base detection window (width, height) declared
// by the loaded cascade, the analogue of cv::cuda::CascadeClassifier::getClassifierSize.
func (c *CascadeClassifier) GetClassifierSize() (w, h int) { return c.inner.WindowSize() }

// DetectMultiScale scans img and returns a [GpuMat] carrying the detected object
// rectangles, the analogue of cv::cuda::CascadeClassifier::detectMultiScale. The
// returned GpuMat holds no pixels; pass it to [CascadeClassifier.Convert] to
// obtain the rectangles. The classifier window grows by ScaleFactor between
// levels, clusters with fewer than MinNeighbors members are discarded, and
// detections outside the [MinObjectSize, MaxObjectSize] range are filtered out.
// When FindLargestObject is set, only the single largest surviving detection is
// returned. The stream is ignored. It panics if img holds no image data.
func (c *CascadeClassifier) DetectMultiScale(img *GpuMat, stream *Stream) *GpuMat {
	m := mustImage(img)
	c.inner.ScaleFactor = c.scaleFactor
	c.inner.MinNeighbors = c.minNeighbors
	rects := c.inner.DetectMultiScale(m)
	rects = c.filterBySize(rects)
	if c.findLargestObject {
		rects = keepLargest(rects)
	}
	return &GpuMat{objects: rects}
}

// Convert decodes the packed detection [GpuMat] produced by
// [CascadeClassifier.DetectMultiScale] into a slice of rectangles, the analogue
// of cv::cuda::CascadeClassifier::convert. It returns a copy, so the result is
// safe to retain and mutate. A GpuMat that carries no detections yields an empty
// slice.
func (c *CascadeClassifier) Convert(gpuObjects *GpuMat) []cv.Rect {
	if gpuObjects == nil || len(gpuObjects.objects) == 0 {
		return nil
	}
	out := make([]cv.Rect, len(gpuObjects.objects))
	copy(out, gpuObjects.objects)
	return out
}

// filterBySize drops rectangles outside the configured min/max object size. A
// zero (unset) bound imposes no limit on that side.
func (c *CascadeClassifier) filterBySize(rects []cv.Rect) []cv.Rect {
	minW, minH := c.minObjectSize.Width, c.minObjectSize.Height
	maxW, maxH := c.maxObjectSize.Width, c.maxObjectSize.Height
	if minW <= 0 && minH <= 0 && maxW <= 0 && maxH <= 0 {
		return rects
	}
	var out []cv.Rect
	for _, r := range rects {
		if minW > 0 && r.Width < minW {
			continue
		}
		if minH > 0 && r.Height < minH {
			continue
		}
		if maxW > 0 && r.Width > maxW {
			continue
		}
		if maxH > 0 && r.Height > maxH {
			continue
		}
		out = append(out, r)
	}
	return out
}

// keepLargest returns at most the single rectangle of greatest area.
func keepLargest(rects []cv.Rect) []cv.Rect {
	if len(rects) <= 1 {
		return rects
	}
	best := 0
	bestArea := rects[0].Width * rects[0].Height
	for i := 1; i < len(rects); i++ {
		if a := rects[i].Width * rects[i].Height; a > bestArea {
			best, bestArea = i, a
		}
	}
	return []cv.Rect{rects[best]}
}

// --- parameter accessors ------------------------------------------------------

// SetScaleFactor sets the ratio between successive window sizes; values <= 1 are
// treated as the default 1.1.
func (c *CascadeClassifier) SetScaleFactor(s float64) { c.scaleFactor = s }

// GetScaleFactor returns the configured scale factor.
func (c *CascadeClassifier) GetScaleFactor() float64 { return c.scaleFactor }

// SetMinNeighbors sets the minimum number of overlapping detections a grouped
// result must have; 0 disables grouping and returns every raw hit.
func (c *CascadeClassifier) SetMinNeighbors(n int) { c.minNeighbors = n }

// GetMinNeighbors returns the configured minimum neighbour count.
func (c *CascadeClassifier) GetMinNeighbors() int { return c.minNeighbors }

// SetMinObjectSize sets the smallest object to detect; detections smaller in
// either dimension are discarded. A zero bound disables the lower limit.
func (c *CascadeClassifier) SetMinObjectSize(s Size) { c.minObjectSize = s }

// GetMinObjectSize returns the configured minimum object size.
func (c *CascadeClassifier) GetMinObjectSize() Size { return c.minObjectSize }

// SetMaxObjectSize sets the largest object to detect; detections larger in
// either dimension are discarded. A zero bound disables the upper limit.
func (c *CascadeClassifier) SetMaxObjectSize(s Size) { c.maxObjectSize = s }

// GetMaxObjectSize returns the configured maximum object size.
func (c *CascadeClassifier) GetMaxObjectSize() Size { return c.maxObjectSize }

// SetFindLargestObject toggles returning only the single largest detection.
func (c *CascadeClassifier) SetFindLargestObject(on bool) { c.findLargestObject = on }

// GetFindLargestObject reports whether only the largest detection is returned.
func (c *CascadeClassifier) GetFindLargestObject() bool { return c.findLargestObject }
