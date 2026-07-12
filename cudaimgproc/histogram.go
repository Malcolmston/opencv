package cudaimgproc

import (
	"fmt"

	cv "github.com/malcolmston/opencv"
)

// CalcHist computes a 256-bin intensity histogram of a single-channel GpuMat,
// mirroring cuda::calcHist. Entry i counts pixels whose value equals i. The
// trailing Stream argument is accepted and ignored. It panics unless src is
// single-channel.
func CalcHist(src GpuMat, streams ...Stream) []int {
	_ = firstStream(streams)
	m := src.requireHost("CalcHist")
	if m.Channels != 1 {
		panic(fmt.Sprintf("cudaimgproc: CalcHist requires a single-channel source, got %d", m.Channels))
	}
	return cv.CalcHist(m, 0)
}

// EqualizeHist performs global histogram equalisation on a single-channel
// GpuMat, mirroring cuda::equalizeHist, and returns the equalised image. The
// trailing Stream argument is accepted and ignored. It panics unless src is
// single-channel.
func EqualizeHist(src GpuMat, streams ...Stream) GpuMat {
	_ = firstStream(streams)
	m := src.requireHost("EqualizeHist")
	return wrap(cv.EqualizeHist(m))
}

// CalcBackProject back-projects a 256-bin histogram onto a single-channel
// GpuMat, mirroring cuda::calcBackProject: each output pixel is the histogram
// count for that pixel's intensity, rescaled so the largest bin maps to 255.
// The trailing Stream argument is accepted and ignored. It panics unless src is
// single-channel or hist is not 256 bins.
func CalcBackProject(src GpuMat, hist []int, streams ...Stream) GpuMat {
	_ = firstStream(streams)
	m := src.requireHost("CalcBackProject")
	if m.Channels != 1 {
		panic(fmt.Sprintf("cudaimgproc: CalcBackProject requires a single-channel source, got %d", m.Channels))
	}
	return wrap(cv.CalcBackProject(m, 0, hist))
}

// HistEven computes a histogram with histSize equally spaced bins spanning the
// half-open range [lowerLevel, upperLevel), mirroring cuda::histEven. Values
// below lowerLevel or at/above upperLevel are ignored. The source must be
// single-channel. The trailing Stream argument is accepted and ignored. It
// panics on a non single-channel source, histSize < 1, or lowerLevel >=
// upperLevel.
func HistEven(src GpuMat, histSize, lowerLevel, upperLevel int, streams ...Stream) []int {
	_ = firstStream(streams)
	m := src.requireHost("HistEven")
	if m.Channels != 1 {
		panic(fmt.Sprintf("cudaimgproc: HistEven requires a single-channel source, got %d", m.Channels))
	}
	if histSize < 1 {
		panic("cudaimgproc: HistEven requires histSize >= 1")
	}
	if lowerLevel >= upperLevel {
		panic("cudaimgproc: HistEven requires lowerLevel < upperLevel")
	}
	hist := make([]int, histSize)
	span := float64(upperLevel - lowerLevel)
	for _, v := range m.Data {
		iv := int(v)
		if iv < lowerLevel || iv >= upperLevel {
			continue
		}
		bin := int(float64(iv-lowerLevel) / span * float64(histSize))
		if bin >= histSize {
			bin = histSize - 1
		}
		hist[bin]++
	}
	return hist
}

// HistRange computes a histogram whose bins are delimited by the ascending
// boundaries in levels, mirroring cuda::histRange. For n boundaries there are
// n-1 bins: bin i counts values in the half-open interval
// [levels[i], levels[i+1]). Values outside [levels[0], levels[n-1]) are
// ignored. The source must be single-channel. The trailing Stream argument is
// accepted and ignored. It panics on a non single-channel source, fewer than
// two levels, or non-ascending levels.
func HistRange(src GpuMat, levels []int, streams ...Stream) []int {
	_ = firstStream(streams)
	m := src.requireHost("HistRange")
	if m.Channels != 1 {
		panic(fmt.Sprintf("cudaimgproc: HistRange requires a single-channel source, got %d", m.Channels))
	}
	if len(levels) < 2 {
		panic("cudaimgproc: HistRange requires at least two levels")
	}
	for i := 1; i < len(levels); i++ {
		if levels[i] <= levels[i-1] {
			panic("cudaimgproc: HistRange requires strictly ascending levels")
		}
	}
	hist := make([]int, len(levels)-1)
	for _, v := range m.Data {
		iv := int(v)
		if iv < levels[0] || iv >= levels[len(levels)-1] {
			continue
		}
		// Linear scan is fine for the small level counts typical here.
		for b := 0; b < len(levels)-1; b++ {
			if iv >= levels[b] && iv < levels[b+1] {
				hist[b]++
				break
			}
		}
	}
	return hist
}

// CLAHE is a CPU-backed Contrast-Limited Adaptive Histogram Equalisation
// algorithm object, mirroring cv::cuda::CLAHE. Create one with [CreateCLAHE]
// and run it with [CLAHE.Apply]. The clip limit and tile grid can be adjusted
// between applications.
type CLAHE struct {
	clipLimit    float64
	tileGridSize int
}

// CreateCLAHE returns a [CLAHE] with the given clip limit and a square
// tileGridSize×tileGridSize tiling, mirroring cuda::createCLAHE. A clipLimit <=
// 0 disables clipping (plain adaptive equalisation). It panics if tileGridSize
// < 1.
func CreateCLAHE(clipLimit float64, tileGridSize int) *CLAHE {
	if tileGridSize < 1 {
		panic("cudaimgproc: CreateCLAHE requires tileGridSize >= 1")
	}
	return &CLAHE{clipLimit: clipLimit, tileGridSize: tileGridSize}
}

// SetClipLimit updates the contrast clip limit used by subsequent Apply calls.
func (c *CLAHE) SetClipLimit(clipLimit float64) { c.clipLimit = clipLimit }

// GetClipLimit returns the current clip limit.
func (c *CLAHE) GetClipLimit() float64 { return c.clipLimit }

// SetTilesGridSize updates the square tile-grid size used by subsequent Apply
// calls. It panics if size < 1.
func (c *CLAHE) SetTilesGridSize(size int) {
	if size < 1 {
		panic("cudaimgproc: SetTilesGridSize requires size >= 1")
	}
	c.tileGridSize = size
}

// GetTilesGridSize returns the current tile-grid size.
func (c *CLAHE) GetTilesGridSize() int { return c.tileGridSize }

// Apply runs CLAHE on a single-channel GpuMat and returns the enhanced image.
// The trailing Stream argument is accepted and ignored. It panics unless src is
// single-channel.
func (c *CLAHE) Apply(src GpuMat, streams ...Stream) GpuMat {
	_ = firstStream(streams)
	m := src.requireHost("CLAHE.Apply")
	return wrap(cv.CLAHE(m, c.clipLimit, c.tileGridSize))
}
