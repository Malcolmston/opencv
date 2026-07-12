package face

import (
	cv "github.com/malcolmston/opencv"
)

// This file provides GetFacesHAAR, a self-contained Haar-feature face detector
// built on an integral image. OpenCV's cv::face::getFacesHAAR runs a trained
// cascade loaded from XML; with no external model files this port instead uses a
// small, fixed set of Haar-like features that respond to the coarse light/dark
// structure common to upright faces (a darker eye band above a brighter
// mid-face, and a darker vertical eye/nose axis against brighter cheeks). It is
// a genuine sliding-window, multi-scale detector with integral-image O(1)
// feature evaluation and greedy non-maximum suppression — useful for locating
// face-like regions in synthetic or high-contrast imagery without a trained
// cascade.

// HaarParams tunes the [GetFacesHAAR] sliding-window search.
type HaarParams struct {
	// MinSize is the smallest square window side, in pixels, that is scanned.
	MinSize int
	// MaxSize caps the largest window side; 0 means the smaller image side.
	MaxSize int
	// ScaleStep multiplies the window side between scales (must be > 1);
	// 0 selects the default 1.25.
	ScaleStep float64
	// StepRatio sets the sliding stride as a fraction of the window side
	// (clamped to at least one pixel); 0 selects the default 0.1.
	StepRatio float64
	// MinScore is the minimum combined feature response, in mean grey levels,
	// for a window to be accepted; 0 selects the default 18.
	MinScore float64
}

// DefaultHaarParams returns the parameters [GetFacesHAAR] uses when passed a nil
// configuration: a scan from 24-pixel windows up to the image size, a 1.25
// scale step, a 10%-of-window stride and a moderate acceptance score.
func DefaultHaarParams() HaarParams {
	return HaarParams{MinSize: 24, MaxSize: 0, ScaleStep: 1.25, StepRatio: 0.1, MinScore: 18}
}

// GetFacesHAAR detects upright, face-like regions in img and returns their
// bounding rectangles. img is reduced to luma and summarised by an integral
// image so each window's Haar features cost O(1) regardless of size. Windows are
// scanned over a range of scales and positions; those whose combined feature
// response exceeds the acceptance score are kept and then reduced by greedy
// non-maximum suppression so each face yields a single box. Pass params nil to
// use [DefaultHaarParams]. The returned rectangles are ordered by descending
// score. It panics on a nil or empty image.
func GetFacesHAAR(img *cv.Mat, params *HaarParams) []cv.Rect {
	p := DefaultHaarParams()
	if params != nil {
		p = *params
	}
	if p.ScaleStep <= 1 {
		p.ScaleStep = 1.25
	}
	if p.StepRatio <= 0 {
		p.StepRatio = 0.1
	}
	if p.MinSize < 8 {
		p.MinSize = 8
	}
	if p.MinScore <= 0 {
		p.MinScore = 18
	}

	g := toGrayMat(img)
	rows, cols := g.Rows, g.Cols
	shorter := rows
	if cols < shorter {
		shorter = cols
	}
	maxSize := p.MaxSize
	if maxSize <= 0 || maxSize > shorter {
		maxSize = shorter
	}
	if p.MinSize > maxSize {
		return nil
	}

	ii := newIntegral(g)

	type det struct {
		rect  cv.Rect
		score float64
	}
	var dets []det

	for win := p.MinSize; win <= maxSize; {
		stride := int(float64(win)*p.StepRatio + 0.5)
		if stride < 1 {
			stride = 1
		}
		for y := 0; y+win <= rows; y += stride {
			for x := 0; x+win <= cols; x += stride {
				s := haarScore(ii, x, y, win)
				if s >= p.MinScore {
					dets = append(dets, det{rect: cv.Rect{X: x, Y: y, Width: win, Height: win}, score: s})
				}
			}
		}
		next := int(float64(win)*p.ScaleStep + 0.5)
		if next <= win {
			next = win + 1
		}
		win = next
	}

	// Sort by descending score (selection sort keeps the file dependency-free
	// of sort and is fine for the modest candidate counts produced here).
	for i := 0; i < len(dets); i++ {
		best := i
		for j := i + 1; j < len(dets); j++ {
			if dets[j].score > dets[best].score {
				best = j
			}
		}
		dets[i], dets[best] = dets[best], dets[i]
	}

	// Greedy non-maximum suppression at 30% IoU.
	var kept []cv.Rect
	suppressed := make([]bool, len(dets))
	for i := range dets {
		if suppressed[i] {
			continue
		}
		kept = append(kept, dets[i].rect)
		for j := i + 1; j < len(dets); j++ {
			if !suppressed[j] && rectIoU(dets[i].rect, dets[j].rect) > 0.3 {
				suppressed[j] = true
			}
		}
	}
	return kept
}

// integral is a summed-area table of a single-channel image with a zero-padded
// first row and column, so the sum over any rectangle costs four look-ups.
type integral struct {
	rows, cols int
	sum        []int64 // (rows+1)*(cols+1)
}

func newIntegral(g *cv.Mat) *integral {
	rows, cols := g.Rows, g.Cols
	w := cols + 1
	sum := make([]int64, (rows+1)*w)
	for y := 0; y < rows; y++ {
		var rowAcc int64
		for x := 0; x < cols; x++ {
			rowAcc += int64(g.Data[y*cols+x])
			sum[(y+1)*w+(x+1)] = sum[y*w+(x+1)] + rowAcc
		}
	}
	return &integral{rows: rows, cols: cols, sum: sum}
}

// rectSum returns the sum of pixels in the half-open rectangle [x,x+w)×[y,y+h).
func (in *integral) rectSum(x, y, w, h int) int64 {
	stride := in.cols + 1
	a := in.sum[y*stride+x]
	b := in.sum[y*stride+(x+w)]
	c := in.sum[(y+h)*stride+x]
	d := in.sum[(y+h)*stride+(x+w)]
	return d - b - c + a
}

// rectMean returns the mean grey level of a rectangle.
func (in *integral) rectMean(x, y, w, h int) float64 {
	if w <= 0 || h <= 0 {
		return 0
	}
	return float64(in.rectSum(x, y, w, h)) / float64(w*h)
}

// haarScore evaluates the fixed face template over a win×win window whose
// top-left corner is (x,y). It rewards two arrangements typical of an upright
// face: the upper (eye) band being darker than the mid-face band below it, and a
// darker central vertical strip (eyes/nose shadow) flanked by brighter cheeks.
// The response is expressed in mean grey levels so [HaarParams.MinScore] is
// interpretable.
func haarScore(in *integral, x, y, win int) float64 {
	// Two-rectangle vertical contrast: dark eye band over bright cheek band.
	bandH := win / 4
	if bandH < 1 {
		bandH = 1
	}
	eyeTop := y + win/6
	cheekTop := eyeTop + bandH
	if cheekTop+bandH > y+win {
		return 0
	}
	eye := in.rectMean(x, eyeTop, win, bandH)
	cheek := in.rectMean(x, cheekTop, win, bandH)
	vertical := cheek - eye // positive when the eye band is darker

	// Three-rectangle horizontal contrast around the vertical mid-line: a dark
	// centre column between two brighter side columns.
	third := win / 3
	if third < 1 {
		third = 1
	}
	colTop := y + win/6
	colH := win / 2
	if colH < 1 {
		colH = 1
	}
	if colTop+colH > y+win {
		colH = y + win - colTop
	}
	left := in.rectMean(x, colTop, third, colH)
	center := in.rectMean(x+third, colTop, third, colH)
	right := in.rectMean(x+2*third, colTop, third, colH)
	horizontal := (left+right)/2 - center // positive when the centre is darker

	return vertical + horizontal
}

// rectIoU returns the intersection-over-union of two rectangles.
func rectIoU(a, b cv.Rect) float64 {
	x0 := maxInt(a.X, b.X)
	y0 := maxInt(a.Y, b.Y)
	x1 := minInt(a.X+a.Width, b.X+b.Width)
	y1 := minInt(a.Y+a.Height, b.Y+b.Height)
	iw := x1 - x0
	ih := y1 - y0
	if iw <= 0 || ih <= 0 {
		return 0
	}
	inter := float64(iw * ih)
	union := float64(a.Width*a.Height+b.Width*b.Height) - inter
	if union <= 0 {
		return 0
	}
	return inter / union
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}
