package saliency2_test

import (
	"testing"

	cv "github.com/malcolmston/opencv"
	"github.com/malcolmston/opencv/saliency2"
)

// Compile-time checks that the detectors satisfy the package interfaces.
var (
	_ saliency2.StaticSaliency = (*saliency2.StaticSaliencySpectralResidual)(nil)
	_ saliency2.StaticSaliency = (*saliency2.StaticSaliencyFineGrained)(nil)
	_ saliency2.StaticSaliency = (*saliency2.StaticSaliencyFrequencyTuned)(nil)
	_ saliency2.StaticSaliency = (*saliency2.StaticSaliencyIttiKoch)(nil)
	_ saliency2.StaticSaliency = (*saliency2.ObjectnessBING)(nil)
	_ saliency2.MotionSaliency = (*saliency2.MotionSaliencyByDifference)(nil)
	_ saliency2.MotionSaliency = (*saliency2.MotionSaliencyRunningAverage)(nil)
	_ saliency2.Objectness     = (*saliency2.ObjectnessBING)(nil)
)

// diskImage builds a size x size single-channel image with a filled disk of
// value fg at (cx, cy) radius r over a background of value bg.
func diskImage(size, cx, cy, r int, bg, fg uint8) *cv.Mat {
	m := cv.NewMat(size, size, 1)
	m.SetTo(bg)
	for y := 0; y < size; y++ {
		for x := 0; x < size; x++ {
			dx := x - cx
			dy := y - cy
			if dx*dx+dy*dy <= r*r {
				m.Set(y, x, 0, fg)
			}
		}
	}
	return m
}

// squareImage builds a size x size single-channel image with a filled square of
// value fg with top-left (x0, y0) and the given side over background bg.
func squareImage(size, x0, y0, side int, bg, fg uint8) *cv.Mat {
	m := cv.NewMat(size, size, 1)
	m.SetTo(bg)
	for y := y0; y < y0+side; y++ {
		for x := x0; x < x0+side; x++ {
			m.Set(y, x, 0, fg)
		}
	}
	return m
}

// meanRegion returns the mean saliency of the map over the square window.
func meanRegion(m *saliency2.SaliencyMap, x0, y0, side int) float64 {
	var s float64
	var n int
	for y := y0; y < y0+side; y++ {
		for x := x0; x < x0+side; x++ {
			if y >= 0 && y < m.Rows && x >= 0 && x < m.Cols {
				s += m.At(y, x)
				n++
			}
		}
	}
	if n == 0 {
		return 0
	}
	return s / float64(n)
}

// staticDetectors returns the object-popping static detectors keyed by name.
func staticMapDetectors() map[string]func(*cv.Mat) *saliency2.SaliencyMap {
	return map[string]func(*cv.Mat) *saliency2.SaliencyMap{
		"spectral":  saliency2.NewStaticSaliencySpectralResidual().ComputeSaliencyMap,
		"finegrain": saliency2.NewStaticSaliencyFineGrained().ComputeSaliencyMap,
		"freqtuned": saliency2.NewStaticSaliencyFrequencyTuned().ComputeSaliencyMap,
		"itti":      saliency2.NewStaticSaliencyIttiKoch().ComputeSaliencyMap,
	}
}

// TestStaticDetectorsHighlightObject checks that every static detector assigns
// more saliency to a bright object than to the far background corners.
func TestStaticDetectorsHighlightObject(t *testing.T) {
	const size = 64
	const cx, cy, r = 32, 32, 12
	img := diskImage(size, cx, cy, r, 40, 220)

	for name, compute := range staticMapDetectors() {
		sal := compute(img)
		if sal.Rows != size || sal.Cols != size {
			t.Errorf("%s: map size %dx%d, want %dx%d", name, sal.Rows, sal.Cols, size, size)
			continue
		}
		object := meanRegion(sal, cx-r/2, cy-r/2, r)
		corner := meanRegion(sal, 0, 0, 8)
		if !(object > corner) {
			t.Errorf("%s: object mean %.4f not greater than corner mean %.4f", name, object, corner)
		}
	}
}

// TestStaticDetectorsOutputRange checks the 8-bit map is produced, single
// channel, correct size, and uses its dynamic range.
func TestStaticDetectorsOutputRange(t *testing.T) {
	const size = 48
	img := diskImage(size, 24, 24, 9, 30, 210)
	detectors := map[string]saliency2.StaticSaliency{
		"spectral":  saliency2.NewStaticSaliencySpectralResidual(),
		"finegrain": saliency2.NewStaticSaliencyFineGrained(),
		"freqtuned": saliency2.NewStaticSaliencyFrequencyTuned(),
		"itti":      saliency2.NewStaticSaliencyIttiKoch(),
	}
	for name, d := range detectors {
		out := d.ComputeSaliency(img)
		if out.Channels != 1 || out.Rows != size || out.Cols != size {
			t.Errorf("%s: got %dx%dx%d", name, out.Rows, out.Cols, out.Channels)
		}
		var maxV uint8
		for _, v := range out.Data {
			if v > maxV {
				maxV = v
			}
		}
		if maxV != 255 {
			t.Errorf("%s: normalised max = %d, want 255", name, maxV)
		}
	}
}

// TestConvenienceWrappers exercises the free-function detector wrappers.
func TestConvenienceWrappers(t *testing.T) {
	img := diskImage(48, 24, 24, 9, 30, 210)
	wrappers := map[string]func(*cv.Mat) *cv.Mat{
		"spectral":  saliency2.SpectralResidualSaliency,
		"finegrain": saliency2.FineGrainedSaliency,
		"freqtuned": saliency2.FrequencyTunedSaliency,
		"itti":      saliency2.IttiKochSaliency,
	}
	for name, fn := range wrappers {
		out := fn(img)
		if out.Empty() || out.Channels != 1 {
			t.Errorf("%s wrapper produced bad output", name)
		}
	}
}

// TestColorFrequencyTuned checks the frequency-tuned detector on a genuinely
// coloured scene: a red object on a grey field.
func TestColorFrequencyTuned(t *testing.T) {
	const size = 48
	img := cv.NewMat(size, size, 3)
	for i := 0; i < size*size; i++ {
		img.Data[i*3+0] = 128
		img.Data[i*3+1] = 128
		img.Data[i*3+2] = 128
	}
	for y := 18; y < 30; y++ {
		for x := 18; x < 30; x++ {
			img.Set(y, x, 0, 230)
			img.Set(y, x, 1, 20)
			img.Set(y, x, 2, 20)
		}
	}
	sal := saliency2.NewStaticSaliencyFrequencyTuned().ComputeSaliencyMap(img)
	object := meanRegion(sal, 20, 20, 8)
	corner := meanRegion(sal, 0, 0, 8)
	if !(object > corner) {
		t.Fatalf("colour object mean %.4f not greater than corner %.4f", object, corner)
	}
}
