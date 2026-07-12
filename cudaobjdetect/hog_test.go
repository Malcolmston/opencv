package cudaobjdetect

import (
	"math"
	"testing"

	cv "github.com/malcolmston/opencv"
)

// plantedPerson renders the same dark-figure-on-light-ground silhouette that
// HOG.GetDefaultPeopleDetector is built from, so the detector's SVM correlates
// maximally with it. This makes the people-detector sign check deterministic.
func plantedPerson(w, ht int) *cv.Mat {
	m := cv.NewMat(ht, w, 1)
	const bg, fg = 210, 60
	cx := float64(w) / 2
	headR := float64(w) * 0.16
	headCy := float64(ht) * 0.14
	bodyTop := float64(ht) * 0.24
	bodyHalf := float64(w) * 0.26
	for y := 0; y < ht; y++ {
		fy := float64(y)
		for x := 0; x < w; x++ {
			fx := float64(x)
			v := uint8(bg)
			inHead := math.Hypot(fx-cx, fy-headCy) <= headR
			inBody := fy >= bodyTop && math.Abs(fx-cx) <= bodyHalf
			if inHead || inBody {
				v = fg
			}
			m.Set(y, x, 0, v)
		}
	}
	return m
}

// TestHOGDefaults checks the default geometry and derived sizes.
func TestHOGDefaults(t *testing.T) {
	h := NewDefaultHOG()
	if h.GetDescriptorSize() != 3780 {
		t.Fatalf("GetDescriptorSize() = %d, want 3780", h.GetDescriptorSize())
	}
	// 2x2 cells per 16x16 block, 9 bins => 36 values per block.
	if h.GetBlockHistogramSize() != 36 {
		t.Fatalf("GetBlockHistogramSize() = %d, want 36", h.GetBlockHistogramSize())
	}
	if ws := h.WinSize(); ws.Width != 64 || ws.Height != 128 {
		t.Fatalf("WinSize() = %+v, want 64x128", ws)
	}
}

// TestHOGPeopleDetectorSignCheck is the core determinism test: the default
// people detector must score the planted silhouette positively (a hit) and a
// featureless flat image negatively (no hit).
func TestHOGPeopleDetectorSignCheck(t *testing.T) {
	h := NewDefaultHOG()
	det := h.GetDefaultPeopleDetector()
	if len(det) != h.GetDescriptorSize()+1 {
		t.Fatalf("default detector length = %d, want %d", len(det), h.GetDescriptorSize()+1)
	}
	h.SetSVMDetector(det)

	person := NewGpuMatFromMat(plantedPerson(64, 128))
	locs, conf := h.Detect(person, nil)
	if len(locs) != 1 {
		t.Fatalf("expected exactly one detection on planted person, got %d", len(locs))
	}
	if locs[0] != (cv.Point{X: 0, Y: 0}) {
		t.Fatalf("detection location = %+v, want origin", locs[0])
	}
	if conf[0] <= 0 {
		t.Fatalf("planted person score = %v, want > 0", conf[0])
	}

	flat := cv.NewMat(128, 64, 1)
	flat.SetTo(128)
	if locs, _ := h.Detect(NewGpuMatFromMat(flat), nil); len(locs) != 0 {
		t.Fatalf("expected no detection on flat image, got %d", len(locs))
	}
}

// TestHOGDetectMultiScale exercises the pyramid path and grouping.
func TestHOGDetectMultiScale(t *testing.T) {
	h := NewDefaultHOG()
	h.SetSVMDetector(h.GetDefaultPeopleDetector())
	h.SetGroupThreshold(0) // return raw hits so at least the origin window survives

	// Larger canvas with the silhouette in the top-left window.
	canvas := cv.NewMat(160, 96, 1)
	canvas.SetTo(210)
	person := plantedPerson(64, 128)
	person.CopyTo(canvas, 0, 0)

	rects, scores := h.DetectMultiScale(NewGpuMatFromMat(canvas), NewStream())
	if len(rects) == 0 {
		t.Fatal("expected at least one multi-scale detection")
	}
	if len(rects) != len(scores) {
		t.Fatalf("rects/scores length mismatch %d vs %d", len(rects), len(scores))
	}
	// Grouping enabled should reduce or maintain the raw count without panic.
	h.SetGroupThreshold(1)
	grouped, gscores := h.DetectMultiScale(NewGpuMatFromMat(canvas), nil)
	if len(grouped) != len(gscores) {
		t.Fatalf("grouped rects/scores mismatch %d vs %d", len(grouped), len(gscores))
	}
}

// TestHOGCompute checks the descriptor length matches the geometry.
func TestHOGCompute(t *testing.T) {
	h := NewDefaultHOG()
	img := NewGpuMat(128, 64, 1)
	desc := h.Compute(img, nil)
	if len(desc) != h.GetDescriptorSize() {
		t.Fatalf("Compute length = %d, want %d", len(desc), h.GetDescriptorSize())
	}
}

// TestHOGCustomGeometry verifies NewHOG with a non-default geometry.
func TestHOGCustomGeometry(t *testing.T) {
	h := NewHOG(Size{Width: 32, Height: 32}, Size{Width: 16, Height: 16},
		Size{Width: 8, Height: 8}, Size{Width: 8, Height: 8}, 9)
	// blocksX=blocksY=3, cpb=2x2, 9 bins => 3*3*2*2*9 = 324.
	if h.GetDescriptorSize() != 324 {
		t.Fatalf("custom GetDescriptorSize() = %d, want 324", h.GetDescriptorSize())
	}
}

// TestHOGParams round-trips every parameter accessor.
func TestHOGParams(t *testing.T) {
	h := NewDefaultHOG()

	h.SetHitThreshold(0.25)
	if h.GetHitThreshold() != 0.25 {
		t.Fatalf("hit threshold = %v", h.GetHitThreshold())
	}
	h.SetWinStride(Size{Width: 16, Height: 16})
	if h.GetWinStride() != (Size{Width: 16, Height: 16}) {
		t.Fatalf("win stride = %+v", h.GetWinStride())
	}
	h.SetScaleFactor(1.2)
	if h.GetScaleFactor() != 1.2 {
		t.Fatalf("scale factor = %v", h.GetScaleFactor())
	}
	h.SetNumLevels(32)
	if h.GetNumLevels() != 32 {
		t.Fatalf("num levels = %v", h.GetNumLevels())
	}
	h.SetGroupThreshold(3)
	if h.GetGroupThreshold() != 3 {
		t.Fatalf("group threshold = %v", h.GetGroupThreshold())
	}
	h.SetGammaCorrection(false)
	if h.GetGammaCorrection() {
		t.Fatal("gamma correction should be false")
	}
	h.SetL2HysThreshold(0.3)
	if h.GetL2HysThreshold() != 0.3 {
		t.Fatalf("l2hys = %v", h.GetL2HysThreshold())
	}
	h.SetWinSigma(4.0)
	if h.GetWinSigma() != 4.0 {
		t.Fatalf("win sigma = %v", h.GetWinSigma())
	}
}

// TestHOGSVMRoundTrip checks SetSVMDetector/GetSVMDetector copy semantics.
func TestHOGSVMRoundTrip(t *testing.T) {
	h := NewDefaultHOG()
	if h.GetSVMDetector() != nil {
		t.Fatal("expected nil SVM before SetSVMDetector")
	}
	det := h.GetDefaultPeopleDetector()
	h.SetSVMDetector(det)
	got := h.GetSVMDetector()
	if len(got) != len(det) {
		t.Fatalf("round-trip length %d vs %d", len(got), len(det))
	}
	got[0] = 999 // mutate the copy
	if h.GetSVMDetector()[0] == 999 {
		t.Fatal("GetSVMDetector must return an independent copy")
	}
}

// TestHOGPanics checks the argument-validation panics.
func TestHOGPanics(t *testing.T) {
	mustPanic := func(name string, fn func()) {
		defer func() {
			if recover() == nil {
				t.Fatalf("%s: expected panic", name)
			}
		}()
		fn()
	}
	mustPanic("bad geometry", func() {
		NewHOG(Size{Width: 10, Height: 10}, Size{Width: 3, Height: 3},
			Size{Width: 2, Height: 2}, Size{Width: 2, Height: 2}, 9)
	})
	mustPanic("svm length", func() {
		NewDefaultHOG().SetSVMDetector([]float64{1, 2, 3})
	})
	mustPanic("detect without svm", func() {
		NewDefaultHOG().Detect(NewGpuMat(128, 64, 1), nil)
	})
	mustPanic("scale factor <= 1", func() {
		NewDefaultHOG().SetScaleFactor(1.0)
	})
	mustPanic("nil image", func() {
		NewDefaultHOG().Compute(&GpuMat{}, nil)
	})
}
