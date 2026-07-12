package saliency_test

import (
	"math"
	"testing"

	cv "github.com/malcolmston/opencv"
	"github.com/malcolmston/opencv/saliency"
)

// objectStaticDetectors is the set of new static detectors that should each
// score a distinct bright object above the background.
func objectStaticDetectors() map[string]saliency.StaticSaliency {
	return map[string]saliency.StaticSaliency{
		"itti":           saliency.NewStaticSaliencyIttiKochNiebur(),
		"mbd":            saliency.NewMinimumBarrierSaliency(),
		"frequencytuned": saliency.NewStaticSaliencyFrequencyTuned(),
		"contextaware":   saliency.NewStaticSaliencyContextAware(),
		"gmr":            saliency.NewGMRSaliency(),
		"bms":            saliency.NewStaticSaliencyBooleanMap(),
		"rc":             saliency.NewRegionContrast(),
		"hc":             saliency.NewHistogramContrast(),
	}
}

func TestNewStaticDetectorsPeakOnObject(t *testing.T) {
	img := diskScene(sceneSize, bgValue, fgValue, diskCY, diskCX, diskR)
	for name, det := range objectStaticDetectors() {
		det := det
		t.Run(name, func(t *testing.T) {
			sal := det.ComputeSaliency(img)
			if sal.Rows != img.Rows || sal.Cols != img.Cols || sal.Channels != 1 {
				t.Fatalf("%s map shape = %dx%dx%d, want %dx%dx1", name, sal.Rows, sal.Cols, sal.Channels, img.Rows, img.Cols)
			}
			obj, bg := diskMeans(sal, diskCY, diskCX, diskR)
			t.Logf("%s: object=%.2f background=%.2f", name, obj, bg)
			if obj <= bg+20 {
				t.Errorf("%s object mean saliency %.2f not sufficiently above background %.2f", name, obj, bg)
			}
		})
	}
}

func TestNewStaticDetectorsDeterministic(t *testing.T) {
	img := diskScene(sceneSize, bgValue, fgValue, diskCY, diskCX, diskR)
	for name, det := range objectStaticDetectors() {
		det := det
		t.Run(name, func(t *testing.T) {
			a := det.ComputeSaliency(img)
			b := det.ComputeSaliency(img)
			for i := range a.Data {
				if a.Data[i] != b.Data[i] {
					t.Fatalf("%s not deterministic at %d: %d vs %d", name, i, a.Data[i], b.Data[i])
				}
			}
		})
	}
}

func TestNewStaticDetectorsImplementInterface(t *testing.T) {
	var _ saliency.StaticSaliency = saliency.NewStaticSaliencyIttiKochNiebur()
	var _ saliency.StaticSaliency = saliency.NewMinimumBarrierSaliency()
	var _ saliency.StaticSaliency = saliency.NewStaticSaliencyFrequencyTuned()
	var _ saliency.StaticSaliency = saliency.NewStaticSaliencyContextAware()
	var _ saliency.StaticSaliency = saliency.NewGMRSaliency()
	var _ saliency.StaticSaliency = saliency.NewStaticSaliencyBooleanMap()
	var _ saliency.StaticSaliency = saliency.NewRegionContrast()
	var _ saliency.StaticSaliency = saliency.NewHistogramContrast()
}

func TestObjectnessCascadeProposesObjectWindow(t *testing.T) {
	img := diskScene(sceneSize, bgValue, fgValue, diskCY, diskCX, diskR)
	boxes := saliency.NewObjectnessCascade().ComputeObjectness(img)
	if len(boxes) == 0 {
		t.Fatal("expected at least one cascade proposal")
	}
	for i := 1; i < len(boxes); i++ {
		if boxes[i].Score > boxes[i-1].Score {
			t.Fatalf("cascade boxes not sorted by score at %d: %.3f > %.3f", i, boxes[i].Score, boxes[i-1].Score)
		}
	}
	top := boxes[0]
	if !(diskCX >= top.X && diskCX < top.X+top.W && diskCY >= top.Y && diskCY < top.Y+top.H) {
		t.Errorf("top cascade proposal %+v does not overlap object centre (%d,%d)", top, diskCX, diskCY)
	}
}

func TestObjectnessCascadeSuppressesDuplicates(t *testing.T) {
	img := diskScene(sceneSize, bgValue, fgValue, diskCY, diskCX, diskR)
	boxes := saliency.NewObjectnessCascade().ComputeObjectness(img)
	for i := 0; i < len(boxes); i++ {
		for j := i + 1; j < len(boxes); j++ {
			a, b := boxes[i], boxes[j]
			ix0 := maxi(a.X, b.X)
			iy0 := maxi(a.Y, b.Y)
			ix1 := mini(a.X+a.W, b.X+b.W)
			iy1 := mini(a.Y+a.H, b.Y+b.H)
			iw, ih := ix1-ix0, iy1-iy0
			if iw <= 0 || ih <= 0 {
				continue
			}
			inter := float64(iw * ih)
			union := float64(a.W*a.H+b.W*b.H) - inter
			if union > 0 && inter/union > 0.6 {
				t.Errorf("boxes %+v and %+v overlap with IoU %.2f > 0.6 after NMS", a, b, inter/union)
			}
		}
	}
}

func maxi(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func mini(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// buildFixationMap marks a small cluster of fixation pixels centred on the disk.
func buildFixationMap(size, cy, cx, r int) *cv.Mat {
	f := cv.NewMat(size, size, 1)
	for y := cy - r; y <= cy+r; y++ {
		for x := cx - r; x <= cx+r; x++ {
			if y >= 0 && y < size && x >= 0 && x < size {
				f.Set(y, x, 0, 255)
			}
		}
	}
	return f
}

func TestEvalMetricsRewardAlignedMap(t *testing.T) {
	img := diskScene(sceneSize, bgValue, fgValue, diskCY, diskCX, diskR)
	good := saliency.NewStaticSaliencyFineGrained().ComputeSaliency(img)
	fix := buildFixationMap(sceneSize, diskCY, diskCX, 2)

	auc := saliency.AUCJudd(good, fix)
	if math.IsNaN(auc) || auc <= 0.5 {
		t.Errorf("AUCJudd on aligned map = %.3f, want > 0.5", auc)
	}
	nss := saliency.NSS(good, fix)
	if math.IsNaN(nss) || nss <= 0 {
		t.Errorf("NSS on aligned map = %.3f, want > 0", nss)
	}

	// CC of a map with itself is 1.
	if cc := saliency.CC(good, good); math.Abs(cc-1) > 1e-9 {
		t.Errorf("CC(map, map) = %.6f, want 1", cc)
	}
	// SIM of a map with itself is 1.
	if sim := saliency.SIM(good, good); math.Abs(sim-1) > 1e-9 {
		t.Errorf("SIM(map, map) = %.6f, want 1", sim)
	}
	// KLDiv of a map with itself is 0.
	if kl := saliency.KLDiv(good, good); math.Abs(kl) > 1e-6 {
		t.Errorf("KLDiv(map, map) = %.6f, want 0", kl)
	}
}

func TestAUCJuddDistinguishesGoodFromBad(t *testing.T) {
	img := diskScene(sceneSize, bgValue, fgValue, diskCY, diskCX, diskR)
	good := saliency.NewMinimumBarrierSaliency().ComputeSaliency(img)
	fix := buildFixationMap(sceneSize, diskCY, diskCX, 2)

	// An inverted map should score worse than the aligned one.
	bad := cv.NewMat(good.Rows, good.Cols, 1)
	for i, v := range good.Data {
		bad.Data[i] = 255 - v
	}
	aGood := saliency.AUCJudd(good, fix)
	aBad := saliency.AUCJudd(bad, fix)
	t.Logf("AUCJudd good=%.3f bad=%.3f", aGood, aBad)
	if aGood <= aBad {
		t.Errorf("AUCJudd did not rank aligned map above inverted map (%.3f vs %.3f)", aGood, aBad)
	}
}

func TestSaliencyToHeatmap(t *testing.T) {
	// A controlled ramp: a maximum-saliency pixel and a zero pixel.
	sal := cv.NewMat(4, 4, 1)
	sal.Set(0, 0, 0, 255) // hot
	sal.Set(3, 3, 0, 0)   // cold
	hm := saliency.SaliencyToHeatmap(sal)
	if hm.Channels != 3 {
		t.Fatalf("heatmap channels = %d, want 3", hm.Channels)
	}
	if hm.Rows != sal.Rows || hm.Cols != sal.Cols {
		t.Fatalf("heatmap size = %dx%d, want %dx%d", hm.Rows, hm.Cols, sal.Rows, sal.Cols)
	}
	// Maximum saliency should be reddish (R>B); zero saliency bluish (B>R).
	if hm.At(0, 0, 0) <= hm.At(0, 0, 2) {
		t.Errorf("hot pixel not reddish: R=%d B=%d", hm.At(0, 0, 0), hm.At(0, 0, 2))
	}
	if hm.At(3, 3, 2) <= hm.At(3, 3, 0) {
		t.Errorf("cold pixel not bluish: R=%d B=%d", hm.At(3, 3, 0), hm.At(3, 3, 2))
	}
}

func TestAdaptiveBinaryMapIsolatesObject(t *testing.T) {
	img := diskScene(sceneSize, bgValue, fgValue, diskCY, diskCX, diskR)
	sal := saliency.NewMinimumBarrierSaliency().ComputeSaliency(img)
	mask := saliency.AdaptiveBinaryMap(sal, 2.0)
	if mask.Channels != 1 {
		t.Fatalf("mask channels = %d, want 1", mask.Channels)
	}
	for _, v := range mask.Data {
		if v != 0 && v != 255 {
			t.Fatalf("adaptive binary map not binary: %d", v)
		}
	}
	if mask.At(diskCY, diskCX, 0) != 255 {
		t.Errorf("object centre not foreground: %d", mask.At(diskCY, diskCX, 0))
	}
	if mask.At(0, 0, 0) != 0 {
		t.Errorf("background corner not zero: %d", mask.At(0, 0, 0))
	}
}

func TestCenterBiasPriorPeaksAtCentre(t *testing.T) {
	prior := saliency.CenterBiasPrior(40, 60, 0.3)
	if prior.Rows != 40 || prior.Cols != 60 || prior.Channels != 1 {
		t.Fatalf("prior shape = %dx%dx%d, want 40x60x1", prior.Rows, prior.Cols, prior.Channels)
	}
	if prior.At(20, 30, 0) != 255 {
		t.Errorf("centre value = %d, want 255", prior.At(20, 30, 0))
	}
	if prior.At(0, 0, 0) >= prior.At(20, 30, 0) {
		t.Errorf("corner %d not less than centre %d", prior.At(0, 0, 0), prior.At(20, 30, 0))
	}
}

func TestApplyCenterBiasSuppressesCorners(t *testing.T) {
	img := diskScene(sceneSize, bgValue, fgValue, diskCY, diskCX, diskR)
	sal := saliency.NewStaticSaliencyFineGrained().ComputeSaliency(img)
	biased := saliency.ApplyCenterBias(sal, 0.35)
	if biased.Rows != sal.Rows || biased.Cols != sal.Cols {
		t.Fatalf("biased map size mismatch")
	}
	// The centred object should remain salient.
	obj, bg := diskMeans(biased, diskCY, diskCX, diskR)
	if obj <= bg {
		t.Errorf("center-biased object %.2f not above background %.2f", obj, bg)
	}
}
