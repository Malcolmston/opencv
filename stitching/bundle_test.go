package stitching

import (
	"math"
	"testing"
)

// sumSq returns the sum of squares of a residual vector.
func sumSq(v []float64) float64 { return dotf(v, v) }

func TestBundleAdjusterRayReducesError(t *testing.T) {
	fTrue := 280.0
	trueCams, _, matches := syntheticCameras(fTrue, 320, 240, []float64{-0.2, 0, 0.2}, 11)

	// Perturb the focal lengths away from the truth as an estimator would.
	init := make([]CameraParams, len(trueCams))
	copy(init, trueCams)
	for i := range init {
		init[i].Focal = fTrue * 1.2
	}

	before := sumSq(rayResiduals(init, matches))
	refined, ok := BundleAdjusterRay{}.Refine(init, matches)
	if !ok {
		t.Fatal("ray adjuster returned not-ok")
	}
	after := sumSq(rayResiduals(refined, matches))
	if after >= before {
		t.Errorf("ray adjuster did not reduce cost: before=%.6g after=%.6g", before, after)
	}
	// The refined focal should be closer to the truth than the perturbed start.
	initErr := math.Abs(init[0].Focal - fTrue)
	refErr := math.Abs(refined[0].Focal - fTrue)
	if refErr >= initErr {
		t.Errorf("ray adjuster did not improve focal: init err=%.2f refined err=%.2f", initErr, refErr)
	}
	if rel := refErr / fTrue; rel > 0.1 {
		t.Errorf("refined focal %.2f still %.1f%% from true %.2f", refined[0].Focal, rel*100, fTrue)
	}
}

func TestBundleAdjusterReprojReducesError(t *testing.T) {
	fTrue := 300.0
	trueCams, _, matches := syntheticCameras(fTrue, 360, 260, []float64{-0.18, 0, 0.18}, 5)

	init := make([]CameraParams, len(trueCams))
	copy(init, trueCams)
	for i := range init {
		init[i].Focal = fTrue * 0.85
	}

	before := sumSq(reprojResiduals(init, matches))
	refined, ok := BundleAdjusterReproj{MaxIterations: 60}.Refine(init, matches)
	if !ok {
		t.Fatal("reproj adjuster returned not-ok")
	}
	after := sumSq(reprojResiduals(refined, matches))
	if after >= before {
		t.Errorf("reproj adjuster did not reduce cost: before=%.6g after=%.6g", before, after)
	}
	initErr := math.Abs(init[0].Focal - fTrue)
	refErr := math.Abs(refined[0].Focal - fTrue)
	if refErr >= initErr {
		t.Errorf("reproj adjuster did not improve focal: init err=%.2f refined err=%.2f", initErr, refErr)
	}
}

func TestBundleAdjusterNoMatches(t *testing.T) {
	cams := []CameraParams{defaultCamera(200, 100, 100)}
	ba := BundleAdjusterRay{}
	if _, ok := ba.Refine(cams, nil); ok {
		t.Error("expected not-ok with no matches")
	}
}
