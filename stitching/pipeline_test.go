package stitching

import (
	"math"
	"testing"

	cv "github.com/malcolmston/opencv"
)

func TestPipelinePanoramaCoversUnion(t *testing.T) {
	base := syntheticTexture(90, 170, 24680)
	shift := 70
	a, b := horizontalCrops(base, 110, shift, 100)

	p := NewPipeline(ModePanorama)
	pano, err := p.Stitch([]*cv.Mat{a, b})
	if err != nil {
		t.Fatalf("pipeline Stitch: %v", err)
	}
	// A covers columns [0,110), B covers [shift, shift+100) = [70,170): the union
	// spans the full base width and height.
	if pano.Rows != base.Rows {
		t.Errorf("panorama rows = %d, want %d", pano.Rows, base.Rows)
	}
	if math.Abs(float64(pano.Cols-base.Cols)) > 3 {
		t.Errorf("panorama cols = %d, want ≈ %d (union bounds)", pano.Cols, base.Cols)
	}
}

func TestPipelineScansMode(t *testing.T) {
	base := syntheticTexture(80, 160, 1357)
	a, b := horizontalCrops(base, 100, 60, 100)

	p := NewPipeline(ModeScans)
	if _, ok := p.warper.(PlaneWarper); !ok {
		t.Errorf("scans mode default warper = %T, want PlaneWarper", p.warper)
	}
	pano, err := p.Stitch([]*cv.Mat{a, b})
	if err != nil {
		t.Fatalf("scans Stitch: %v", err)
	}
	if pano.Rows != base.Rows {
		t.Errorf("rows = %d, want %d", pano.Rows, base.Rows)
	}
	if math.Abs(float64(pano.Cols-base.Cols)) > 3 {
		t.Errorf("cols = %d, want ≈ %d", pano.Cols, base.Cols)
	}
}

func TestPipelineExposureAndSeam(t *testing.T) {
	base := syntheticTexture(90, 170, 2468)
	shift := 70
	a, b := horizontalCrops(base, 110, shift, 100)
	// Brighten B to create an exposure step across the seam.
	bBright := b.Clone()
	for i := range bBright.Data {
		bBright.Data[i] = clampUint8(float64(bBright.Data[i]) + 40)
	}

	p := NewPipeline(ModePanorama)
	p.SetExposureCompensator(&GainCompensator{})
	p.SetSeamFinder(&DpSeamFinder{})
	p.SetBlender(Feather{})

	pano, err := p.Stitch([]*cv.Mat{a, bBright})
	if err != nil {
		t.Fatalf("pipeline Stitch: %v", err)
	}
	// Canvas size is unaffected by exposure/seam stages; still the union bounds.
	if pano.Rows != base.Rows {
		t.Errorf("rows = %d, want %d", pano.Rows, base.Rows)
	}
	if math.Abs(float64(pano.Cols-base.Cols)) > 3 {
		t.Errorf("cols = %d, want ≈ %d", pano.Cols, base.Cols)
	}
	// The panorama must be non-trivial (covered pixels present).
	var nonZero int
	for _, v := range pano.Data {
		if v > 0 {
			nonZero++
		}
	}
	if nonZero < pano.Rows*pano.Cols/2 {
		t.Errorf("panorama mostly empty: %d/%d covered", nonZero, pano.Rows*pano.Cols)
	}
}

func TestPipelineCylindricalWarper(t *testing.T) {
	base := syntheticTexture(90, 200, 9753)
	a, b := horizontalCrops(base, 130, 80, 120)

	p := NewPipeline(ModePanorama)
	p.SetWarper(CylindricalWarper{})
	p.Focal = 400 // gentle curvature keeps overlap features matchable

	pano, err := p.Stitch([]*cv.Mat{a, b})
	if err != nil {
		t.Fatalf("cylindrical pipeline Stitch: %v", err)
	}
	if pano.Rows <= 0 || pano.Cols <= 0 {
		t.Fatalf("empty panorama %dx%d", pano.Cols, pano.Rows)
	}
	// The stitched result must be wider than a single warped input (the two crops
	// were joined, not merely overlaid).
	warpedA, _ := CylindricalWarper{}.Warp(a, 400)
	if pano.Cols <= warpedA.Cols {
		t.Errorf("panorama cols = %d, want > single warped width %d", pano.Cols, warpedA.Cols)
	}
}

func TestPipelineErrors(t *testing.T) {
	p := NewPipeline(ModePanorama)
	if _, err := p.Stitch(nil); err != ErrNoImages {
		t.Errorf("Stitch(nil) = %v, want ErrNoImages", err)
	}
	one := syntheticTexture(20, 20, 1)
	out, err := p.Stitch([]*cv.Mat{one})
	if err != nil || out.Cols != 20 {
		t.Errorf("single-image passthrough failed: err=%v", err)
	}
}
