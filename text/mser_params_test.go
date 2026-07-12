package text

import (
	"testing"
)

func TestMSERRegionsWithParamsFindsBlobs(t *testing.T) {
	blobs := []blobSpec{
		{x: 4, y: 4, w: 6, h: 8}, {x: 16, y: 4, w: 6, h: 8},
	}
	img := newBlobImage(20, 30, 220, 40, blobs)
	p := DefaultMSERParams()
	p.MinArea = 10
	p.MaxArea = 200 // reject the whole-image background level set
	regions := MSERRegionsWithParams(img, p)
	if len(regions) != 2 {
		t.Fatalf("got %d regions, want 2: %+v", len(regions), regions)
	}
}

func TestMSERParamsPolaritySelects(t *testing.T) {
	// A bright blob on a dark background is only found by the bright polarity.
	blobs := []blobSpec{{x: 6, y: 6, w: 8, h: 8}}
	img := newBlobImage(28, 28, 20, 230, blobs)

	dark := DefaultMSERParams()
	dark.Polarity = MSERDark
	dark.MinArea = 8
	dark.MaxArea = 200
	if got := MSERRegionsWithParams(img, dark); len(got) != 0 {
		t.Errorf("dark polarity found %d bright blobs, want 0", len(got))
	}

	bright := DefaultMSERParams()
	bright.Polarity = MSERBright
	bright.MinArea = 8
	bright.MaxArea = 200
	got := MSERRegionsWithParams(img, bright)
	if len(got) != 1 || !got[0].Bright {
		t.Errorf("bright polarity got %+v, want one bright region", got)
	}
}

func TestMSERMinDiversityPrunes(t *testing.T) {
	// A single blob can surface at several nearly-identical threshold levels.
	// Diversity pruning collapses those to one; disabling it keeps at least as
	// many.
	blobs := []blobSpec{{x: 5, y: 5, w: 8, h: 8}}
	img := newBlobImage(24, 24, 210, 30, blobs)

	withDiv := DefaultMSERParams()
	withDiv.MinArea = 8
	withDiv.MaxArea = 200
	withDiv.MinDiversity = 0.2

	noDiv := withDiv
	noDiv.MinDiversity = 0

	a := MSERRegionsWithParams(img, withDiv)
	b := MSERRegionsWithParams(img, noDiv)
	if len(a) != 1 {
		t.Errorf("with diversity got %d regions, want 1: %+v", len(a), a)
	}
	if len(b) < len(a) {
		t.Errorf("disabling diversity produced fewer regions (%d < %d)", len(b), len(a))
	}
}

func TestDetectRegionsMSERParamsBoxes(t *testing.T) {
	blobs := []blobSpec{{x: 4, y: 4, w: 6, h: 8}, {x: 16, y: 4, w: 6, h: 8}}
	img := newBlobImage(20, 30, 220, 40, blobs)
	p := DefaultMSERParams()
	p.MinArea = 10
	p.MaxArea = 200
	boxes := DetectRegionsMSERParams(img, p)
	if len(boxes) != 2 {
		t.Fatalf("got %d boxes, want 2", len(boxes))
	}
}
