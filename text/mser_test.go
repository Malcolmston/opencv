package text

import (
	"testing"

	cv "github.com/malcolmston/opencv"
)

// blobSpec describes a filled rectangle (a synthetic "character") to draw.
type blobSpec struct {
	x, y, w, h int
}

// newBlobImage builds a single-channel image with a uniform background and a set
// of darker filled rectangles.
func newBlobImage(rows, cols int, bg, fg uint8, blobs []blobSpec) *cv.Mat {
	m := cv.NewMat(rows, cols, 1)
	m.SetTo(bg)
	for _, b := range blobs {
		for y := b.y; y < b.y+b.h; y++ {
			for x := b.x; x < b.x+b.w; x++ {
				m.Set(y, x, 0, fg)
			}
		}
	}
	return m
}

func rectsOverlap(a, b cv.Rect) bool {
	return a.X < b.X+b.Width && b.X < a.X+a.Width &&
		a.Y < b.Y+b.Height && b.Y < a.Y+a.Height
}

func TestDetectRegionsMSERCoversBlobs(t *testing.T) {
	blobs := []blobSpec{
		{x: 4, y: 4, w: 5, h: 7}, {x: 14, y: 4, w: 5, h: 7},
		{x: 24, y: 4, w: 5, h: 7}, {x: 34, y: 4, w: 5, h: 7},
		{x: 4, y: 24, w: 5, h: 7}, {x: 14, y: 24, w: 5, h: 7},
		{x: 24, y: 24, w: 5, h: 7}, {x: 34, y: 24, w: 5, h: 7},
	}
	img := newBlobImage(36, 46, 220, 40, blobs)

	boxes := DetectRegionsMSER(img, 5, 10, 200, 0.5)

	if len(boxes) != len(blobs) {
		t.Fatalf("got %d regions, want %d: %+v", len(boxes), len(blobs), boxes)
	}

	total := img.Rows * img.Cols
	// Every blob must be covered by exactly one region, and no region may span
	// the background (the whole image).
	for i, b := range blobs {
		blobRect := cv.Rect{X: b.x, Y: b.y, Width: b.w, Height: b.h}
		matches := 0
		for _, box := range boxes {
			if rectsOverlap(box, blobRect) {
				matches++
			}
		}
		if matches != 1 {
			t.Errorf("blob %d covered by %d regions, want 1", i, matches)
		}
	}
	for _, box := range boxes {
		if box.Width*box.Height > total/2 {
			t.Errorf("region %+v looks like background (area %d of %d)", box, box.Width*box.Height, total)
		}
	}
}

func TestMSERRegionsPixelSets(t *testing.T) {
	blobs := []blobSpec{{x: 5, y: 5, w: 6, h: 6}}
	img := newBlobImage(24, 24, 210, 30, blobs)

	regions := MSERRegions(img, 3, 4, 100, 0.5)
	if len(regions) != 1 {
		t.Fatalf("got %d regions, want 1", len(regions))
	}
	r := regions[0]
	if r.Area != 6*6 {
		t.Errorf("region area = %d, want %d", r.Area, 6*6)
	}
	if len(r.Points) != r.Area {
		t.Errorf("len(Points) = %d, want Area %d", len(r.Points), r.Area)
	}
	// Every returned pixel must actually be a foreground (dark blob) pixel.
	for _, p := range r.Points {
		if img.At(p.Y, p.X, 0) != 30 {
			t.Errorf("point %+v is not a blob pixel (val %d)", p, img.At(p.Y, p.X, 0))
		}
	}
	if r.Rect.Width != 6 || r.Rect.Height != 6 {
		t.Errorf("bbox = %+v, want 6x6", r.Rect)
	}
}

func TestMSERBrightOnDark(t *testing.T) {
	// A bright blob on a dark background must be found via the inverted (MSER-)
	// polarity.
	blobs := []blobSpec{{x: 6, y: 6, w: 8, h: 8}}
	img := newBlobImage(28, 28, 20, 230, blobs) // fg brighter than bg

	regions := MSERRegions(img, 4, 8, 200, 0.5)
	if len(regions) != 1 {
		t.Fatalf("got %d regions, want 1: %+v", len(regions), regions)
	}
	if !regions[0].Bright {
		t.Errorf("expected a bright-polarity region, got %+v", regions[0])
	}
}

func TestMSERDeterministic(t *testing.T) {
	blobs := []blobSpec{{x: 3, y: 3, w: 5, h: 5}, {x: 15, y: 10, w: 5, h: 5}}
	img := newBlobImage(24, 24, 200, 50, blobs)
	a := DetectRegionsMSER(img, 4, 6, 150, 0.5)
	b := DetectRegionsMSER(img, 4, 6, 150, 0.5)
	if len(a) != len(b) {
		t.Fatalf("nondeterministic length %d vs %d", len(a), len(b))
	}
	for i := range a {
		if a[i] != b[i] {
			t.Errorf("region %d differs: %+v vs %+v", i, a[i], b[i])
		}
	}
}
