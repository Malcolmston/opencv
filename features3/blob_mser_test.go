package features3

import (
	"math"
	"testing"
)

// blobNearCenter reports whether any blob lies within tol pixels of (cx, cy).
func blobNearCenter(blobs []Blob, cx, cy, tol float64) (Blob, bool) {
	for _, b := range blobs {
		if math.Hypot(b.X-cx, b.Y-cy) <= tol {
			return b, true
		}
	}
	return Blob{}, false
}

func TestLoGBlobsDetectsDisc(t *testing.T) {
	img := filledDisc(41, 41, 20, 20, 6, 255)
	blobs := LoGBlobs(img, 2, 6, 5, 50)
	if len(blobs) == 0 {
		t.Fatal("LoG found no blobs on a disc")
	}
	b, ok := blobNearCenter(blobs, 20, 20, 3)
	if !ok {
		t.Fatalf("no LoG blob near disc centre; got %d blobs", len(blobs))
	}
	if b.Sigma < 2 || b.Sigma > 6 {
		t.Fatalf("blob sigma %.2f out of scanned range", b.Sigma)
	}
	if b.Radius() <= 0 || b.Area() <= 0 {
		t.Fatal("blob geometry non-positive")
	}
}

func TestDoGBlobsDetectsDisc(t *testing.T) {
	img := filledDisc(41, 41, 20, 20, 6, 255)
	blobs := DoGBlobs(img, 2, 6, 4, 5)
	if _, ok := blobNearCenter(blobs, 20, 20, 3); !ok {
		t.Fatalf("no DoG blob near disc centre; got %d blobs", len(blobs))
	}
}

func TestDoHBlobsDetectsDisc(t *testing.T) {
	img := filledDisc(41, 41, 20, 20, 6, 255)
	blobs := DoHBlobs(img, 2, 6, 5, 10)
	if _, ok := blobNearCenter(blobs, 20, 20, 3); !ok {
		t.Fatalf("no DoH blob near disc centre; got %d blobs", len(blobs))
	}
}

func TestBlobToKeyPoint(t *testing.T) {
	b := Blob{X: 5, Y: 6, Sigma: 2, Response: 3}
	kp := b.ToKeyPoint()
	if kp.Pt.X != 5 || kp.Pt.Y != 6 {
		t.Fatalf("ToKeyPoint position wrong: %+v", kp)
	}
	if math.Abs(kp.Size-2*b.Radius()) > 1e-9 {
		t.Fatalf("ToKeyPoint size wrong: %v", kp.Size)
	}
}

func TestGaussianScaleSpaceAndDoG(t *testing.T) {
	img := filledDisc(31, 31, 15, 15, 5, 255)
	space := GaussianScaleSpace(img, []float64{1, 2, 4})
	if len(space) != 3 {
		t.Fatalf("scale space len %d", len(space))
	}
	// A larger blur spreads energy: the peak value decreases with sigma.
	peak := func(idx int) float64 {
		var m float64
		for _, v := range space[idx].Data {
			if v > m {
				m = v
			}
		}
		return m
	}
	if !(peak(0) >= peak(1) && peak(1) >= peak(2)) {
		t.Fatalf("blur did not reduce peak: %.1f %.1f %.1f", peak(0), peak(1), peak(2))
	}
	dog := DifferenceOfGaussians(img, 1, 3)
	// At the bright disc centre the smaller-sigma blur exceeds the larger one.
	if dog.At(15, 15) <= 0 {
		t.Fatalf("DoG at bright centre = %.3f, want positive", dog.At(15, 15))
	}
}

func TestLaplacianAndHessianSigns(t *testing.T) {
	img := filledDisc(31, 31, 15, 15, 5, 255)
	log := LaplacianResponse(img, 2)
	// Bright blob centre gives a negative Laplacian.
	if log.At(15, 15) >= 0 {
		t.Fatalf("LoG centre = %.3f, want negative", log.At(15, 15))
	}
	doh := HessianDeterminant(img, 2)
	if doh.At(15, 15) <= 0 {
		t.Fatalf("Hessian determinant centre = %.3f, want positive", doh.At(15, 15))
	}
}

func TestMSERFindsBrightSquare(t *testing.T) {
	img := filledSquare(40, 40, 12, 12, 27, 27, 220)
	// The square is 16x16 = 256 px.
	regions := MSERRegions(img, 5, 30, 1000, 0.25)
	if len(regions) == 0 {
		t.Fatal("MSER found no regions")
	}
	found := false
	for _, r := range regions {
		if !r.Dark && math.Abs(float64(r.Area()-256)) <= 20 {
			found = true
			c := r.Centroid()
			if math.Abs(c.X-19.5) > 3 || math.Abs(c.Y-19.5) > 3 {
				t.Fatalf("bright region centroid off: %+v", c)
			}
			rect := r.BoundingRect()
			if rect.Dx() < 14 || rect.Dx() > 20 {
				t.Fatalf("bright region width %d unexpected", rect.Dx())
			}
		}
	}
	if !found {
		t.Fatalf("MSER did not recover the bright square; regions=%d", len(regions))
	}
}
