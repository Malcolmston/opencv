package connected

import (
	"math"
	"testing"

	cv "github.com/malcolmston/opencv"
)

func TestAnalyzeBlobs(t *testing.T) {
	img := matFromRows([]string{
		".....",
		".###.",
		".#.#.",
		".###.",
		".....",
	})
	blobs := AnalyzeBlobs(img, Conn8)
	if len(blobs) != 1 {
		t.Fatalf("got %d blobs, want 1", len(blobs))
	}
	b := blobs[0]
	if b.Area != 8 {
		t.Errorf("area = %d, want 8", b.Area)
	}
	if b.Holes != 1 {
		t.Errorf("holes = %d, want 1", b.Holes)
	}
	// Every pixel of the ring is on the boundary, so perimeter == area.
	if b.Perimeter != 8 {
		t.Errorf("perimeter = %d, want 8", b.Perimeter)
	}
	if b.Extent() <= 0 || b.Extent() > 1 {
		t.Errorf("extent = %v, out of (0,1]", b.Extent())
	}
	// EquivalentDiameter of area 8 = sqrt(32/pi) ~= 3.19.
	if d := b.EquivalentDiameter(); math.Abs(d-math.Sqrt(32/math.Pi)) > 1e-9 {
		t.Errorf("equiv diameter = %v", d)
	}
}

func TestBlobCircularity(t *testing.T) {
	disc := matFromRows([]string{
		".###.",
		"#####",
		"#####",
		"#####",
		".###.",
	})
	b := AnalyzeBlobs(disc, Conn8)[0]
	// Circularity is the isoperimetric factor over the pixel-count perimeter;
	// verify it against the definition using the reported area and perimeter.
	want := 4 * math.Pi * float64(b.Area) / (float64(b.Perimeter) * float64(b.Perimeter))
	if math.Abs(b.Circularity()-want) > 1e-12 {
		t.Errorf("circularity = %v, want %v", b.Circularity(), want)
	}
	if b.Circularity() <= 0 {
		t.Errorf("circularity must be positive, got %v", b.Circularity())
	}
	// A blob with no perimeter cannot occur here, but the guard must hold.
	if (Blob{}).Circularity() != 0 {
		t.Errorf("zero blob circularity must be 0")
	}
}

func TestBlobHolesPerComponent(t *testing.T) {
	// Two rings side by side: each blob has exactly one hole.
	img := matFromRows([]string{
		"###.###",
		"#.#.#.#",
		"###.###",
	})
	blobs := AnalyzeBlobs(img, Conn8)
	if len(blobs) != 2 {
		t.Fatalf("got %d blobs, want 2", len(blobs))
	}
	for i, b := range blobs {
		if b.Holes != 1 {
			t.Errorf("blob %d holes = %d, want 1", i, b.Holes)
		}
	}
}

// benchImage builds a deterministic w x h checkerboard-of-blocks image so the
// benchmark exercises many components, holes and boundaries.
func benchImage(w, h int) *cv.Mat {
	m := cv.NewMat(h, w, 1)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			if (x/3+y/3)%2 == 0 && x%3 != 1 || y%7 == 0 {
				m.Data[y*w+x] = 255
			}
		}
	}
	return m
}

func BenchmarkLabel(b *testing.B) {
	img := benchImage(256, 256)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		lbl := Label(img, Conn8)
		if lbl.Count == 0 {
			b.Fatal("no components")
		}
	}
}
