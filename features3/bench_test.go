package features3

import "testing"

// BenchmarkLoGBlobs measures the heaviest routine: the multi-scale
// Laplacian-of-Gaussian blob detector, which builds a Gaussian scale space and
// scans it for 3x3x3 extrema.
func BenchmarkLoGBlobs(b *testing.B) {
	img := texturedImage(128, 128)
	// Stamp a few bright discs so there are genuine blobs to find.
	for _, c := range [][3]int{{32, 32, 6}, {90, 40, 8}, {64, 96, 5}} {
		cx, cy, r := c[0], c[1], c[2]
		r2 := r * r
		for y := -r; y <= r; y++ {
			for x := -r; x <= r; x++ {
				if x*x+y*y <= r2 {
					img.Data[(cy+y)*img.Cols+(cx+x)] = 255
				}
			}
		}
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = LoGBlobs(img, 2, 8, 6, 50)
	}
}
