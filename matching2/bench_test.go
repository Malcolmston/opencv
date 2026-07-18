package matching2

import (
	"math"
	"testing"

	"github.com/malcolmston/opencv/core"
)

// benchScene builds a larger deterministic two-view point set for benchmarking
// the fundamental-matrix RANSAC estimator, the heaviest routine in the package
// (many eight-point fits, each an SVD via Jacobi eigendecomposition).
func benchScene(n int) (img1, img2 []core.Point2d) {
	R2 := RodriguesToMatrix([3]float64{0, 15 * math.Pi / 180, 0.05})
	t2 := [3]float64{1.0, 0.1, 0.2}
	world := make([]core.Point3d, n)
	for i := 0; i < n; i++ {
		// Deterministic pseudo-random spread via trig, no rand package.
		fi := float64(i)
		world[i] = core.Point3d{
			X: math.Sin(fi*1.3) * 1.5,
			Y: math.Cos(fi*0.7) * 1.5,
			Z: 5 + math.Sin(fi*0.37)*2,
		}
	}
	img1 = ProjectPoints(world, Mat3Identity(), [3]float64{0, 0, 0}, testK)
	img2 = ProjectPoints(world, R2, t2, testK)
	// Corrupt 20% of the second-image points into outliers.
	for i := 0; i < n; i += 5 {
		img2[i] = core.Point2d{X: img2[i].X + 120, Y: img2[i].Y - 90}
	}
	return img1, img2
}

func BenchmarkFindFundamentalMatRANSAC(b *testing.B) {
	img1, img2 := benchScene(100)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		res := FindFundamentalMatRANSAC(img1, img2, 1.5, 200, DefaultRANSACSeed)
		if !res.Ok {
			b.Fatal("RANSAC failed")
		}
	}
}
