package tracking

import (
	"math"
	"testing"

	cv "github.com/malcolmston/opencv"
)

// blobProb builds a single-channel probability image with a Gaussian blob of the
// given center and sigma, peaking at 255.
func blobProb(rows, cols int, cx, cy, sigma float64) *cv.Mat {
	m := cv.NewMat(rows, cols, 1)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			dx := float64(x) - cx
			dy := float64(y) - cy
			v := 255 * math.Exp(-(dx*dx+dy*dy)/(2*sigma*sigma))
			m.Data[y*cols+x] = clampU8(v)
		}
	}
	return m
}

func TestMeanShiftConverges(t *testing.T) {
	prob := blobProb(64, 64, 40, 30, 6)
	// Start a window whose centre is well away from the blob.
	win := NewRect(6, 6, 24, 24) // centre (18, 18)
	result, iters := MeanShift(prob, win, NewTermCriteria(30, 0.3))
	c := result.Center()
	requireTrue(t, iters >= 1, "expected at least one iteration")
	requireTrue(t, approx(c.X, 40, 4), "converged X = %v, want ~40", c.X)
	requireTrue(t, approx(c.Y, 30, 4), "converged Y = %v, want ~30", c.Y)
}

func TestBackProjectionPipeline(t *testing.T) {
	// An image whose central square is bright (value 200) on a dark (value 30)
	// background.
	rows, cols := 40, 40
	img := cv.NewMat(rows, cols, 1)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			v := uint8(30)
			if x >= 15 && x < 25 && y >= 15 && y < 25 {
				v = 200
			}
			img.Data[y*cols+x] = v
		}
	}
	// Model histogram taken from the bright square.
	hist := CalcHistGray(img, NewRect(15, 15, 10, 10), 32)
	hist.Normalize()
	bp := CalcBackProjection(img, hist)
	// Inside the square the back-projection should be high, outside low.
	requireTrue(t, bp.Data[20*cols+20] > 200, "inside square prob = %v, want high", bp.Data[20*cols+20])
	requireTrue(t, bp.Data[2*cols+2] < 30, "background prob = %v, want low", bp.Data[2*cols+2])
}

func TestCamShiftAdapts(t *testing.T) {
	prob := blobProb(64, 64, 32, 32, 8)
	win := NewRect(20, 20, 24, 24)
	rr, adapted, iters := CamShift(prob, win, NewTermCriteria(20, 0.5))
	requireTrue(t, iters >= 1, "expected iterations")
	requireTrue(t, approx(rr.Center.X, 32, 4), "camshift center X = %v", rr.Center.X)
	requireTrue(t, approx(rr.Center.Y, 32, 4), "camshift center Y = %v", rr.Center.Y)
	requireTrue(t, rr.Width > 0 && rr.Height > 0, "camshift produced degenerate size")
	requireTrue(t, !adapted.Empty(), "adapted window should be non-empty")
}
