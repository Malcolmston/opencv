package tracking

import (
	"testing"

	cv "github.com/malcolmston/opencv"
)

// squareFrame renders a bright square of the given size at (sqX, sqY) on a
// lightly textured mid-gray background.
func squareFrame(rows, cols, sqX, sqY, size int) *cv.Mat {
	m := cv.NewMat(rows, cols, 1)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			v := 90 + (x%7)*3 + (y%5)*2 // faint non-flat background
			if x >= sqX && x < sqX+size && y >= sqY && y < sqY+size {
				v = 230
			}
			m.Data[y*cols+x] = clampU8(float64(v))
		}
	}
	return m
}

// cropMat copies the sub-rectangle r of a single-channel image.
func cropMat(src *cv.Mat, r Rect) *cv.Mat {
	out := cv.NewMat(r.Height, r.Width, 1)
	for j := 0; j < r.Height; j++ {
		for i := 0; i < r.Width; i++ {
			out.Data[j*r.Width+i] = src.Data[(r.Y+j)*src.Cols+(r.X+i)]
		}
	}
	return out
}

func TestMatchTemplateNCC(t *testing.T) {
	img := squareFrame(64, 64, 30, 24, 12)
	tpl := cropMat(img, NewRect(28, 22, 16, 16))
	res := MatchTemplateNCC(img, tpl)
	requireTrue(t, res.X == 28 && res.Y == 22, "match at (%d,%d), want (28,22)", res.X, res.Y)
	requireTrue(t, res.Score > 0.99, "score = %v, want ~1", res.Score)
	box := res.BoundingBox()
	requireTrue(t, box.Width == 16 && box.Height == 16, "bounding box size wrong")
}

func TestKCFTrackerFollowsSquare(t *testing.T) {
	frame0 := squareFrame(80, 80, 30, 30, 14)
	tr := NewKCFTracker(DefaultKCFParams())
	tr.Init(frame0, NewRect(28, 28, 18, 18))

	// Move the square by (3, 2); the tracker should follow.
	frame1 := squareFrame(80, 80, 33, 32, 14)
	box, conf := tr.Update(frame1)
	requireTrue(t, conf > 0.5, "confidence = %v, want > 0.5", conf)
	requireTrue(t, approx(float64(box.X), 31, 2), "box X = %d, want ~31", box.X)
	requireTrue(t, approx(float64(box.Y), 30, 2), "box Y = %d, want ~30", box.Y)

	// A second step of the same motion.
	frame2 := squareFrame(80, 80, 36, 34, 14)
	box2, _ := tr.Update(frame2)
	requireTrue(t, box2.X > box.X, "tracker should keep moving right")
}
