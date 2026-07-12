package saliency_test

import (
	"testing"

	cv "github.com/malcolmston/opencv"
	"github.com/malcolmston/opencv/saliency"
)

// Scene geometry shared by the static-saliency tests: a single bright, compact
// disk on a flat dark background — "one distinct object on a flat background".
const (
	sceneSize = 64
	bgValue   = 30
	fgValue   = 220
	diskCY    = 32
	diskCX    = 32
	diskR     = 6
)

// diskScene builds a size×size single-channel image with a flat background of
// value bg and a filled bright disk of value fg.
func diskScene(size, bg, fg, cy, cx, r int) *cv.Mat {
	m := cv.NewMat(size, size, 1)
	m.SetTo(uint8(bg))
	for y := 0; y < size; y++ {
		for x := 0; x < size; x++ {
			dy, dx := y-cy, x-cx
			if dy*dy+dx*dx <= r*r {
				m.Set(y, x, 0, uint8(fg))
			}
		}
	}
	return m
}

// squareScene builds a size×size single-channel image with a flat background of
// value bg and a filled square of value fg spanning [y0,y0+side)×[x0,x0+side).
func squareScene(size, bg, fg, y0, x0, side int) *cv.Mat {
	m := cv.NewMat(size, size, 1)
	m.SetTo(uint8(bg))
	for y := y0; y < y0+side; y++ {
		for x := x0; x < x0+side; x++ {
			m.Set(y, x, 0, uint8(fg))
		}
	}
	return m
}

// diskMeans returns the mean saliency inside the disk and over the background.
func diskMeans(sal *cv.Mat, cy, cx, r int) (object, background float64) {
	var objSum, objN, bgSum, bgN float64
	for y := 0; y < sal.Rows; y++ {
		for x := 0; x < sal.Cols; x++ {
			v := float64(sal.At(y, x, 0))
			dy, dx := y-cy, x-cx
			if dy*dy+dx*dx <= r*r {
				objSum += v
				objN++
			} else {
				bgSum += v
				bgN++
			}
		}
	}
	return objSum / objN, bgSum / bgN
}

func TestSpectralResidualPeaksOnObject(t *testing.T) {
	img := diskScene(sceneSize, bgValue, fgValue, diskCY, diskCX, diskR)
	sal := saliency.NewStaticSaliencySpectralResidual().ComputeSaliency(img)

	if sal.Rows != img.Rows || sal.Cols != img.Cols || sal.Channels != 1 {
		t.Fatalf("saliency map shape = %dx%dx%d, want %dx%dx1", sal.Rows, sal.Cols, sal.Channels, img.Rows, img.Cols)
	}
	obj, bg := diskMeans(sal, diskCY, diskCX, diskR)
	t.Logf("spectral residual: object=%.2f background=%.2f", obj, bg)
	if obj <= bg+20 {
		t.Errorf("object mean saliency %.2f not sufficiently above background %.2f", obj, bg)
	}
}

func TestFineGrainedPeaksOnObject(t *testing.T) {
	img := diskScene(sceneSize, bgValue, fgValue, diskCY, diskCX, diskR)
	sal := saliency.NewStaticSaliencyFineGrained().ComputeSaliency(img)

	if sal.Rows != img.Rows || sal.Cols != img.Cols || sal.Channels != 1 {
		t.Fatalf("saliency map shape = %dx%dx%d, want %dx%dx1", sal.Rows, sal.Cols, sal.Channels, img.Rows, img.Cols)
	}
	obj, bg := diskMeans(sal, diskCY, diskCX, diskR)
	t.Logf("fine grained: object=%.2f background=%.2f", obj, bg)
	if obj <= bg+20 {
		t.Errorf("object mean saliency %.2f not sufficiently above background %.2f", obj, bg)
	}
}

func TestComputeBinaryMapIsolatesObject(t *testing.T) {
	img := diskScene(sceneSize, bgValue, fgValue, diskCY, diskCX, diskR)
	sal := saliency.NewStaticSaliencyFineGrained().ComputeSaliency(img)
	mask := saliency.ComputeBinaryMap(sal)

	if mask.Channels != 1 {
		t.Fatalf("binary map channels = %d, want 1", mask.Channels)
	}
	for i, v := range mask.Data {
		if v != 0 && v != 255 {
			t.Fatalf("binary map sample %d = %d, want 0 or 255", i, v)
		}
	}
	if mask.At(diskCY, diskCX, 0) != 255 {
		t.Errorf("object centre (%d,%d) = %d, want 255", diskCY, diskCX, mask.At(diskCY, diskCX, 0))
	}
	if mask.At(0, 0, 0) != 0 {
		t.Errorf("background corner (0,0) = %d, want 0", mask.At(0, 0, 0))
	}

	// Foreground should be concentrated on the object.
	var fgInObj, fgTotal int
	for y := 0; y < mask.Rows; y++ {
		for x := 0; x < mask.Cols; x++ {
			if mask.At(y, x, 0) == 255 {
				fgTotal++
				dy, dx := y-diskCY, x-diskCX
				if dy*dy+dx*dx <= (diskR+2)*(diskR+2) {
					fgInObj++
				}
			}
		}
	}
	if fgTotal == 0 {
		t.Fatal("binary map has no foreground")
	}
	if frac := float64(fgInObj) / float64(fgTotal); frac < 0.5 {
		t.Errorf("only %.0f%% of foreground lies on the object, want >= 50%%", frac*100)
	}
}

func TestMotionSaliencyFlagsMovingBlob(t *testing.T) {
	const size = 32
	const bg = 50
	const fg = 210
	const blobSide = 8

	det := saliency.NewMotionSaliencyBinWangApr2014(size, size)

	// Learn a stationary background from two identical empty frames.
	empty := squareScene(size, bg, bg, 0, 0, 1)
	det.ComputeSaliency(empty)
	det.ComputeSaliency(empty.Clone())

	// A blob appears; its pixels must be flagged as moving.
	by, bx := 12, 12
	frame := squareScene(size, bg, fg, by, bx, blobSide)
	motion := det.ComputeSaliency(frame)

	if motion.Rows != size || motion.Cols != size || motion.Channels != 1 {
		t.Fatalf("motion map shape = %dx%dx%d, want %dx%dx1", motion.Rows, motion.Cols, motion.Channels, size, size)
	}
	for _, v := range motion.Data {
		if v != 0 && v != 255 {
			t.Fatalf("motion map is not binary, saw %d", v)
		}
	}
	cy, cx := by+blobSide/2, bx+blobSide/2
	if motion.At(cy, cx, 0) != 255 {
		t.Errorf("blob centre (%d,%d) = %d, want 255", cy, cx, motion.At(cy, cx, 0))
	}
	if motion.At(0, 0, 0) != 0 {
		t.Errorf("background corner (0,0) = %d, want 0", motion.At(0, 0, 0))
	}

	var flaggedInBlob, flaggedTotal int
	for y := 0; y < size; y++ {
		for x := 0; x < size; x++ {
			if motion.At(y, x, 0) == 255 {
				flaggedTotal++
				if y >= by && y < by+blobSide && x >= bx && x < bx+blobSide {
					flaggedInBlob++
				}
			}
		}
	}
	if flaggedTotal == 0 {
		t.Fatal("no moving pixels flagged")
	}
	if float64(flaggedInBlob)/float64(flaggedTotal) < 0.9 {
		t.Errorf("flagged pixels not concentrated on blob: %d in blob of %d total", flaggedInBlob, flaggedTotal)
	}
}

func TestMotionSaliencyFirstFrameIsBlank(t *testing.T) {
	const size = 16
	det := saliency.NewMotionSaliencyBinWangApr2014(size, size)
	first := det.ComputeSaliency(squareScene(size, 40, 200, 4, 4, 6))
	for _, v := range first.Data {
		if v != 0 {
			t.Fatalf("first frame should be all zero, saw %d", v)
		}
	}
}

func TestObjectnessProposesObjectWindow(t *testing.T) {
	img := diskScene(sceneSize, bgValue, fgValue, diskCY, diskCX, diskR)
	boxes := saliency.NewObjectnessBING().ComputeObjectness(img)
	if len(boxes) == 0 {
		t.Fatal("expected at least one objectness proposal")
	}
	for i := 1; i < len(boxes); i++ {
		if boxes[i].Score > boxes[i-1].Score {
			t.Fatalf("boxes not sorted by score at %d: %.3f > %.3f", i, boxes[i].Score, boxes[i-1].Score)
		}
	}
	found := false
	limit := 8
	if limit > len(boxes) {
		limit = len(boxes)
	}
	for _, b := range boxes[:limit] {
		if diskCX >= b.X && diskCX < b.X+b.W && diskCY >= b.Y && diskCY < b.Y+b.H {
			found = true
			break
		}
	}
	if !found {
		t.Errorf("no top objectness proposal overlaps the object centre (%d,%d)", diskCX, diskCY)
	}
}

func TestSaliencyImplementsInterface(t *testing.T) {
	var _ saliency.StaticSaliency = saliency.NewStaticSaliencySpectralResidual()
	var _ saliency.StaticSaliency = saliency.NewStaticSaliencyFineGrained()
}
