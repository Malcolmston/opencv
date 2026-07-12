package face_test

import (
	"math"
	"math/rand"
	"sort"
	"testing"

	cv "github.com/malcolmston/opencv"
	"github.com/malcolmston/opencv/face"
)

// --- Threshold and PredictCollect ------------------------------------------

func TestPredictCollectSortedAndConsistent(t *testing.T) {
	trainImgs, trainLbl := buildDataset(4, 6, faceSize, 12, 30)
	r := face.NewEigenFaceRecognizer(0)
	r.Train(trainImgs, trainLbl)

	q := makeFace(2, faceSize, 0, nil)
	results := r.PredictCollect(q)
	if len(results) != len(trainImgs) {
		t.Fatalf("PredictCollect returned %d results, want %d", len(results), len(trainImgs))
	}
	if !sort.SliceIsSorted(results, func(i, j int) bool { return results[i].Distance < results[j].Distance }) {
		t.Fatal("PredictCollect results are not sorted by ascending distance")
	}
	predLabel, predDist := r.Predict(q)
	if results[0].Label != predLabel || math.Abs(results[0].Distance-predDist) > 1e-9 {
		t.Fatalf("best collected (%d,%.4f) disagrees with Predict (%d,%.4f)",
			results[0].Label, results[0].Distance, predLabel, predDist)
	}
}

func TestThresholdRejectsUnknown(t *testing.T) {
	trainImgs, trainLbl := buildDataset(4, 6, faceSize, 12, 31)
	r := face.NewLBPHFaceRecognizerWithParams(4, 4, false)
	r.Train(trainImgs, trainLbl)

	q := makeFace(1, faceSize, 0, nil)

	// No threshold: a real label.
	if lbl, _ := r.PredictThreshold(q); lbl == face.Unknown {
		t.Fatal("expected a known label with no threshold set")
	}
	// Impossibly tight threshold: everything rejected.
	r.SetThreshold(1e-9)
	if lbl, _ := r.PredictThreshold(q); lbl != face.Unknown {
		t.Fatalf("expected Unknown under a tiny threshold, got %d", lbl)
	}
	if got := len(r.PredictCollect(q)); got != 0 {
		t.Fatalf("expected no candidates under a tiny threshold, got %d", got)
	}
	// Generous threshold: known again.
	r.SetThreshold(1e12)
	if lbl, _ := r.PredictThreshold(q); lbl == face.Unknown {
		t.Fatal("expected a known label under a generous threshold")
	}
	// Clearing restores the default.
	r.SetThreshold(0)
	if r.GetThreshold() != 0 {
		t.Fatalf("threshold not cleared: %v", r.GetThreshold())
	}
}

// --- Eigenface / Fisherface accessors --------------------------------------

func TestEigenVectorAccessors(t *testing.T) {
	trainImgs, trainLbl := buildDataset(4, 6, faceSize, 12, 32)
	r := face.NewEigenFaceRecognizer(0)
	r.Train(trainImgs, trainLbl)

	vecs := r.EigenVectors()
	if len(vecs) != r.NumComponents() {
		t.Fatalf("EigenVectors count %d != NumComponents %d", len(vecs), r.NumComponents())
	}
	rows, cols := r.Dims()
	if rows != faceSize || cols != faceSize {
		t.Fatalf("Dims = %dx%d, want %dx%d", rows, cols, faceSize, faceSize)
	}
	for _, v := range vecs {
		if len(v) != rows*cols {
			t.Fatalf("eigenvector length %d, want %d", len(v), rows*cols)
		}
	}
	mean := r.MeanFace()
	if mean.Rows != rows || mean.Cols != cols || mean.Channels != 1 {
		t.Fatalf("MeanFace geometry wrong: %dx%dx%d", mean.Rows, mean.Cols, mean.Channels)
	}
	ef := r.EigenFaceImage(0)
	if ef.Rows != rows || ef.Cols != cols {
		t.Fatalf("EigenFaceImage geometry wrong: %dx%d", ef.Rows, ef.Cols)
	}

	// Reconstructing an image from its own projection should be close to it.
	orig := trainImgs[0]
	rec := r.ReconstructImage(r.Project(orig))
	var mse float64
	for i := range orig.Data {
		d := float64(orig.Data[i]) - float64(rec.Data[i])
		mse += d * d
	}
	mse /= float64(len(orig.Data))
	if mse > 25 { // within ~5 grey levels RMS
		t.Fatalf("reconstruction MSE too high: %.2f", mse)
	}
}

func TestFisherAxes(t *testing.T) {
	trainImgs, trainLbl := buildDataset(4, 6, faceSize, 12, 33)
	r := face.NewFisherFaceRecognizer(0)
	r.Train(trainImgs, trainLbl)
	axes := r.DiscriminantAxes()
	if len(axes) != r.NumComponents() {
		t.Fatalf("DiscriminantAxes count %d != NumComponents %d", len(axes), r.NumComponents())
	}
	if len(axes[0]) != faceSize*faceSize {
		t.Fatalf("discriminant axis length %d, want %d", len(axes[0]), faceSize*faceSize)
	}
	img := r.FisherFaceImage(0)
	if img.Rows != faceSize || img.Cols != faceSize {
		t.Fatalf("FisherFaceImage geometry wrong: %dx%d", img.Rows, img.Cols)
	}
}

// --- Extended LBP ----------------------------------------------------------

func TestLBPCircularSolidAndSize(t *testing.T) {
	solid := cv.NewMat(7, 7, 1)
	solid.SetTo(120)
	out := face.LBPCircular(solid, 1, 8)
	if out.Rows != 5 || out.Cols != 5 {
		t.Fatalf("LBPCircular size = %dx%d, want 5x5", out.Rows, out.Cols)
	}
	// On a flat field every neighbour equals the centre (>=), so all bits set.
	for _, v := range out.Data {
		if v != 255 {
			t.Fatalf("flat-field circular code = %d, want 255", v)
		}
	}
	// A larger radius trims a wider border.
	out2 := face.LBPCircular(solid, 2, 8)
	if out2.Rows != 3 || out2.Cols != 3 {
		t.Fatalf("radius-2 size = %dx%d, want 3x3", out2.Rows, out2.Cols)
	}
}

// TestRIU2RotationInvariance checks the rotation-invariant uniform operator gives
// the same label to a pattern and its 90° rotation, where the plain uniform
// operator does not.
func TestRIU2RotationInvariance(t *testing.T) {
	right := cv.NewMat(3, 3, 1)
	copy(right.Data, []uint8{
		0, 0, 0,
		0, 100, 255,
		0, 0, 0,
	})
	down := cv.NewMat(3, 3, 1)
	copy(down.Data, []uint8{
		0, 0, 0,
		0, 100, 0,
		0, 255, 0,
	})
	rr := face.LBPUniformRotInvariant(right).Data[0]
	rd := face.LBPUniformRotInvariant(down).Data[0]
	if rr != rd {
		t.Fatalf("riu2 not rotation invariant: %d vs %d", rr, rd)
	}
	if rr > 9 {
		t.Fatalf("riu2 label %d out of range", rr)
	}
	// The plain uniform labels should differ for these two rotations.
	ur := face.LBPUniform(right).Data[0]
	ud := face.LBPUniform(down).Data[0]
	if ur == ud {
		t.Fatal("expected plain uniform labels to differ under rotation")
	}
}

// --- Haar detector ----------------------------------------------------------

// plantFace draws a face-like template (dark eye band and central column over a
// bright field) filling the square window at (x0,y0,win) of a mid-grey image.
func plantFace(img *cv.Mat, x0, y0, win int) {
	for y := y0; y < y0+win; y++ {
		for x := x0; x < x0+win; x++ {
			ry := y - y0
			rx := x - x0
			v := uint8(210) // bright cheeks/forehead
			if ry >= win/6 && ry < win/6+win/4 {
				v = 40 // dark eye band
			}
			if rx >= win/3 && rx < 2*win/3 && ry >= win/6 && ry < win/6+win/2 {
				v = 40 // dark central (eye/nose) column
			}
			img.Set(y, x, 0, v)
		}
	}
}

func TestGetFacesHAAR(t *testing.T) {
	img := cv.NewMat(72, 72, 1)
	img.SetTo(130)
	plantFace(img, 20, 18, 32)

	p := face.DefaultHaarParams()
	p.MinSize = 24
	p.MaxSize = 48
	rects := face.GetFacesHAAR(img, &p)
	if len(rects) == 0 {
		t.Fatal("expected at least one detection on a planted face")
	}
	// The top detection should overlap the planted region.
	planted := cv.Rect{X: 20, Y: 18, Width: 32, Height: 32}
	if iou := rectIoUTest(rects[0], planted); iou < 0.2 {
		t.Fatalf("top detection %v overlaps planted %v poorly (IoU %.2f)", rects[0], planted, iou)
	}

	// A flat image has no contrast and must yield nothing.
	flat := cv.NewMat(72, 72, 1)
	flat.SetTo(130)
	if got := face.GetFacesHAAR(flat, &p); len(got) != 0 {
		t.Fatalf("expected no detections on a flat image, got %d", len(got))
	}
}

func rectIoUTest(a, b cv.Rect) float64 {
	x0 := maxi(a.X, b.X)
	y0 := maxi(a.Y, b.Y)
	x1 := mini(a.X+a.Width, b.X+b.Width)
	y1 := mini(a.Y+a.Height, b.Y+b.Height)
	iw, ih := x1-x0, y1-y0
	if iw <= 0 || ih <= 0 {
		return 0
	}
	inter := float64(iw * ih)
	return inter / (float64(a.Width*a.Height+b.Width*b.Height) - inter)
}

func maxi(a, b int) int {
	if a > b {
		return a
	}
	return b
}
func mini(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// --- BIF descriptor ---------------------------------------------------------

func TestBIFDeterministicAndDiscriminative(t *testing.T) {
	b := face.NewBIF(3, 4, 2)

	a := makeFace(2, 24, 0, nil) // checkerboard
	c := makeFace(0, 24, 0, nil) // horizontal gradient

	fa1 := b.Compute(a)
	fa2 := b.Compute(a)
	if len(fa1) != b.FeatureLength() {
		t.Fatalf("BIF length %d != FeatureLength %d", len(fa1), b.FeatureLength())
	}
	for i := range fa1 {
		if fa1[i] != fa2[i] {
			t.Fatal("BIF is not deterministic")
		}
	}
	fc := b.Compute(c)
	var dist float64
	for i := range fa1 {
		d := fa1[i] - fc[i]
		dist += d * d
	}
	if math.Sqrt(dist) < 1e-3 {
		t.Fatal("BIF failed to distinguish two different textures")
	}
}

// TestBIFRecognition uses nearest-neighbour BIF descriptors as a simple face
// classifier over the synthetic classes, confirming the descriptor is genuinely
// discriminative rather than merely well-formed.
func TestBIFRecognition(t *testing.T) {
	b := face.NewBIF(3, 4, 3)
	trainImgs, trainLbl := buildDataset(4, 5, 24, 8, 40)
	testImgs, testLbl := buildDataset(4, 3, 24, 8, 41)

	train := make([][]float64, len(trainImgs))
	for i, im := range trainImgs {
		train[i] = b.Compute(im)
	}
	correct := 0
	for i, im := range testImgs {
		q := b.Compute(im)
		best, bestD := -1, math.Inf(1)
		for j, tr := range train {
			var d float64
			for k := range tr {
				e := tr[k] - q[k]
				d += e * e
			}
			if d < bestD {
				bestD, best = d, trainLbl[j]
			}
		}
		if best == testLbl[i] {
			correct++
		}
	}
	acc := float64(correct) / float64(len(testImgs))
	if acc < 0.9 {
		t.Fatalf("BIF nearest-neighbour accuracy = %.2f, want >= 0.9", acc)
	}
}

// --- MACE correlation filter ------------------------------------------------

func TestMACEAuthenticVsImpostor(t *testing.T) {
	rng := rand.New(rand.NewSource(50))
	var authentic []*cv.Mat
	for i := 0; i < 8; i++ {
		authentic = append(authentic, makeFace(2, faceSize, 6, rng)) // class 2 (textured)
	}
	m := face.NewMACE(faceSize)
	m.Train(authentic)

	psrAuth := m.PSR(makeFace(2, faceSize, 0, nil)) // same class
	psrImp := m.PSR(makeFace(0, faceSize, 0, nil))  // different class

	if !(psrAuth > psrImp) {
		t.Fatalf("MACE did not separate classes: authentic PSR %.2f <= impostor PSR %.2f", psrAuth, psrImp)
	}
	if psrAuth < 10 {
		t.Fatalf("authentic PSR unexpectedly low: %.2f", psrAuth)
	}

	m.SetThreshold((psrAuth + psrImp) / 2)
	if !m.Same(makeFace(2, faceSize, 0, nil)) {
		t.Fatal("MACE.Same rejected an authentic query")
	}
	if m.Same(makeFace(0, faceSize, 0, nil)) {
		t.Fatal("MACE.Same accepted an impostor query")
	}
}

// --- Facemark landmark localiser --------------------------------------------

// blobFace renders a mid-grey face with a bright Gaussian blob at each landmark;
// the returned points are the (integer) blob centres in image coordinates.
func blobFace(rect cv.Rect, base []cv.Point, jitter float64, rng *rand.Rand) (*cv.Mat, []cv.Point) {
	const pad = 8
	img := cv.NewMat(rect.Y+rect.Height+pad, rect.X+rect.Width+pad, 1)
	img.SetTo(90)
	pts := make([]cv.Point, len(base))
	for i, p := range base {
		dx, dy := 0.0, 0.0
		if rng != nil {
			dx = (rng.Float64()*2 - 1) * jitter * float64(rect.Width)
			dy = (rng.Float64()*2 - 1) * jitter * float64(rect.Height)
		}
		cx := float64(p.X) + dx
		cy := float64(p.Y) + dy
		pts[i] = cv.Point{X: int(math.Round(cx)), Y: int(math.Round(cy))}
		// Paint a Gaussian blob.
		for y := 0; y < img.Rows; y++ {
			for x := 0; x < img.Cols; x++ {
				d2 := (float64(x)-cx)*(float64(x)-cx) + (float64(y)-cy)*(float64(y)-cy)
				add := 140 * math.Exp(-d2/(2*2.4*2.4))
				if add < 1 {
					continue
				}
				nv := float64(img.Data[y*img.Cols+x]) + add
				img.Data[y*img.Cols+x] = clampByteTest(nv)
			}
		}
	}
	return img, pts
}

func clampByteTest(v float64) uint8 {
	if v < 0 {
		return 0
	}
	if v > 255 {
		return 255
	}
	return uint8(v + 0.5)
}

func shapeError(a, b []cv.Point) float64 {
	var s float64
	for i := range a {
		dx := float64(a[i].X - b[i].X)
		dy := float64(a[i].Y - b[i].Y)
		s += math.Sqrt(dx*dx + dy*dy)
	}
	return s / float64(len(a))
}

func TestFacemarkFitBeatsMeanShape(t *testing.T) {
	rect := cv.Rect{X: 10, Y: 10, Width: 40, Height: 40}
	base := []cv.Point{
		{X: rect.X + 12, Y: rect.Y + 14}, // left eye
		{X: rect.X + 28, Y: rect.Y + 14}, // right eye
		{X: rect.X + 20, Y: rect.Y + 27}, // nose
	}
	const jitter = 0.1

	rng := rand.New(rand.NewSource(60))
	var imgs []*cv.Mat
	var rects []cv.Rect
	var shapes [][]cv.Point
	for i := 0; i < 40; i++ {
		im, pts := blobFace(rect, base, jitter, rng)
		imgs = append(imgs, im)
		rects = append(rects, rect)
		shapes = append(shapes, pts)
	}

	fm := face.NewFacemarkLBF(3)
	fm.Train(imgs, rects, shapes)
	if fm.NumLandmarks() != 3 {
		t.Fatalf("NumLandmarks = %d, want 3", fm.NumLandmarks())
	}

	testRng := rand.New(rand.NewSource(61))
	var fitErr, baseErr float64
	const nTest = 20
	for i := 0; i < nTest; i++ {
		im, truth := blobFace(rect, base, jitter, testRng)
		fitErr += shapeError(fm.Fit(im, rect), truth)
		baseErr += shapeError(fm.MeanShapeAt(rect), truth)
	}
	fitErr /= nTest
	baseErr /= nTest
	if fitErr >= baseErr {
		t.Fatalf("fit did not improve on mean shape: fit %.3f vs mean %.3f", fitErr, baseErr)
	}
	if fitErr > 0.6*baseErr {
		t.Fatalf("fit improvement too small: fit %.3f vs mean %.3f", fitErr, baseErr)
	}
}
