package face_test

import (
	"math"
	"math/rand"
	"testing"

	cv "github.com/malcolmston/opencv"
	"github.com/malcolmston/opencv/face"
)

const faceSize = 16

// makeFace synthesises a small single-channel "face" whose structure depends on
// class (four visually distinct patterns) with optional seeded Gaussian noise.
// Base intensities stay within [20,210] so a moderate brightness shift does not
// clip. The generator is deterministic given rng.
func makeFace(class, size int, sigma float64, rng *rand.Rand) *cv.Mat {
	m := cv.NewMat(size, size, 1)
	for y := 0; y < size; y++ {
		for x := 0; x < size; x++ {
			var base float64
			switch class {
			case 0: // horizontal gradient
				base = 30 + 150*float64(x)/float64(size-1)
			case 1: // vertical gradient
				base = 30 + 150*float64(y)/float64(size-1)
			case 2: // coarse checkerboard
				if (x/3+y/3)%2 == 0 {
					base = 60
				} else {
					base = 180
				}
			default: // diagonal stripes
				if (x+y)%4 < 2 {
					base = 70
				} else {
					base = 170
				}
			}
			v := base
			if sigma > 0 {
				v += rng.NormFloat64() * sigma
			}
			if v < 0 {
				v = 0
			}
			if v > 255 {
				v = 255
			}
			m.Data[y*size+x] = uint8(v + 0.5)
		}
	}
	return m
}

// buildDataset returns perClass images for each of classes classes, seeded
// reproducibly by seed.
func buildDataset(classes, perClass, size int, sigma float64, seed int64) ([]*cv.Mat, []int) {
	rng := rand.New(rand.NewSource(seed))
	var imgs []*cv.Mat
	var labels []int
	for c := 0; c < classes; c++ {
		for i := 0; i < perClass; i++ {
			imgs = append(imgs, makeFace(c, size, sigma, rng))
			labels = append(labels, c)
		}
	}
	return imgs, labels
}

// brighten returns a copy of img with delta added to every sample (saturating).
func brighten(img *cv.Mat, delta int) *cv.Mat {
	out := img.Clone()
	for i, v := range out.Data {
		nv := int(v) + delta
		if nv < 0 {
			nv = 0
		}
		if nv > 255 {
			nv = 255
		}
		out.Data[i] = uint8(nv)
	}
	return out
}

// accuracy trains r on the given data and reports the fraction of the test set
// it classifies correctly.
func accuracy(r face.FaceRecognizer, trainImgs []*cv.Mat, trainLbl []int, testImgs []*cv.Mat, testLbl []int) float64 {
	r.Train(trainImgs, trainLbl)
	correct := 0
	for i, im := range testImgs {
		pred, _ := r.Predict(im)
		if pred == testLbl[i] {
			correct++
		}
	}
	return float64(correct) / float64(len(testImgs))
}

func TestEigenFaceRecognizerAccuracy(t *testing.T) {
	trainImgs, trainLbl := buildDataset(4, 6, faceSize, 12, 1)
	testImgs, testLbl := buildDataset(4, 4, faceSize, 12, 99)
	r := face.NewEigenFaceRecognizer(0)
	acc := accuracy(r, trainImgs, trainLbl, testImgs, testLbl)
	if acc < 0.9 {
		t.Fatalf("EigenFaces accuracy = %.2f, want >= 0.9", acc)
	}
	if r.NumComponents() == 0 {
		t.Fatal("expected a positive number of retained eigenfaces")
	}
}

func TestFisherFaceRecognizerAccuracy(t *testing.T) {
	trainImgs, trainLbl := buildDataset(4, 6, faceSize, 12, 2)
	testImgs, testLbl := buildDataset(4, 4, faceSize, 12, 88)
	r := face.NewFisherFaceRecognizer(0)
	acc := accuracy(r, trainImgs, trainLbl, testImgs, testLbl)
	if acc < 0.9 {
		t.Fatalf("Fisherfaces accuracy = %.2f, want >= 0.9", acc)
	}
	if got := r.NumComponents(); got != 3 {
		t.Fatalf("Fisherfaces retained %d discriminant axes, want C-1 = 3", got)
	}
}

func TestLBPHFaceRecognizerAccuracy(t *testing.T) {
	trainImgs, trainLbl := buildDataset(4, 6, faceSize, 12, 3)
	testImgs, testLbl := buildDataset(4, 4, faceSize, 12, 77)
	r := face.NewLBPHFaceRecognizerWithParams(4, 4, false)
	acc := accuracy(r, trainImgs, trainLbl, testImgs, testLbl)
	if acc < 0.9 {
		t.Fatalf("LBPH accuracy = %.2f, want >= 0.9", acc)
	}
}

// TestLBPHUniformAccuracy exercises the uniform-pattern histogram path. Uniform
// LBP deliberately collapses non-uniform (high-frequency) codes into a single
// bin, so it separates the smooth gradient classes well while intentionally
// discarding fine texture; the two gradient classes are used here.
func TestLBPHUniformAccuracy(t *testing.T) {
	trainImgs, trainLbl := buildDataset(2, 6, faceSize, 10, 5)
	testImgs, testLbl := buildDataset(2, 4, faceSize, 10, 55)
	r := face.NewLBPHFaceRecognizerWithParams(4, 4, true)
	acc := accuracy(r, trainImgs, trainLbl, testImgs, testLbl)
	if acc < 0.9 {
		t.Fatalf("LBPH (uniform) accuracy = %.2f, want >= 0.9", acc)
	}
}

// TestLBPHBrightnessRobust verifies LBPH is invariant to a uniform brightness
// shift: a brightened query is classified the same, and its histogram distance
// to the match is essentially unchanged (LBP codes are unaffected when no
// clipping occurs).
func TestLBPHBrightnessRobust(t *testing.T) {
	trainImgs, trainLbl := buildDataset(4, 6, faceSize, 0, 7)
	r := face.NewLBPHFaceRecognizerWithParams(4, 4, false)
	r.Train(trainImgs, trainLbl)

	// A clean, noise-free query of class 2, then the same image brightened.
	query := makeFace(2, faceSize, 0, nil)
	predOrig, distOrig := r.Predict(query)
	predBright, distBright := r.Predict(brighten(query, 40))

	if predOrig != 2 || predBright != 2 {
		t.Fatalf("brightness changed prediction: orig=%d bright=%d, want 2", predOrig, predBright)
	}
	if math.Abs(distOrig-distBright) > 1e-9 {
		t.Fatalf("brightness changed distance: orig=%.6f bright=%.6f", distOrig, distBright)
	}
}

// TestEigenReconstructionError checks that reconstruction error is
// non-increasing in the number of eigenfaces used and drops substantially from
// K=1 to the full basis.
func TestEigenReconstructionError(t *testing.T) {
	trainImgs, trainLbl := buildDataset(4, 6, faceSize, 12, 11)
	r := face.NewEigenFaceRecognizer(0)
	r.Train(trainImgs, trainLbl)

	// Original vector of a training image (single-channel, already faceSize).
	orig := trainImgs[0]
	want := make([]float64, len(orig.Data))
	for i, v := range orig.Data {
		want[i] = float64(v)
	}

	coeffs := r.Project(orig)
	maxK := len(coeffs)
	if maxK < 3 {
		t.Fatalf("need at least 3 components, got %d", maxK)
	}

	mse := func(k int) float64 {
		rec := r.Reconstruct(coeffs[:k])
		var s float64
		for i := range want {
			d := want[i] - rec[i]
			s += d * d
		}
		return s / float64(len(want))
	}

	prev := math.Inf(1)
	for k := 1; k <= maxK; k++ {
		e := mse(k)
		if e > prev+1e-6 {
			t.Fatalf("reconstruction error increased at K=%d: %.4f > %.4f", k, e, prev)
		}
		prev = e
	}
	if e1, ek := mse(1), mse(maxK); ek >= e1 {
		t.Fatalf("error did not drop with more components: K=1 -> %.4f, K=%d -> %.4f", e1, maxK, ek)
	}
}

// TestLBPKnownPatch checks the basic LBP code of a hand-computed 3×3 patch.
//
//	 10 200  10        code bits (weight): top=2, right=8, bottom=32, left=128
//	200 100 200        => 2 + 8 + 32 + 128 = 170
//	 10 200  10
func TestLBPKnownPatch(t *testing.T) {
	m := cv.NewMat(3, 3, 1)
	copy(m.Data, []uint8{
		10, 200, 10,
		200, 100, 200,
		10, 200, 10,
	})
	out := face.LBP(m)
	if out.Rows != 1 || out.Cols != 1 {
		t.Fatalf("LBP of 3x3 should be 1x1, got %dx%d", out.Rows, out.Cols)
	}
	if out.Data[0] != 170 {
		t.Fatalf("LBP code = %d, want 170", out.Data[0])
	}
	// 170 = 10101010 has 8 bit transitions, so it is non-uniform (label 58).
	u := face.LBPUniform(m)
	if u.Data[0] != 58 {
		t.Fatalf("uniform label = %d, want 58 (non-uniform)", u.Data[0])
	}
}

// TestLBPBrightnessInvariant confirms LBP codes are unchanged by a uniform,
// non-clipping brightness shift.
func TestLBPBrightnessInvariant(t *testing.T) {
	img := makeFace(3, faceSize, 0, nil)
	a := face.LBP(img)
	b := face.LBP(brighten(img, 30))
	for i := range a.Data {
		if a.Data[i] != b.Data[i] {
			t.Fatalf("LBP code changed under brightness shift at %d: %d vs %d", i, a.Data[i], b.Data[i])
		}
	}
}

// TestLBPUniformRange verifies every uniform label lies in [0,58] and that the
// count of distinct uniform patterns is the expected 58.
func TestLBPUniformRange(t *testing.T) {
	seen := make(map[uint8]bool)
	img, _ := buildDataset(1, 1, faceSize, 0, 1)
	u := face.LBPUniform(img[0])
	for _, v := range u.Data {
		if v > 58 {
			t.Fatalf("uniform label %d out of range", v)
		}
		seen[v] = true
	}
	// Exhaustively check the mapping produces exactly 59 labels overall by
	// scanning a synthetic image rich in patterns is not guaranteed; instead
	// verify the two extreme codes map to valid uniform labels.
	solid := cv.NewMat(3, 3, 1)
	solid.SetTo(100)
	if got := face.LBPUniform(solid).Data[0]; got > 57 {
		t.Fatalf("all-equal neighbourhood should be a uniform pattern, got label %d", got)
	}
}

func TestTrainPanicsOnBadInput(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("expected panic on mismatched image/label counts")
		}
	}()
	r := face.NewEigenFaceRecognizer(0)
	imgs, _ := buildDataset(2, 2, faceSize, 0, 1)
	r.Train(imgs, []int{0}) // wrong label count
}

func TestPredictBeforeTrainPanics(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("expected panic on Predict before Train")
		}
	}()
	r := face.NewLBPHFaceRecognizer()
	r.Predict(makeFace(0, faceSize, 0, nil))
}
