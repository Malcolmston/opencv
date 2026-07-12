package face_test

import (
	"bytes"
	"testing"

	cv "github.com/malcolmston/opencv"
	"github.com/malcolmston/opencv/face"
)

// predictAll returns the predictions of r over imgs, for round-trip comparison.
func predictAll(r face.FaceRecognizer, imgs []*cv.Mat) []int {
	out := make([]int, len(imgs))
	for i, im := range imgs {
		out[i], _ = r.Predict(im)
	}
	return out
}

func TestEigenSaveLoadRoundTrip(t *testing.T) {
	trainImgs, trainLbl := buildDataset(4, 6, faceSize, 12, 20)
	testImgs, _ := buildDataset(4, 4, faceSize, 12, 21)

	orig := face.NewEigenFaceRecognizer(0)
	orig.Train(trainImgs, trainLbl)
	orig.SetThreshold(1234.5)
	want := predictAll(orig, testImgs)

	var buf bytes.Buffer
	if err := orig.Save(&buf); err != nil {
		t.Fatalf("Save: %v", err)
	}

	loaded := face.NewEigenFaceRecognizer(0)
	if err := loaded.Load(&buf); err != nil {
		t.Fatalf("Load: %v", err)
	}
	got := predictAll(loaded, testImgs)
	for i := range want {
		if want[i] != got[i] {
			t.Fatalf("prediction %d differs after round trip: %d vs %d", i, want[i], got[i])
		}
	}
	if loaded.GetThreshold() != 1234.5 {
		t.Fatalf("threshold not persisted: got %v", loaded.GetThreshold())
	}
	// Full float projection should be identical too.
	a := orig.Project(testImgs[0])
	b := loaded.Project(testImgs[0])
	for i := range a {
		if a[i] != b[i] {
			t.Fatalf("projection coeff %d differs: %v vs %v", i, a[i], b[i])
		}
	}
}

func TestFisherSaveLoadRoundTrip(t *testing.T) {
	trainImgs, trainLbl := buildDataset(4, 6, faceSize, 12, 22)
	testImgs, testLbl := buildDataset(4, 4, faceSize, 12, 23)

	orig := face.NewFisherFaceRecognizer(0)
	orig.Train(trainImgs, trainLbl)
	want := predictAll(orig, testImgs)

	var buf bytes.Buffer
	if err := orig.Save(&buf); err != nil {
		t.Fatalf("Save: %v", err)
	}
	loaded := face.NewFisherFaceRecognizer(0)
	if err := loaded.Load(&buf); err != nil {
		t.Fatalf("Load: %v", err)
	}
	got := predictAll(loaded, testImgs)
	correct := 0
	for i := range want {
		if want[i] != got[i] {
			t.Fatalf("prediction %d differs after round trip: %d vs %d", i, want[i], got[i])
		}
		if got[i] == testLbl[i] {
			correct++
		}
	}
	if loaded.NumComponents() != orig.NumComponents() {
		t.Fatalf("component count differs: %d vs %d", loaded.NumComponents(), orig.NumComponents())
	}
}

func TestLBPHSaveLoadRoundTrip(t *testing.T) {
	trainImgs, trainLbl := buildDataset(4, 6, faceSize, 12, 24)
	testImgs, _ := buildDataset(4, 4, faceSize, 12, 25)

	orig := face.NewLBPHFaceRecognizerWithParams(4, 4, true)
	orig.Train(trainImgs, trainLbl)
	want := predictAll(orig, testImgs)

	var buf bytes.Buffer
	if err := orig.Save(&buf); err != nil {
		t.Fatalf("Save: %v", err)
	}
	loaded := face.NewLBPHFaceRecognizer() // different params on purpose
	if err := loaded.Load(&buf); err != nil {
		t.Fatalf("Load: %v", err)
	}
	if loaded.GridX != 4 || loaded.GridY != 4 || !loaded.Uniform {
		t.Fatalf("grid/uniform not restored: %d %d %v", loaded.GridX, loaded.GridY, loaded.Uniform)
	}
	got := predictAll(loaded, testImgs)
	for i := range want {
		if want[i] != got[i] {
			t.Fatalf("prediction %d differs after round trip: %d vs %d", i, want[i], got[i])
		}
	}
}

func TestLoadVersionMismatch(t *testing.T) {
	// A stream that is valid gob but not a recognizer snapshot decodes into
	// zero-value fields, so Version != snapshotVersion and Load reports it.
	var buf bytes.Buffer
	buf.WriteString("not a valid model stream")
	r := face.NewEigenFaceRecognizer(0)
	if err := r.Load(&buf); err == nil {
		t.Fatal("expected an error loading a corrupt stream")
	}
}
