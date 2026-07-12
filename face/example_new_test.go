package face_test

import (
	"bytes"
	"fmt"

	cv "github.com/malcolmston/opencv"
	"github.com/malcolmston/opencv/face"
)

// ExampleEigenFaceRecognizer_Save trains a model, serialises it with gob and
// reloads it into a fresh recognizer that reproduces the original prediction.
func ExampleEigenFaceRecognizer_Save() {
	imgs := []*cv.Mat{gradient(0), gradient(0), gradient(1), gradient(1)}
	labels := []int{0, 0, 1, 1}

	orig := face.NewEigenFaceRecognizer(0)
	orig.Train(imgs, labels)

	var buf bytes.Buffer
	if err := orig.Save(&buf); err != nil {
		panic(err)
	}

	loaded := face.NewEigenFaceRecognizer(0)
	if err := loaded.Load(&buf); err != nil {
		panic(err)
	}
	label, _ := loaded.Predict(gradient(1))
	fmt.Println(label)
	// Output: 1
}

// ExampleLBPHFaceRecognizer_SetThreshold rejects a query that lies beyond the
// configured recognition distance, returning face.Unknown.
func ExampleLBPHFaceRecognizer_SetThreshold() {
	imgs := []*cv.Mat{gradient(0), gradient(1)}
	labels := []int{0, 1}

	r := face.NewLBPHFaceRecognizerWithParams(2, 2, false)
	r.Train(imgs, labels)
	r.SetThreshold(1e-9) // impossibly strict: nothing is close enough

	// A flat grey image is unlike either gradient, so it is rejected.
	flat := cv.NewMat(12, 12, 1)
	flat.SetTo(128)
	label, _ := r.PredictThreshold(flat)
	fmt.Println(label == face.Unknown)
	// Output: true
}

// ExampleLBPUniformRotInvariant labels a small patch with the rotation-invariant
// uniform operator, and shows a 90° rotation gets the same label.
func ExampleLBPUniformRotInvariant() {
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
	fmt.Println(face.LBPUniformRotInvariant(right).Data[0], face.LBPUniformRotInvariant(down).Data[0])
	// Output: 1 1
}

// ExampleGetFacesHAAR locates a planted face-like pattern with the integral-image
// detector.
func ExampleGetFacesHAAR() {
	img := cv.NewMat(72, 72, 1)
	img.SetTo(130)
	// A dark eye band over a bright field inside a 32-pixel window.
	for y := 24; y < 32; y++ {
		for x := 20; x < 52; x++ {
			img.Set(y, x, 0, 40)
		}
	}
	for y := 32; y < 50; y++ {
		for x := 20; x < 52; x++ {
			img.Set(y, x, 0, 210)
		}
	}
	p := face.DefaultHaarParams()
	p.MinSize = 24
	p.MaxSize = 48
	fmt.Println(len(face.GetFacesHAAR(img, &p)) > 0)
	// Output: true
}

// ExampleMACE synthesises a correlation filter and checks that an authentic
// image scores a far higher peak-to-sidelobe ratio than an impostor.
func ExampleMACE() {
	tile := func(v0, v1 uint8) *cv.Mat {
		m := cv.NewMat(16, 16, 1)
		for y := 0; y < 16; y++ {
			for x := 0; x < 16; x++ {
				if (x/2+y/2)%2 == 0 {
					m.Set(y, x, 0, v0)
				} else {
					m.Set(y, x, 0, v1)
				}
			}
		}
		return m
	}
	authentic := []*cv.Mat{tile(40, 200), tile(45, 195), tile(35, 205)}
	m := face.NewMACE(16)
	m.Train(authentic)

	gradientImg := cv.NewMat(16, 16, 1)
	for y := 0; y < 16; y++ {
		for x := 0; x < 16; x++ {
			gradientImg.Set(y, x, 0, uint8(16*x))
		}
	}
	fmt.Println(m.PSR(tile(40, 200)) > m.PSR(gradientImg))
	// Output: true
}
