package face_test

import (
	"fmt"

	cv "github.com/malcolmston/opencv"
	"github.com/malcolmston/opencv/face"
)

// gradient builds a deterministic 12×12 single-channel test image: a horizontal
// ramp for dir 0 and a vertical ramp for dir 1.
func gradient(dir int) *cv.Mat {
	const n = 12
	m := cv.NewMat(n, n, 1)
	for y := 0; y < n; y++ {
		for x := 0; x < n; x++ {
			var v float64
			if dir == 0 {
				v = 20 + 200*float64(x)/float64(n-1)
			} else {
				v = 20 + 200*float64(y)/float64(n-1)
			}
			m.Data[y*n+x] = uint8(v)
		}
	}
	return m
}

// ExampleLBP computes the Local Binary Pattern code of a hand-built 3×3 patch.
func ExampleLBP() {
	m := cv.NewMat(3, 3, 1)
	copy(m.Data, []uint8{
		10, 200, 10,
		200, 100, 200,
		10, 200, 10,
	})
	out := face.LBP(m)
	fmt.Println(out.Rows, out.Cols, out.Data[0])
	// Output: 1 1 170
}

// ExampleEigenFaceRecognizer trains an Eigenfaces model on two gradient classes
// and classifies a fresh horizontal gradient.
func ExampleEigenFaceRecognizer() {
	imgs := []*cv.Mat{gradient(0), gradient(0), gradient(1), gradient(1)}
	labels := []int{0, 0, 1, 1}

	r := face.NewEigenFaceRecognizer(0)
	r.Train(imgs, labels)

	label, _ := r.Predict(gradient(0))
	fmt.Println(label)
	// Output: 0
}

// ExampleLBPHFaceRecognizer trains an LBPH model on two gradient classes and
// classifies a fresh vertical gradient.
func ExampleLBPHFaceRecognizer() {
	imgs := []*cv.Mat{gradient(0), gradient(1)}
	labels := []int{0, 1}

	r := face.NewLBPHFaceRecognizerWithParams(2, 2, false)
	r.Train(imgs, labels)

	label, _ := r.Predict(gradient(1))
	fmt.Println(label)
	// Output: 1
}
