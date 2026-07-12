package xobjdetect_test

import (
	"bytes"
	"fmt"
	"math/rand"

	cv "github.com/malcolmston/opencv"
	"github.com/malcolmston/opencv/xobjdetect"
)

// brightBlock builds a 24x24 patch of the toy target: a dark field with a
// bright central block.
func brightBlock(rng *rand.Rand) *cv.Mat {
	m := cv.NewMat(24, 24, 1)
	for y := 0; y < 24; y++ {
		for x := 0; x < 24; x++ {
			v := 30 + rng.Intn(11)
			if x >= 6 && x < 18 && y >= 6 && y < 18 {
				v = 200 + rng.Intn(41)
			}
			m.Set(y, x, 0, uint8(v))
		}
	}
	return m
}

// noisePatch builds a 24x24 patch without the central block.
func noisePatch(rng *rand.Rand) *cv.Mat {
	m := cv.NewMat(24, 24, 1)
	for i := range m.Data {
		m.Data[i] = uint8(90 + rng.Intn(80))
	}
	return m
}

// ExampleWBDetector trains a detector on toy patches and locates the object in
// a larger image.
func ExampleWBDetector() {
	rng := rand.New(rand.NewSource(1))
	var pos, neg []*cv.Mat
	for i := 0; i < 30; i++ {
		pos = append(pos, brightBlock(rng))
		neg = append(neg, noisePatch(rng))
	}

	d := xobjdetect.NewWBDetector()
	d.Rounds = 30
	d.NumFeatures = 150
	if err := d.Train(pos, neg); err != nil {
		panic(err)
	}

	scene := cv.NewMat(80, 80, 1)
	for i := range scene.Data {
		scene.Data[i] = 35
	}
	brightBlock(rand.New(rand.NewSource(2))).CopyTo(scene, 30, 40)

	rects, _ := d.Detect(scene)
	fmt.Println(len(rects) > 0)
	// Output: true
}

// ExampleWaldBoost trains the boosting core directly on feature vectors.
func ExampleWaldBoost() {
	pos := [][]float64{{0.9, 0.1}, {0.85, 0.2}, {0.95, 0.05}}
	neg := [][]float64{{0.1, 0.9}, {0.2, 0.85}, {0.05, 0.95}}

	wb := xobjdetect.NewWaldBoost(10)
	if err := wb.Train(pos, neg); err != nil {
		panic(err)
	}
	_, accepted := wb.Predict([]float64{0.9, 0.1})
	fmt.Println(accepted)
	// Output: true
}

// ExampleACFFeatureEvaluator turns an image patch into an integral-channel
// feature vector.
func ExampleACFFeatureEvaluator() {
	pool := xobjdetect.NewFeaturePool(24, 24, 64, rand.New(rand.NewSource(0)))
	e := xobjdetect.NewACFFeatureEvaluator(pool)
	v := e.Sample(brightBlock(rand.New(rand.NewSource(3))))
	fmt.Println(len(v))
	// Output: 64
}

// ExampleWBDetector_persistence shows a trained detector surviving a gob
// round-trip unchanged.
func ExampleWBDetector_persistence() {
	rng := rand.New(rand.NewSource(1))
	var pos, neg []*cv.Mat
	for i := 0; i < 20; i++ {
		pos = append(pos, brightBlock(rng))
		neg = append(neg, noisePatch(rng))
	}
	d := xobjdetect.NewWBDetector()
	d.Rounds = 20
	d.NumFeatures = 120
	if err := d.Train(pos, neg); err != nil {
		panic(err)
	}

	var buf bytes.Buffer
	if err := d.Write(&buf); err != nil {
		panic(err)
	}
	var restored xobjdetect.WBDetector
	if err := restored.Read(&buf); err != nil {
		panic(err)
	}
	fmt.Println(restored.Trained())
	// Output: true
}
