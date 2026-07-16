package xobjdetect_test

import (
	"fmt"
	"math/rand"

	cv "github.com/malcolmston/opencv"
	"github.com/malcolmston/opencv/xobjdetect"
)

// Example trains a WaldBoost object detector on synthetic positive and negative
// 24×24 patches, then slides it over a scene containing one planted target and
// prints how many objects it locates. Every random draw is seeded, so the count
// is deterministic.
func Example() {
	// Positives are a dark field with a bright central block; negatives are
	// uniform noise.
	rng := rand.New(rand.NewSource(1))
	target := func(r *rand.Rand) *cv.Mat {
		m := cv.NewMat(24, 24, 1)
		for y := 0; y < 24; y++ {
			for x := 0; x < 24; x++ {
				v := 30 + r.Intn(11)
				if x >= 6 && x < 18 && y >= 6 && y < 18 {
					v = 200 + r.Intn(41)
				}
				m.Set(y, x, 0, uint8(v))
			}
		}
		return m
	}
	var pos, neg []*cv.Mat
	for i := 0; i < 30; i++ {
		pos = append(pos, target(rng))
		p := cv.NewMat(24, 24, 1)
		for j := range p.Data {
			p.Data[j] = uint8(90 + rng.Intn(80))
		}
		neg = append(neg, p)
	}

	d := xobjdetect.NewWBDetector()
	d.Rounds = 30
	d.NumFeatures = 150
	if err := d.Train(pos, neg); err != nil {
		panic(err)
	}

	// Plant one target in a dark scene and detect it.
	scene := cv.NewMat(80, 80, 1)
	for i := range scene.Data {
		scene.Data[i] = 35
	}
	target(rand.New(rand.NewSource(2))).CopyTo(scene, 30, 40)

	rects, _ := d.Detect(scene)
	fmt.Printf("detections=%d\n", len(rects))
	// Output: detections=35
}
