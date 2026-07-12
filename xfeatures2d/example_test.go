package xfeatures2d_test

import (
	"fmt"

	cv "github.com/malcolmston/opencv"
	"github.com/malcolmston/opencv/xfeatures2d"
)

// ExampleSimpleBlobDetector detects filled circles drawn on a white background.
func ExampleSimpleBlobDetector() {
	img := cv.NewMat(100, 100, 1)
	img.SetTo(255)
	for _, c := range []cv.Point{{X: 30, Y: 30}, {X: 70, Y: 70}} {
		cv.Circle(img, c, 11, cv.NewScalar(0), cv.Filled)
	}

	det := xfeatures2d.NewSimpleBlobDetector()
	blobs := det.Detect(img)

	fmt.Println("blobs:", len(blobs))
	// Output:
	// blobs: 2
}

// ExampleAGAST finds corners with the adaptive-threshold FAST segment test.
func ExampleAGAST() {
	// A white square on a black background: its four convex corners are strong
	// FAST/AGAST features.
	img := cv.NewMat(60, 60, 1)
	for y := 20; y < 40; y++ {
		for x := 20; x < 40; x++ {
			img.Data[y*60+x] = 255
		}
	}

	kps := xfeatures2d.NewAGAST(30).Detect(img)
	fmt.Println("found corners:", len(kps) > 0)
	// Output:
	// found corners: true
}

// ExampleGFTTDetector wraps the Shi–Tomasi corner detector.
func ExampleGFTTDetector() {
	img := cv.NewMat(80, 80, 1)
	for y := 0; y < 80; y++ {
		for x := 0; x < 80; x++ {
			if (x/20+y/20)%2 == 0 {
				img.Data[y*80+x] = 255
			}
		}
	}

	det := xfeatures2d.NewGFTTDetector(10)
	kps := det.Detect(img)
	fmt.Println("at most 10 corners:", len(kps) <= 10 && len(kps) > 0)
	// Output:
	// at most 10 corners: true
}

// ExampleBRISK_Compute shows that two identical neighbourhoods produce the same
// binary descriptor (Hamming distance 0).
func ExampleBRISK_Compute() {
	img := cv.NewMat(120, 120, 1)
	for y := 0; y < 120; y++ {
		for x := 0; x < 120; x++ {
			img.Data[y*120+x] = uint8((x*3 + y) % 200)
		}
	}
	// Stamp an identical 40×40 patch at two locations.
	for _, c := range []cv.Point{{X: 30, Y: 60}, {X: 90, Y: 60}} {
		for i := 0; i < 40; i++ {
			for j := 0; j < 40; j++ {
				img.Data[(c.Y-20+i)*120+(c.X-20+j)] = uint8((i*9 + j*5) % 256)
			}
		}
	}

	b := xfeatures2d.NewBRISK(20)
	_, descs := b.Compute(img, []xfeatures2d.KeyPoint{
		{Pt: cv.Point{X: 30, Y: 60}},
		{Pt: cv.Point{X: 90, Y: 60}},
	})
	fmt.Println("hamming:", xfeatures2d.HammingDistance(descs[0], descs[1]))
	// Output:
	// hamming: 0
}

// tiledExample builds a 220×220 image whose intensity is exactly periodic with
// the given period, so points a whole period apart have identical
// neighbourhoods. It is the example counterpart of the test helper of the same
// shape.
func tiledExample(period int) *cv.Mat {
	m := cv.NewMat(220, 220, 1)
	for y := 0; y < 220; y++ {
		for x := 0; x < 220; x++ {
			u, v := x%period, y%period
			m.Data[y*220+x] = uint8((u*7 + v*13 + u*v*3) % 256)
		}
	}
	return m
}

// ExampleFREAK shows that two points a full texture period apart yield the same
// rotation-invariant binary descriptor (Hamming distance 0).
func ExampleFREAK() {
	img := tiledExample(40)
	f := xfeatures2d.NewFREAK(20)
	_, descs := f.Compute(img, []xfeatures2d.KeyPoint{
		{Pt: cv.Point{X: 70, Y: 110}},
		{Pt: cv.Point{X: 110, Y: 110}},
	})
	fmt.Println("match:", xfeatures2d.HammingDistance(descs[0], descs[1]) == 0)
	// Output:
	// match: true
}

// ExampleDAISY computes the dense gradient-histogram descriptor and compares two
// identical neighbourhoods (L2 distance 0).
func ExampleDAISY() {
	img := tiledExample(40)
	d := xfeatures2d.NewDAISY()
	_, descs := d.Compute(img, []xfeatures2d.KeyPoint{
		{Pt: cv.Point{X: 70, Y: 110}},
		{Pt: cv.Point{X: 110, Y: 110}},
	})
	fmt.Println("dim:", len(descs[0]))
	fmt.Println("identical:", xfeatures2d.L2Distance(descs[0], descs[1]) < 1e-9)
	// Output:
	// dim: 200
	// identical: true
}

// ExampleSURF_Compute describes a shifted copy of a neighbourhood and confirms
// the two SURF descriptors coincide.
func ExampleSURF_Compute() {
	img := tiledExample(40)
	s := xfeatures2d.NewSURF(200)
	_, descs := s.Compute(img, []xfeatures2d.KeyPoint{
		{Pt: cv.Point{X: 70, Y: 110}},
		{Pt: cv.Point{X: 110, Y: 110}},
	})
	fmt.Println("shifted copy matches:", xfeatures2d.L2Distance(descs[0], descs[1]) < 1e-9)
	// Output:
	// shifted copy matches: true
}

// ExampleLUCID builds the rank-order descriptor and shows it is illumination
// invariant for identical neighbourhoods (distance 0).
func ExampleLUCID() {
	img := tiledExample(40)
	l := xfeatures2d.NewLUCID(5, 2)
	_, descs := l.Compute(img, []xfeatures2d.KeyPoint{
		{Pt: cv.Point{X: 70, Y: 110}},
		{Pt: cv.Point{X: 110, Y: 110}},
	})
	fmt.Println("distance:", xfeatures2d.LUCIDDistance(descs[0], descs[1]))
	// Output:
	// distance: 0
}

// ExampleBEBLID computes the boosted box descriptor for two identical
// neighbourhoods.
func ExampleBEBLID() {
	img := tiledExample(40)
	b := xfeatures2d.NewBEBLID(32)
	_, descs := b.Compute(img, []xfeatures2d.KeyPoint{
		{Pt: cv.Point{X: 70, Y: 110}},
		{Pt: cv.Point{X: 110, Y: 110}},
	})
	fmt.Println("hamming:", xfeatures2d.HammingDistance(descs[0], descs[1]))
	// Output:
	// hamming: 0
}

// ExampleMSDDetector finds maximal-self-dissimilarity keypoints on a grid of
// squares.
func ExampleMSDDetector() {
	img := cv.NewMat(120, 120, 1)
	for by := 0; by < 3; by++ {
		for bx := 0; bx < 3; bx++ {
			ox, oy := 12+bx*32, 12+by*32
			for y := 0; y < 20; y++ {
				for x := 0; x < 20; x++ {
					img.Data[(oy+y)*120+(ox+x)] = 255
				}
			}
		}
	}
	det := xfeatures2d.NewMSDDetector()
	det.Threshold = 100
	fmt.Println("found:", len(det.Detect(img)) > 0)
	// Output:
	// found: true
}

// ExamplePCTSignatures computes a PCT signature and shows the SQFD of a
// signature with itself is zero.
func ExamplePCTSignatures() {
	img := cv.NewMat(120, 120, 1)
	for y := 0; y < 120; y++ {
		for x := 0; x < 120; x++ {
			if (x/20+y/20)%2 == 0 {
				img.Data[y*120+x] = 255
			}
		}
	}
	sig := xfeatures2d.NewPCTSignatures().ComputeSignature(img)
	fmt.Println("non-empty:", len(sig) > 0)
	fmt.Printf("self SQFD: %.0f\n", xfeatures2d.SQFD(sig, sig, 1.0))
	// Output:
	// non-empty: true
	// self SQFD: 0
}

// ExampleMatchGMS filters a set of putative matches, keeping the geometrically
// consistent cluster and discarding scattered outliers.
func ExampleMatchGMS() {
	var kp1, kp2 []xfeatures2d.KeyPoint
	var matches []xfeatures2d.DMatch
	// A consistent block of matches translated by (20, 20).
	for gy := 0; gy < 10; gy++ {
		for gx := 0; gx < 10; gx++ {
			x, y := 40+gx*4, 40+gy*4
			i := len(kp1)
			kp1 = append(kp1, xfeatures2d.KeyPoint{Pt: cv.Point{X: x, Y: y}})
			kp2 = append(kp2, xfeatures2d.KeyPoint{Pt: cv.Point{X: x + 20, Y: y + 20}})
			matches = append(matches, xfeatures2d.DMatch{QueryIdx: i, TrainIdx: i, Distance: 1})
		}
	}
	inliers := len(matches)
	// Scattered outliers.
	for k := 0; k < 40; k++ {
		i := len(kp1)
		kp1 = append(kp1, xfeatures2d.KeyPoint{Pt: cv.Point{X: 5 + (k*37)%180, Y: 5 + (k*53)%180}})
		kp2 = append(kp2, xfeatures2d.KeyPoint{Pt: cv.Point{X: 5 + (k*91)%180, Y: 5 + (k*29)%180}})
		matches = append(matches, xfeatures2d.DMatch{QueryIdx: i, TrainIdx: i, Distance: 1})
	}
	kept := xfeatures2d.MatchGMS(200, 200, 200, 200, kp1, kp2, matches, false, false, 6)
	fmt.Println("kept fewer than input:", len(kept) < len(matches))
	fmt.Println("kept the cluster:", len(kept) >= inliers-10)
	// Output:
	// kept fewer than input: true
	// kept the cluster: true
}
