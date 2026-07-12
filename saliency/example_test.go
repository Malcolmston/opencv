package saliency_test

import (
	"fmt"

	cv "github.com/malcolmston/opencv"
	"github.com/malcolmston/opencv/saliency"
)

// brightDisk builds a small single-channel image with a flat dark background
// and a bright disk in the middle — a single distinct object.
func brightDisk() *cv.Mat {
	const size, cy, cx, r = 48, 24, 24, 6
	m := cv.NewMat(size, size, 1)
	m.SetTo(30)
	for y := 0; y < size; y++ {
		for x := 0; x < size; x++ {
			if (y-cy)*(y-cy)+(x-cx)*(x-cx) <= r*r {
				m.Set(y, x, 0, 220)
			}
		}
	}
	return m
}

// ExampleStaticSaliencySpectralResidual detects a bright object with the
// spectral-residual method and confirms it is more salient than the background.
func ExampleStaticSaliencySpectralResidual() {
	img := brightDisk()
	sal := saliency.NewStaticSaliencySpectralResidual().ComputeSaliency(img)
	center := sal.At(24, 24, 0)
	corner := sal.At(0, 0, 0)
	fmt.Println(center > corner)
	// Output: true
}

// ExampleStaticSaliencyFineGrained detects the same object with the
// fine-grained center-surround method.
func ExampleStaticSaliencyFineGrained() {
	img := brightDisk()
	sal := saliency.NewStaticSaliencyFineGrained().ComputeSaliency(img)
	center := sal.At(24, 24, 0)
	corner := sal.At(0, 0, 0)
	fmt.Println(center > corner)
	// Output: true
}

// ExampleComputeBinaryMap turns a saliency map into a binary mask that isolates
// the salient object.
func ExampleComputeBinaryMap() {
	img := brightDisk()
	sal := saliency.NewStaticSaliencyFineGrained().ComputeSaliency(img)
	mask := saliency.ComputeBinaryMap(sal)
	fmt.Println(mask.At(24, 24, 0), mask.At(0, 0, 0))
	// Output: 255 0
}

// ExampleMotionSaliencyBinWangApr2014 learns a static background from two
// frames, then flags a blob that appears in the third.
func ExampleMotionSaliencyBinWangApr2014() {
	const size = 24
	det := saliency.NewMotionSaliencyBinWangApr2014(size, size)

	background := cv.NewMat(size, size, 1)
	background.SetTo(50)
	det.ComputeSaliency(background)         // seed
	det.ComputeSaliency(background.Clone()) // confirm background

	frame := background.Clone()
	for y := 8; y < 16; y++ {
		for x := 8; x < 16; x++ {
			frame.Set(y, x, 0, 200)
		}
	}
	motion := det.ComputeSaliency(frame)
	fmt.Println(motion.At(11, 11, 0), motion.At(0, 0, 0))
	// Output: 255 0
}

// ExampleObjectnessBING proposes candidate object windows ranked by objectness.
func ExampleObjectnessBING() {
	img := brightDisk()
	boxes := saliency.NewObjectnessBING().ComputeObjectness(img)
	top := boxes[0]
	overlaps := 24 >= top.X && 24 < top.X+top.W && 24 >= top.Y && 24 < top.Y+top.H
	fmt.Println(len(boxes) > 0, overlaps)
	// Output: true true
}
