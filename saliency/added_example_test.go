package saliency_test

import (
	"fmt"

	cv "github.com/malcolmston/opencv"
	"github.com/malcolmston/opencv/saliency"
)

// ExampleStaticSaliencyIttiKochNiebur highlights a bright object with the
// classical Itti-Koch-Niebur attention model.
func ExampleStaticSaliencyIttiKochNiebur() {
	img := brightDisk()
	sal := saliency.NewStaticSaliencyIttiKochNiebur().ComputeSaliency(img)
	fmt.Println(sal.At(24, 24, 0) > sal.At(0, 0, 0))
	// Output: true
}

// ExampleMinimumBarrierSaliency scores the interior of an object by its minimum
// barrier distance to the image border.
func ExampleMinimumBarrierSaliency() {
	img := brightDisk()
	sal := saliency.NewMinimumBarrierSaliency().ComputeSaliency(img)
	fmt.Println(sal.At(24, 24, 0) > sal.At(0, 0, 0))
	// Output: true
}

// ExampleStaticSaliencyFrequencyTuned detects an object via its Lab-colour
// deviation from the image mean.
func ExampleStaticSaliencyFrequencyTuned() {
	img := brightDisk()
	sal := saliency.NewStaticSaliencyFrequencyTuned().ComputeSaliency(img)
	fmt.Println(sal.At(24, 24, 0) > sal.At(0, 0, 0))
	// Output: true
}

// ExampleGMRSaliency ranks image regions against the border to find the salient
// object.
func ExampleGMRSaliency() {
	img := brightDisk()
	sal := saliency.NewGMRSaliency().ComputeSaliency(img)
	fmt.Println(sal.At(24, 24, 0) > sal.At(0, 0, 0))
	// Output: true
}

// ExampleStaticSaliencyBooleanMap activates surrounded regions across many
// Boolean maps.
func ExampleStaticSaliencyBooleanMap() {
	img := brightDisk()
	sal := saliency.NewStaticSaliencyBooleanMap().ComputeSaliency(img)
	fmt.Println(sal.At(24, 24, 0) > sal.At(0, 0, 0))
	// Output: true
}

// ExampleHistogramContrast scores rare, contrasting colours as salient.
func ExampleHistogramContrast() {
	img := brightDisk()
	sal := saliency.NewHistogramContrast().ComputeSaliency(img)
	fmt.Println(sal.At(24, 24, 0) > sal.At(0, 0, 0))
	// Output: true
}

// ExampleRegionContrast measures global region contrast weighted by spatial
// distance.
func ExampleRegionContrast() {
	img := brightDisk()
	sal := saliency.NewRegionContrast().ComputeSaliency(img)
	fmt.Println(sal.At(24, 24, 0) > sal.At(0, 0, 0))
	// Output: true
}

// ExampleStaticSaliencyContextAware highlights colour-unique regions.
func ExampleStaticSaliencyContextAware() {
	img := brightDisk()
	sal := saliency.NewStaticSaliencyContextAware().ComputeSaliency(img)
	fmt.Println(sal.At(24, 24, 0) > sal.At(0, 0, 0))
	// Output: true
}

// ExampleObjectnessCascade proposes ranked object windows with a two-stage
// scorer and non-maximum suppression.
func ExampleObjectnessCascade() {
	img := brightDisk()
	boxes := saliency.NewObjectnessCascade().ComputeObjectness(img)
	top := boxes[0]
	overlaps := 24 >= top.X && 24 < top.X+top.W && 24 >= top.Y && 24 < top.Y+top.H
	fmt.Println(len(boxes) > 0, overlaps)
	// Output: true true
}

// ExampleAUCJudd scores a saliency map against a binary fixation map.
func ExampleAUCJudd() {
	img := brightDisk()
	sal := saliency.NewStaticSaliencyFineGrained().ComputeSaliency(img)
	fix := cv.NewMat(48, 48, 1)
	fix.Set(24, 24, 0, 255)
	fix.Set(23, 24, 0, 255)
	fix.Set(24, 23, 0, 255)
	fmt.Println(saliency.AUCJudd(sal, fix) > 0.5)
	// Output: true
}

// ExampleSaliencyToHeatmap renders a saliency map as a jet-coloured image.
func ExampleSaliencyToHeatmap() {
	sal := cv.NewMat(2, 2, 1)
	sal.Set(0, 0, 0, 255)
	heat := saliency.SaliencyToHeatmap(sal)
	// Maximum saliency is rendered red (R>B).
	fmt.Println(heat.Channels, heat.At(0, 0, 0) > heat.At(0, 0, 2))
	// Output: 3 true
}

// ExampleAdaptiveBinaryMap thresholds a saliency map at twice its mean.
func ExampleAdaptiveBinaryMap() {
	img := brightDisk()
	sal := saliency.NewMinimumBarrierSaliency().ComputeSaliency(img)
	mask := saliency.AdaptiveBinaryMap(sal, 2.0)
	fmt.Println(mask.At(24, 24, 0), mask.At(0, 0, 0))
	// Output: 255 0
}

// ExampleCenterBiasPrior builds a Gaussian centre-prior map.
func ExampleCenterBiasPrior() {
	prior := saliency.CenterBiasPrior(32, 32, 0.3)
	fmt.Println(prior.At(16, 16, 0), prior.At(16, 16, 0) > prior.At(0, 0, 0))
	// Output: 255 true
}
