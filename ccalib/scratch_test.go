package ccalib

import (
	"fmt"
	"testing"

	cv "github.com/malcolmston/opencv"
)

func TestScratchPattern(t *testing.T) {
	gen := NewRandomPatternGenerator(400, 300, 60, 7)
	pat := gen.Generate()
	blobs := detectBlobs(pat, 6, 0)
	fmt.Printf("generated %d dots, detected %d blobs\n", len(gen.Centers()), len(blobs))

	// Warp the pattern with a mild perspective transform into a bigger scene.
	src := [4]cv.Point{{X: 0, Y: 0}, {X: 399, Y: 0}, {X: 399, Y: 299}, {X: 0, Y: 299}}
	dst := [4]cv.Point{{X: 60, Y: 40}, {X: 470, Y: 70}, {X: 500, Y: 360}, {X: 30, Y: 330}}
	H := cv.GetPerspectiveTransform(src, dst)
	scene := cv.WarpPerspective(pat, H, 560, 420, cv.InterLinear)
	sblobs := detectBlobs(scene, 6, 0)
	fmt.Printf("scene detected %d blobs\n", len(sblobs))

	cp := NewCustomPattern()
	okc := cp.Create(pat, 200, 150)
	fmt.Printf("create ok=%v keypoints=%d\n", okc, cp.KeypointCount())
	obj, img, ok := cp.FindPattern(scene)
	fmt.Printf("find ok=%v matches=%d\n", ok, len(obj))
	_ = img
}
