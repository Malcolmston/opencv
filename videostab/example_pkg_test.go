package videostab_test

import (
	"fmt"
	"math/rand"

	cv "github.com/malcolmston/opencv"
	"github.com/malcolmston/opencv/videostab"
)

// Example runs the one-pass stabilization pipeline end to end: it synthesises a
// short, analytically jittered sequence by cropping a moving window out of a
// larger texture, stabilizes it with a causal Gaussian OnePassStabilizer, and
// prints how many frames came back and their size.
func Example() {
	const size, margin, n = 64, 10, 3

	// A larger textured plate we pan a fixed-size window across to fake camera shake.
	base := makeTexture(size+2*margin, size+2*margin, 3)
	rng := rand.New(rand.NewSource(4))
	frames := make([]*cv.Mat, n)
	for i := 0; i < n; i++ {
		ox := rng.Intn(2*margin + 1)
		oy := rng.Intn(2*margin + 1)
		frames[i] = base.Region(oy, ox, size, size)
	}

	s := videostab.NewOnePassStabilizer(5)
	s.SetFrames(frames)
	out := s.Stabilize()

	fmt.Printf("stabilized %d frames, size %dx%d\n", len(out), out[0].Rows, out[0].Cols)
	// Output: stabilized 3 frames, size 64x64
}
