package videoio_test

import (
	"fmt"
	"log"
	"path/filepath"

	cv "github.com/malcolmston/opencv"
	"github.com/malcolmston/opencv/videoio"
)

// ExampleWriteAPNG encodes frames to a lossless animated PNG and reads them back.
func ExampleWriteAPNG() {
	dir, err := makeExampleDir()
	if err != nil {
		log.Fatal(err)
	}
	path := filepath.Join(dir, "anim.png")

	frames := make([]*cv.Mat, 3)
	for i := range frames {
		m := cv.NewMat(4, 4, 3)
		m.SetTo(uint8(i * 60))
		frames[i] = m
	}
	if err := videoio.WriteAPNG(path, frames, 8); err != nil {
		log.Fatal(err)
	}

	got, delays, err := videoio.ReadAPNG(path)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("%d frames, delay %d cs, exact=%v\n",
		len(got), delays[0], got[0].At(0, 0, 0) == frames[0].At(0, 0, 0))
	// Output: 3 frames, delay 8 cs, exact=true
}

// ExampleWriteMJPEGAVI writes a Motion-JPEG AVI and reports its frame rate.
func ExampleWriteMJPEGAVI() {
	dir, err := makeExampleDir()
	if err != nil {
		log.Fatal(err)
	}
	path := filepath.Join(dir, "clip.avi")

	frames := []*cv.Mat{cv.NewMat(8, 8, 3), cv.NewMat(8, 8, 3)}
	if err := videoio.WriteMJPEGAVI(path, frames, 30); err != nil {
		log.Fatal(err)
	}
	got, fps, err := videoio.ReadMJPEGAVI(path)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("%d frames at %.0f fps\n", len(got), fps)
	// Output: 2 frames at 30 fps
}

// ExampleWriteImageSequence saves frames as numbered PNG files and reloads them.
func ExampleWriteImageSequence() {
	dir, err := makeExampleDir()
	if err != nil {
		log.Fatal(err)
	}
	frames := []*cv.Mat{cv.NewMat(2, 2, 3), cv.NewMat(2, 2, 3), cv.NewMat(2, 2, 3)}
	if _, err := videoio.WriteImageSequence(dir, "frame%03d.png", frames, 0); err != nil {
		log.Fatal(err)
	}
	got, err := videoio.ReadImageSequence(dir, "frame%03d.png", 0)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("round-tripped %d frames\n", len(got))
	// Output: round-tripped 3 frames
}

// ExampleResampleFrames retimes a two-frame clip to a higher frame rate.
func ExampleResampleFrames() {
	frames := []*cv.Mat{cv.NewMat(2, 2, 3), cv.NewMat(2, 2, 3)}
	delays := []int{50, 50} // half a second each: one second total
	out, outDelays := videoio.ResampleFrames(frames, delays, 10)
	fmt.Printf("%d frames, %d cs each\n", len(out), outDelays[0])
	// Output: 10 frames, 10 cs each
}

// ExampleVideoCapture_Get reads standard properties from a capture.
func ExampleVideoCapture_Get() {
	dir, err := makeExampleDir()
	if err != nil {
		log.Fatal(err)
	}
	path := filepath.Join(dir, "p.gif")
	frames := []*cv.Mat{cv.NewMat(6, 8, 3), cv.NewMat(6, 8, 3)}
	if err := videoio.WriteGIF(path, frames, 10); err != nil {
		log.Fatal(err)
	}
	cap, err := videoio.OpenGIF(path)
	if err != nil {
		log.Fatal(err)
	}
	defer cap.Close()
	fmt.Printf("%.0fx%.0f, %.0f frames, %.0f fps\n",
		cap.Get(videoio.CAP_PROP_FRAME_WIDTH),
		cap.Get(videoio.CAP_PROP_FRAME_HEIGHT),
		cap.Get(videoio.CAP_PROP_FRAME_COUNT),
		cap.Get(videoio.CAP_PROP_FPS))
	// Output: 8x6, 2 frames, 10 fps
}
