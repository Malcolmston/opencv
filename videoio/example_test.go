package videoio_test

import (
	"fmt"
	"log"
	"os"
	"path/filepath"

	cv "github.com/malcolmston/opencv"
	"github.com/malcolmston/opencv/videoio"
)

// makeExampleDir returns a fresh temporary directory for example output.
func makeExampleDir() (string, error) {
	return os.MkdirTemp("", "videoio-example-")
}

// ExampleWriteGIF encodes a few solid-colour frames to an animated GIF and reads
// them back, showing the basic write-then-read round trip.
func ExampleWriteGIF() {
	dir, err := makeExampleDir()
	if err != nil {
		log.Fatal(err)
	}
	path := filepath.Join(dir, "solid.gif")

	frames := make([]*cv.Mat, 3)
	for i := range frames {
		m := cv.NewMat(4, 4, 3)
		m.SetTo(uint8(i * 80)) // a distinct grey per frame
		frames[i] = m
	}

	if err := videoio.WriteGIF(path, frames, 10); err != nil {
		log.Fatal(err)
	}

	got, delays, err := videoio.ReadGIF(path)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("%d frames, delay %d cs\n", len(got), delays[0])
	// Output: 3 frames, delay 10 cs
}

// ExampleVideoCapture streams frames from a GIF one at a time.
func ExampleVideoCapture() {
	dir, err := makeExampleDir()
	if err != nil {
		log.Fatal(err)
	}
	path := filepath.Join(dir, "cap.gif")

	src := []*cv.Mat{cv.NewMat(2, 2, 3), cv.NewMat(2, 2, 3)}
	if err := videoio.WriteGIF(path, src, 5); err != nil {
		log.Fatal(err)
	}

	cap, err := videoio.OpenGIF(path)
	if err != nil {
		log.Fatal(err)
	}
	defer cap.Close()

	count := 0
	for {
		if _, ok := cap.Read(); !ok {
			break
		}
		count++
	}
	fmt.Printf("read %d frames\n", count)
	// Output: read 2 frames
}
