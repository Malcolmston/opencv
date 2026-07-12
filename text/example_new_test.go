package text

import (
	"fmt"

	cv "github.com/malcolmston/opencv"
)

// ExampleDetectRegions runs the full classical pipeline on two rows of blocky
// "characters" and reports how many text lines it recovers.
func ExampleDetectRegions() {
	blobs := [][4]int{
		{4, 4, 5, 7}, {14, 4, 5, 7}, {24, 4, 5, 7},
		{4, 24, 5, 7}, {14, 24, 5, 7}, {24, 24, 5, 7},
	}
	img := cv.NewMat(36, 36, 1)
	img.SetTo(220)
	for _, b := range blobs {
		for y := b[1]; y < b[1]+b[3]; y++ {
			for x := b[0]; x < b[0]+b[2]; x++ {
				img.Set(y, x, 0, 40)
			}
		}
	}

	lines := DetectRegions(img, DefaultTextDetectorParams())
	fmt.Printf("%d text lines\n", len(lines))
	// Output: 2 text lines
}

// ExampleOCRTemplate renders a word in the built-in font and reads it back with
// the nearest-template recognizer.
func ExampleOCRTemplate() {
	img := RenderText("OPENCV7", 3, 1)
	o := NewOCRTemplateAlnum()
	fmt.Println(o.RecognizeWord(img))
	// Output: OPENCV7
}

// ExampleBeamSearchDecoder decodes a per-character score matrix into a word,
// using a lexicon to override a slightly higher-scoring non-word symbol.
func ExampleBeamSearchDecoder() {
	alphabet := "ACRTZ"
	scores := [][]float64{
		{0.9, 0.0, 0.0, 0.0, 0.0}, // C
		{0.0, 0.9, 0.0, 0.0, 0.0}, // A
		{0.0, 0.0, 0.3, 0.4, 0.5}, // Z (0.5) edges out T (0.4)
	}
	lex := NewLexicon([]string{"CAT", "CAR"})
	d := NewBeamSearchDecoder(alphabet, 8).WithLexicon(lex)
	fmt.Println(d.Decode(scores))
	// Output: CAT
}

// ExampleTextDetectorSWT finds two constant-width strokes with the Stroke Width
// Transform.
func ExampleTextDetectorSWT() {
	img := cv.NewMat(30, 40, 1)
	for _, x0 := range []int{6, 24} {
		for y := 3; y < 27; y++ {
			for x := x0; x < x0+3; x++ {
				img.Set(y, x, 0, 255)
			}
		}
	}
	d := NewTextDetectorSWT(DefaultSWTParams())
	fmt.Printf("%d strokes\n", len(d.Detect(img)))
	// Output: 2 strokes
}

// ExampleERGrouping groups a row of character boxes into one horizontal line.
func ExampleERGrouping() {
	boxes := []cv.Rect{
		{X: 0, Y: 0, Width: 8, Height: 10},
		{X: 10, Y: 0, Width: 8, Height: 10},
		{X: 20, Y: 0, Width: 8, Height: 10},
	}
	lines := ERGrouping(boxes, OrientationHoriz)
	fmt.Printf("%d line of %d chars\n", len(lines), len(lines[0]))
	// Output: 1 line of 3 chars
}
