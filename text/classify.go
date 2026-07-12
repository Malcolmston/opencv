package text

import (
	"sort"

	cv "github.com/malcolmston/opencv"
)

// NearestGlyphClassifier is a trivial nearest-template recognizer. Each label is
// backed by one binary glyph template; a query image is resized to the common
// template size, binarized, and assigned the label of the closest template by
// Hamming distance. It is intended for reading a small fixed set of shapes such
// as the digits of a seven-segment display, not general OCR — a full trained or
// neural recognizer is deferred (see the package overview).
type NearestGlyphClassifier struct {
	width     int
	height    int
	threshold uint8
	labels    []string
	templates [][]bool // parallel to labels; length width*height each
}

// NewNearestGlyphClassifier builds a classifier over templates keyed by label.
// Every template Mat is reduced to grayscale, resized to width×height by
// nearest-neighbour sampling and binarized at threshold (a sample is "on" when
// its grayscale value exceeds threshold). width, height and the template map
// must be non-empty; it panics otherwise.
func NewNearestGlyphClassifier(width, height int, threshold uint8, templates map[string]*cv.Mat) *NearestGlyphClassifier {
	if width <= 0 || height <= 0 {
		panic("text: NewNearestGlyphClassifier requires positive template size")
	}
	if len(templates) == 0 {
		panic("text: NewNearestGlyphClassifier requires at least one template")
	}
	labels := make([]string, 0, len(templates))
	for label := range templates {
		labels = append(labels, label)
	}
	sort.Strings(labels) // deterministic tie-breaking
	c := &NearestGlyphClassifier{
		width: width, height: height, threshold: threshold,
		labels: labels,
	}
	for _, label := range labels {
		c.templates = append(c.templates, binarizeResized(templates[label], width, height, threshold))
	}
	return c
}

// Classify returns the label of the template closest to img and the Hamming
// distance (number of differing pixels) to it. Ties are broken by lexical label
// order, so results are deterministic.
func (c *NearestGlyphClassifier) Classify(img *cv.Mat) (label string, distance int) {
	q := binarizeResized(img, c.width, c.height, c.threshold)
	best := -1
	bestDist := 0
	for i, tmpl := range c.templates {
		d := hamming(q, tmpl)
		if best == -1 || d < bestDist {
			best = i
			bestDist = d
		}
	}
	return c.labels[best], bestDist
}

// binarizeResized reduces img to grayscale, resizes it to width×height by
// nearest-neighbour sampling and returns a boolean mask that is true where the
// grayscale value exceeds threshold.
func binarizeResized(img *cv.Mat, width, height int, threshold uint8) []bool {
	gray := toGray(img)
	out := make([]bool, width*height)
	for y := 0; y < height; y++ {
		sy := y * gray.Rows / height
		if sy >= gray.Rows {
			sy = gray.Rows - 1
		}
		for x := 0; x < width; x++ {
			sx := x * gray.Cols / width
			if sx >= gray.Cols {
				sx = gray.Cols - 1
			}
			out[y*width+x] = gray.Data[sy*gray.Cols+sx] > threshold
		}
	}
	return out
}

// hamming counts the positions at which two equal-length boolean masks differ.
func hamming(a, b []bool) int {
	d := 0
	for i := range a {
		if a[i] != b[i] {
			d++
		}
	}
	return d
}
