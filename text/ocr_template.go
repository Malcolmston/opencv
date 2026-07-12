package text

import (
	"strings"

	cv "github.com/malcolmston/opencv"
)

// OCR canonical template size: every glyph, whether stored template or query
// crop, is tightened to its ink and resampled to this size before matching, so a
// glyph rendered at any scale reads back to the same label.
const (
	ocrTemplateW = 10
	ocrTemplateH = 14
	ocrThreshold = 127
)

// OCRTemplate is a nearest-template optical character recognizer over the
// built-in bitmap font (see [SupportedChars]). It is the segmentation-and-match
// pipeline OpenCV's cv::text::OCRTesseract/OCRHMMDecoder occupy, implemented here
// without any trained model: an input image is cut into character boxes by
// projection ([SegmentChars]), each box is tightened and resampled to a canonical
// size, and its label is the closest font template by Hamming distance
// ([NearestGlyphClassifier]).
//
// Input is expected as bright ink on a dark background, matching [RenderText]. It
// reads the fixed built-in font reliably; it is not a general scene-text OCR.
type OCRTemplate struct {
	clf       *NearestGlyphClassifier
	threshold uint8
	alphabet  string
}

// NewOCRTemplateDigits returns a recognizer restricted to the digits 0-9.
func NewOCRTemplateDigits() *OCRTemplate {
	return newOCRTemplate("0123456789")
}

// NewOCRTemplateAlnum returns a recognizer over the digits 0-9 and the uppercase
// letters A-Z — the full built-in alphabet.
func NewOCRTemplateAlnum() *OCRTemplate {
	return newOCRTemplate(SupportedChars())
}

// newOCRTemplate builds a recognizer whose templates are the tight-cropped glyphs
// of the given characters. Unsupported characters are skipped.
func newOCRTemplate(chars string) *OCRTemplate {
	templates := map[string]*cv.Mat{}
	var alphabet []rune
	for _, ch := range chars {
		glyph, ok := FontGlyph(ch, 1)
		if !ok {
			continue
		}
		templates[string(ch)] = tightCrop(glyph, ocrThreshold)
		alphabet = append(alphabet, ch)
	}
	if len(templates) == 0 {
		panic("text: newOCRTemplate requires at least one supported character")
	}
	return &OCRTemplate{
		clf:       NewNearestGlyphClassifier(ocrTemplateW, ocrTemplateH, ocrThreshold, templates),
		threshold: ocrThreshold,
		alphabet:  string(alphabet),
	}
}

// Alphabet returns the characters this recognizer can output, in ascending order.
func (o *OCRTemplate) Alphabet() string { return o.alphabet }

// RecognizeChar classifies a single isolated glyph image, returning its label and
// the Hamming distance to the matched template. The image is tightened to its ink
// before matching, so surrounding blank margin does not matter.
func (o *OCRTemplate) RecognizeChar(img *cv.Mat) (label string, distance int) {
	return o.clf.Classify(tightCrop(img, o.threshold))
}

// RecognizeWord segments one line of text into characters ([SegmentChars]) and
// returns the concatenation of their labels, left-to-right. No space is inserted
// for gaps; use [OCRTemplate.Run] for multi-line input.
func (o *OCRTemplate) RecognizeWord(img *cv.Mat) string {
	boxes := SegmentChars(img, o.threshold)
	var sb strings.Builder
	for _, b := range boxes {
		crop := img.Region(b.Y, b.X, b.Height, b.Width)
		label, _ := o.RecognizeChar(crop)
		sb.WriteString(label)
	}
	return sb.String()
}

// Run recognizes a full text image: it splits it into lines ([SegmentLines]),
// reads each line with [OCRTemplate.RecognizeWord], and joins the lines with
// newlines, top-to-bottom.
func (o *OCRTemplate) Run(img *cv.Mat) string {
	lines := SegmentLines(img, o.threshold)
	parts := make([]string, 0, len(lines))
	for _, lb := range lines {
		crop := img.Region(lb.Y, lb.X, lb.Height, lb.Width)
		parts = append(parts, o.RecognizeWord(crop))
	}
	return strings.Join(parts, "\n")
}

// CharScores classifies each segmented character of one text line and returns,
// per character, a score for every alphabet symbol. The score is a similarity in
// [0,1] equal to 1 minus the normalized Hamming distance, suitable as input to a
// [BeamSearchDecoder]. The returned matrix has one row per detected character and
// one column per rune of [OCRTemplate.Alphabet].
func (o *OCRTemplate) CharScores(img *cv.Mat) [][]float64 {
	boxes := SegmentChars(img, o.threshold)
	alpha := []rune(o.alphabet)
	scores := make([][]float64, len(boxes))
	for i, b := range boxes {
		crop := img.Region(b.Y, b.X, b.Height, b.Width)
		q := binarizeResized(tightCrop(crop, o.threshold), ocrTemplateW, ocrTemplateH, o.threshold)
		row := make([]float64, len(alpha))
		total := float64(ocrTemplateW * ocrTemplateH)
		for j, ch := range alpha {
			tmpl := o.clf.templates[o.templateIndex(string(ch))]
			row[j] = 1 - float64(hamming(q, tmpl))/total
		}
		scores[i] = row
	}
	return scores
}

// templateIndex returns the index of the template for label within the backing
// classifier (its labels are sorted), or panics if the label is absent.
func (o *OCRTemplate) templateIndex(label string) int {
	for i, lbl := range o.clf.labels {
		if lbl == label {
			return i
		}
	}
	panic("text: OCRTemplate has no template for " + label)
}

// tightCrop returns the sub-image of img covering its inked bounding box (samples
// with grayscale value greater than threshold). An image with no ink is returned
// unchanged (cloned).
func tightCrop(img *cv.Mat, threshold uint8) *cv.Mat {
	gray := toGray(img)
	rows, cols := gray.Rows, gray.Cols
	minX, minY, maxX, maxY := cols, rows, -1, -1
	for y := 0; y < rows; y++ {
		base := y * cols
		for x := 0; x < cols; x++ {
			if gray.Data[base+x] > threshold {
				if x < minX {
					minX = x
				}
				if x > maxX {
					maxX = x
				}
				if y < minY {
					minY = y
				}
				if y > maxY {
					maxY = y
				}
			}
		}
	}
	if maxX < 0 {
		return gray.Clone()
	}
	return gray.Region(minY, minX, maxY-minY+1, maxX-minX+1)
}
