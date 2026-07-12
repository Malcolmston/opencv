package text

import (
	"testing"

	cv "github.com/malcolmston/opencv"
)

func TestOCRTemplateReadsEveryGlyph(t *testing.T) {
	o := NewOCRTemplateAlnum()
	for _, ch := range SupportedChars() {
		g, _ := FontGlyph(ch, 4)
		if label, dist := o.RecognizeChar(g); label != string(ch) {
			t.Errorf("RecognizeChar(%q) = %q dist %d, want %q", string(ch), label, dist, string(ch))
		}
	}
}

func TestOCRTemplateRecognizeWord(t *testing.T) {
	o := NewOCRTemplateAlnum()
	for _, word := range []string{"HELLO", "OPENCV", "GO123", "TEXT7"} {
		img := RenderText(word, 3, 1)
		if got := o.RecognizeWord(img); got != word {
			t.Errorf("RecognizeWord(%q) = %q", word, got)
		}
	}
}

func TestOCRTemplateDigitsOnly(t *testing.T) {
	o := NewOCRTemplateDigits()
	if o.Alphabet() != "0123456789" {
		t.Fatalf("digit alphabet = %q", o.Alphabet())
	}
	img := RenderText("0123456789", 2, 1)
	if got := o.RecognizeWord(img); got != "0123456789" {
		t.Errorf("RecognizeWord = %q, want 0123456789", got)
	}
}

func TestOCRTemplateRunMultiLine(t *testing.T) {
	o := NewOCRTemplateAlnum()
	top := RenderText("ABC", 3, 1)
	bottom := RenderText("XYZ", 3, 1)
	// Stack two rendered lines into one image with a blank separator.
	rowsTop, rowsBot := top.Rows, bottom.Rows
	cols := top.Cols
	if bottom.Cols > cols {
		cols = bottom.Cols
	}
	gap := 6
	img := cv.NewMat(rowsTop+gap+rowsBot, cols, 1)
	top.CopyTo(img, 0, 0)
	bottom.CopyTo(img, rowsTop+gap, 0)

	if got := o.Run(img); got != "ABC\nXYZ" {
		t.Errorf("Run = %q, want \"ABC\\nXYZ\"", got)
	}
}

func TestOCRTemplateCharScoresShape(t *testing.T) {
	o := NewOCRTemplateAlnum()
	img := RenderText("CAT", 3, 1)
	scores := o.CharScores(img)
	if len(scores) != 3 {
		t.Fatalf("got %d score rows, want 3", len(scores))
	}
	alpha := []rune(o.Alphabet())
	for i, row := range scores {
		if len(row) != len(alpha) {
			t.Fatalf("row %d width %d, want %d", i, len(row), len(alpha))
		}
		// The argmax of each row must be the correct character of "CAT".
		best, bestIdx := row[0], 0
		for j, v := range row {
			if v > best {
				best, bestIdx = v, j
			}
		}
		want := []rune("CAT")[i]
		if alpha[bestIdx] != want {
			t.Errorf("char %d argmax = %q, want %q", i, string(alpha[bestIdx]), string(want))
		}
		if best <= 0 || best > 1.0001 {
			t.Errorf("char %d score %v out of (0,1]", i, best)
		}
	}
}
