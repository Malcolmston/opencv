package text

import (
	"testing"
)

// scoreMatrix builds a per-character score matrix over alphabet from a slice of
// per-position rune->score maps.
func scoreMatrix(alphabet string, positions []map[rune]float64) [][]float64 {
	alpha := []rune(alphabet)
	out := make([][]float64, len(positions))
	for i, pos := range positions {
		row := make([]float64, len(alpha))
		for j, ch := range alpha {
			row[j] = pos[ch]
		}
		out[i] = row
	}
	return out
}

func TestBeamSearchGreedyNoLexicon(t *testing.T) {
	alphabet := "ABCTRO"
	scores := scoreMatrix(alphabet, []map[rune]float64{
		{'C': 0.9, 'O': 0.1},
		{'A': 0.8, 'O': 0.2},
		{'T': 0.7, 'R': 0.3},
	})
	d := NewBeamSearchDecoder(alphabet, 8)
	if got := d.Decode(scores); got != "CAT" {
		t.Errorf("Decode = %q, want CAT", got)
	}
}

func TestBeamSearchLexiconCorrects(t *testing.T) {
	alphabet := "ABCTROZ"
	// Position 2 slightly favours the non-word letter Z over T; the lexicon prunes
	// "CAZ" (not a prefix of any word) and recovers the real word CAT.
	scores := scoreMatrix(alphabet, []map[rune]float64{
		{'C': 0.9},
		{'A': 0.9},
		{'Z': 0.55, 'T': 0.45, 'R': 0.30},
	})
	lex := NewLexicon([]string{"CAT", "CAR", "DOG"})
	d := NewBeamSearchDecoder(alphabet, 8).WithLexicon(lex)
	if got := d.Decode(scores); got != "CAT" {
		t.Errorf("lexicon Decode = %q, want CAT", got)
	}

	// Without the lexicon the greedy result keeps the spurious Z.
	plain := NewBeamSearchDecoder(alphabet, 8)
	if got := plain.Decode(scores); got != "CAZ" {
		t.Errorf("plain Decode = %q, want CAZ", got)
	}
}

func TestBeamSearchLexiconPicksHigherScoringWord(t *testing.T) {
	alphabet := "CARTO"
	// Both CAR and CAT are valid words; scores favour CAR at the last position.
	scores := scoreMatrix(alphabet, []map[rune]float64{
		{'C': 0.9},
		{'A': 0.9},
		{'R': 0.7, 'T': 0.3},
	})
	lex := NewLexicon([]string{"CAT", "CAR"})
	d := NewBeamSearchDecoder(alphabet, 8).WithLexicon(lex)
	if got := d.Decode(scores); got != "CAR" {
		t.Errorf("Decode = %q, want CAR", got)
	}
}

func TestLexiconPrefixAndWord(t *testing.T) {
	lex := NewLexicon([]string{"CAT"})
	if !lex.IsPrefix("") || !lex.IsPrefix("C") || !lex.IsPrefix("CA") || !lex.IsPrefix("CAT") {
		t.Errorf("expected all prefixes of CAT to be recognized")
	}
	if lex.IsPrefix("D") {
		t.Errorf("D should not be a prefix")
	}
	if !lex.IsWord("CAT") || lex.IsWord("CA") {
		t.Errorf("word membership wrong")
	}
}

func TestBeamSearchEndToEndWithOCR(t *testing.T) {
	// Recognize a rendered word by feeding OCR per-character scores through the
	// lexicon-constrained decoder.
	o := NewOCRTemplateAlnum()
	img := RenderText("DOG", 3, 1)
	scores := o.CharScores(img)
	lex := NewLexicon([]string{"CAT", "DOG", "COW"})
	d := NewBeamSearchDecoder(o.Alphabet(), 12).WithLexicon(lex)
	if got := d.Decode(scores); got != "DOG" {
		t.Errorf("end-to-end Decode = %q, want DOG", got)
	}
}
