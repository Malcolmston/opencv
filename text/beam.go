package text

import (
	"sort"
)

// Lexicon is a fixed vocabulary used to constrain a [BeamSearchDecoder]. It
// stores both the complete words and all their prefixes so the decoder can prune
// any partial hypothesis that cannot extend to a real word.
type Lexicon struct {
	words    map[string]bool
	prefixes map[string]bool
}

// NewLexicon builds a lexicon from a word list. Words are used verbatim (match
// them to your decoder's alphabet case). The empty string is always a valid
// prefix.
func NewLexicon(words []string) *Lexicon {
	lex := &Lexicon{
		words:    make(map[string]bool, len(words)),
		prefixes: map[string]bool{"": true},
	}
	for _, w := range words {
		if w == "" {
			continue
		}
		lex.words[w] = true
		r := []rune(w)
		for i := 1; i <= len(r); i++ {
			lex.prefixes[string(r[:i])] = true
		}
	}
	return lex
}

// IsWord reports whether s is a complete word in the lexicon.
func (l *Lexicon) IsWord(s string) bool { return l.words[s] }

// IsPrefix reports whether s is a prefix of (or equal to) some lexicon word.
func (l *Lexicon) IsPrefix(s string) bool { return l.prefixes[s] }

// BeamSearchDecoder decodes a per-character score matrix into a word by
// beam search, optionally constrained to a [Lexicon]. It mirrors OpenCV's
// OCRBeamSearchDecoder step: the recognizer emits, for each character position, a
// score for every alphabet symbol, and the decoder keeps the BeamWidth
// highest-scoring symbol sequences, summing scores across positions.
//
// Unlike a CTC decoder there is no blank symbol and no repeat collapsing: the
// input already has exactly one column per output character (as produced by
// [OCRTemplate.CharScores]).
type BeamSearchDecoder struct {
	// Alphabet maps score-matrix column index to output rune.
	Alphabet []rune
	// BeamWidth is the number of hypotheses retained at each step (values below 1
	// are treated as 1).
	BeamWidth int
	// Lexicon, when non-nil, prunes partial hypotheses that are not the prefix of
	// any lexicon word and restricts the final answer to complete words.
	Lexicon *Lexicon
}

// NewBeamSearchDecoder returns an unconstrained decoder over alphabet with the
// given beam width.
func NewBeamSearchDecoder(alphabet string, beamWidth int) *BeamSearchDecoder {
	return &BeamSearchDecoder{Alphabet: []rune(alphabet), BeamWidth: beamWidth}
}

// WithLexicon returns a copy of the decoder constrained to lex.
func (d *BeamSearchDecoder) WithLexicon(lex *Lexicon) *BeamSearchDecoder {
	cp := *d
	cp.Lexicon = lex
	return &cp
}

// beam is one hypothesis: a decoded string and its cumulative score.
type beam struct {
	text  string
	score float64
}

// Decode returns the highest-scoring string for the score matrix. scores[t][c] is
// the score of alphabet rune c at character position t; higher is better and
// scores are summed across positions. With a lexicon set, the result is the
// best-scoring complete lexicon word reachable from the scores, falling back to
// the unconstrained best hypothesis when no complete word is reachable.
func (d *BeamSearchDecoder) Decode(scores [][]float64) string {
	width := d.BeamWidth
	if width < 1 {
		width = 1
	}
	beams := []beam{{text: "", score: 0}}
	for _, row := range scores {
		var next []beam
		for _, b := range beams {
			for c, ch := range d.Alphabet {
				if c >= len(row) {
					break
				}
				cand := b.text + string(ch)
				if d.Lexicon != nil && !d.Lexicon.IsPrefix(cand) {
					continue
				}
				next = append(next, beam{text: cand, score: b.score + row[c]})
			}
		}
		if len(next) == 0 {
			// Lexicon pruned everything; fall back to unconstrained expansion so a
			// result is still produced.
			for _, b := range beams {
				for c, ch := range d.Alphabet {
					if c >= len(row) {
						break
					}
					next = append(next, beam{text: b.text + string(ch), score: b.score + row[c]})
				}
			}
		}
		beams = topBeams(next, width)
	}

	if d.Lexicon != nil {
		best := ""
		bestScore := 0.0
		found := false
		for _, b := range beams {
			if !d.Lexicon.IsWord(b.text) {
				continue
			}
			if !found || b.score > bestScore || (b.score == bestScore && b.text < best) {
				best, bestScore, found = b.text, b.score, true
			}
		}
		if found {
			return best
		}
	}
	if len(beams) == 0 {
		return ""
	}
	return beams[0].text
}

// topBeams sorts hypotheses by descending score (ties broken by ascending text
// for determinism) and returns the best width of them.
func topBeams(beams []beam, width int) []beam {
	sort.SliceStable(beams, func(i, j int) bool {
		if beams[i].score != beams[j].score {
			return beams[i].score > beams[j].score
		}
		return beams[i].text < beams[j].text
	})
	if len(beams) > width {
		beams = beams[:width]
	}
	return beams
}
