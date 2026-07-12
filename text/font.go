package text

import (
	"sort"

	cv "github.com/malcolmston/opencv"
)

// glyphW and glyphH are the native cell size of the built-in bitmap font.
const (
	glyphW = 5
	glyphH = 7
)

// fontBitmaps holds the built-in 5x7 bitmap font: a legible fixed-width face
// covering the digits 0-9 and the uppercase letters A-Z. Each glyph is seven rows
// of five characters, '#' inked and '.' blank. It exists so the recognizer and
// tests can render and read text without any external font file.
var fontBitmaps = map[rune][7]string{
	'0': {".###.", "#...#", "#..##", "#.#.#", "##..#", "#...#", ".###."},
	'1': {"..#..", ".##..", "..#..", "..#..", "..#..", "..#..", ".###."},
	'2': {".###.", "#...#", "....#", "...#.", "..#..", ".#...", "#####"},
	'3': {"#####", "...#.", "..#..", "...#.", "....#", "#...#", ".###."},
	'4': {"...#.", "..##.", ".#.#.", "#..#.", "#####", "...#.", "...#."},
	'5': {"#####", "#....", "####.", "....#", "....#", "#...#", ".###."},
	'6': {"..##.", ".#...", "#....", "####.", "#...#", "#...#", ".###."},
	'7': {"#####", "....#", "...#.", "..#..", ".#...", ".#...", ".#..."},
	'8': {".###.", "#...#", "#...#", ".###.", "#...#", "#...#", ".###."},
	'9': {".###.", "#...#", "#...#", ".####", "....#", "...#.", ".##.."},
	'A': {".###.", "#...#", "#...#", "#####", "#...#", "#...#", "#...#"},
	'B': {"####.", "#...#", "#...#", "####.", "#...#", "#...#", "####."},
	'C': {".###.", "#...#", "#....", "#....", "#....", "#...#", ".###."},
	'D': {"###..", "#..#.", "#...#", "#...#", "#...#", "#..#.", "###.."},
	'E': {"#####", "#....", "#....", "####.", "#....", "#....", "#####"},
	'F': {"#####", "#....", "#....", "####.", "#....", "#....", "#...."},
	'G': {".###.", "#...#", "#....", "#.###", "#...#", "#...#", ".###."},
	'H': {"#...#", "#...#", "#...#", "#####", "#...#", "#...#", "#...#"},
	'I': {".###.", "..#..", "..#..", "..#..", "..#..", "..#..", ".###."},
	'J': {"..###", "...#.", "...#.", "...#.", "...#.", "#..#.", ".##.."},
	'K': {"#...#", "#..#.", "#.#..", "##...", "#.#..", "#..#.", "#...#"},
	'L': {"#....", "#....", "#....", "#....", "#....", "#....", "#####"},
	'M': {"#...#", "##.##", "#.#.#", "#.#.#", "#...#", "#...#", "#...#"},
	'N': {"#...#", "#...#", "##..#", "#.#.#", "#..##", "#...#", "#...#"},
	'O': {".###.", "#...#", "#...#", "#...#", "#...#", "#...#", ".###."},
	'P': {"####.", "#...#", "#...#", "####.", "#....", "#....", "#...."},
	'Q': {".###.", "#...#", "#...#", "#...#", "#.#.#", "#..#.", ".##.#"},
	'R': {"####.", "#...#", "#...#", "####.", "#.#..", "#..#.", "#...#"},
	'S': {".###.", "#...#", "#....", ".###.", "....#", "#...#", ".###."},
	'T': {"#####", "..#..", "..#..", "..#..", "..#..", "..#..", "..#.."},
	'U': {"#...#", "#...#", "#...#", "#...#", "#...#", "#...#", ".###."},
	'V': {"#...#", "#...#", "#...#", "#...#", "#...#", ".#.#.", "..#.."},
	'W': {"#...#", "#...#", "#...#", "#.#.#", "#.#.#", "##.##", "#...#"},
	'X': {"#...#", "#...#", ".#.#.", "..#..", ".#.#.", "#...#", "#...#"},
	'Y': {"#...#", "#...#", ".#.#.", "..#..", "..#..", "..#..", "..#.."},
	'Z': {"#####", "....#", "...#.", "..#..", ".#...", "#....", "#####"},
}

// SupportedChars returns the characters the built-in font can render and the
// [OCRTemplate] recognizers can read, in ascending rune order (0-9 then A-Z).
func SupportedChars() string {
	runes := make([]rune, 0, len(fontBitmaps))
	for r := range fontBitmaps {
		runes = append(runes, r)
	}
	sort.Slice(runes, func(i, j int) bool { return runes[i] < runes[j] })
	return string(runes)
}

// FontGlyph renders one character of the built-in font as a single-channel Mat at
// the given integer scale (each font pixel becomes a scale×scale block), inked
// pixels set to 255 on a 0 background. It reports ok=false for an unsupported
// character or a non-positive scale.
func FontGlyph(ch rune, scale int) (glyph *cv.Mat, ok bool) {
	bmp, found := fontBitmaps[ch]
	if !found || scale <= 0 {
		return nil, false
	}
	out := cv.NewMat(glyphH*scale, glyphW*scale, 1)
	for gy := 0; gy < glyphH; gy++ {
		row := bmp[gy]
		for gx := 0; gx < glyphW; gx++ {
			if row[gx] != '#' {
				continue
			}
			for sy := 0; sy < scale; sy++ {
				for sx := 0; sx < scale; sx++ {
					out.Set(gy*scale+sy, gx*scale+sx, 0, 255)
				}
			}
		}
	}
	return out, true
}

// RenderText draws a line of built-in-font text into a fresh single-channel Mat:
// inked glyph pixels are 255 on a 0 background, glyphs are scaled by scale and
// separated (and margined) by spacing font-pixels' worth of blank columns. Unknown
// characters and spaces render as a blank glyph-width gap, so segmentation still
// sees word boundaries. It panics if scale or spacing is negative or scale is 0.
//
// RenderText is the inverse of the [OCRTemplate] recognizers and is meant for
// building deterministic fixtures and examples.
func RenderText(text string, scale, spacing int) *cv.Mat {
	if scale <= 0 || spacing < 0 {
		panic("text: RenderText requires scale > 0 and spacing >= 0")
	}
	gap := spacing * scale
	cellW := glyphW * scale
	cellH := glyphH * scale
	runes := []rune(text)
	// Width: leading gap, each glyph cell, a gap after each glyph.
	width := gap + len(runes)*(cellW+gap)
	if width < 1 {
		width = 1
	}
	height := cellH + 2*gap
	if height < 1 {
		height = 1
	}
	out := cv.NewMat(height, width, 1)

	x := gap
	y := gap
	for _, ch := range runes {
		if glyph, ok := FontGlyph(ch, scale); ok {
			glyph.CopyTo(out, y, x)
		}
		x += cellW + gap
	}
	return out
}
