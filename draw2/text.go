package draw2

import (
	cv "github.com/malcolmston/opencv"
)

// FontWidth is the glyph cell width in font pixels (before scaling).
const FontWidth = draw2fontWidth

// FontHeight is the glyph cell height in font pixels (before scaling).
const FontHeight = draw2fontHeight

// PutText renders text using a built-in 5x7 bitmap font. org is the
// bottom-left corner of the first glyph. scale integer-magnifies each glyph
// pixel (values below 1 are treated as 1). Only runes present in the font are
// drawn; unknown runes advance the cursor as blanks.
func PutText(m *cv.Mat, text string, org cv.Point, scale int, color cv.Scalar) {
	if scale < 1 {
		scale = 1
	}
	cursorX := org.X
	top := org.Y - draw2fontHeight*scale
	for _, r := range text {
		glyph, ok := draw2font5x7[r]
		if !ok {
			cursorX += (draw2fontWidth + 1) * scale
			continue
		}
		for row := 0; row < draw2fontHeight; row++ {
			bits := glyph[row]
			for col := 0; col < draw2fontWidth; col++ {
				if bits&(1<<(draw2fontWidth-1-col)) != 0 {
					for sy := 0; sy < scale; sy++ {
						for sx := 0; sx < scale; sx++ {
							draw2set(m, cursorX+col*scale+sx, top+row*scale+sy, color)
						}
					}
				}
			}
		}
		cursorX += (draw2fontWidth + 1) * scale
	}
}

// TextSize returns the pixel width and height that [PutText] would occupy when
// rendering text at the given scale, matching the cursor advance PutText uses.
// The width includes the one-pixel inter-glyph gap after every glyph except
// none is subtracted for an empty string.
func TextSize(text string, scale int) (width, height int) {
	if scale < 1 {
		scale = 1
	}
	n := 0
	for range text {
		n++
	}
	if n == 0 {
		return 0, 0
	}
	width = n*(draw2fontWidth+1)*scale - scale
	height = draw2fontHeight * scale
	return width, height
}

// PutTextCentered renders text horizontally centred on the point center, with
// center.Y giving the text's vertical middle. It is a convenience wrapper over
// [PutText] and [TextSize].
func PutTextCentered(m *cv.Mat, text string, center cv.Point, scale int, color cv.Scalar) {
	w, h := TextSize(text, scale)
	org := cv.Point{X: center.X - w/2, Y: center.Y + h/2}
	PutText(m, text, org, scale, color)
}
