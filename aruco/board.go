package aruco

import (
	"fmt"
	"math"

	cv "github.com/malcolmston/opencv"
)

// GridBoard is a planar array of ArUco markers laid out on a regular grid, the
// analogue of OpenCV's cv::aruco::GridBoard. All markers come from a single
// [Dictionary], share one physical side length and are separated by a fixed gap.
// A board provides many correspondences at once, so its pose can be recovered
// far more stably than a single marker's (see [EstimatePoseBoard]).
//
// The board lies on its own Z=0 plane. The origin is the board's bottom-left
// corner with +X to the right and +Y up, matching OpenCV's board frame. Markers
// are numbered in reading order: the top-left marker is first, advancing left to
// right then top to bottom, with identifiers starting at the board's first id.
// The zero value is not usable; construct a board with [NewGridBoard].
type GridBoard struct {
	markersX, markersY      int
	markerLength, markerSep float64
	dict                    *Dictionary
	ids                     []int
	objPoints               [][4][3]float64
	boardWidth, boardHeight float64
}

// NewGridBoard builds a markersX-by-markersY grid board. markerLength is each
// marker's physical side length and markerSeparation is the gap between adjacent
// markers, both in the caller's world units. dictionary supplies the markers and
// firstMarker is the identifier of the top-left marker; the remaining markers
// take consecutive identifiers in reading order.
//
// It panics if the grid is empty, the lengths are non-positive, the dictionary
// is nil, or the requested identifiers do not all exist in the dictionary.
func NewGridBoard(markersX, markersY int, markerLength, markerSeparation float64, dictionary *Dictionary, firstMarker int) *GridBoard {
	if markersX <= 0 || markersY <= 0 {
		panic(fmt.Sprintf("aruco: NewGridBoard requires a positive grid, got %dx%d", markersX, markersY))
	}
	if markerLength <= 0 || markerSeparation < 0 {
		panic("aruco: NewGridBoard requires positive markerLength and non-negative markerSeparation")
	}
	if dictionary == nil {
		panic("aruco: NewGridBoard nil dictionary")
	}
	n := markersX * markersY
	if firstMarker < 0 || firstMarker+n > dictionary.Size() {
		panic(fmt.Sprintf("aruco: NewGridBoard needs ids [%d,%d) but dictionary has %d", firstMarker, firstMarker+n, dictionary.Size()))
	}
	b := &GridBoard{
		markersX:     markersX,
		markersY:     markersY,
		markerLength: markerLength,
		markerSep:    markerSeparation,
		dict:         dictionary,
	}
	step := markerLength + markerSeparation
	b.boardWidth = float64(markersX)*markerLength + float64(markersX-1)*markerSeparation
	b.boardHeight = float64(markersY)*markerLength + float64(markersY-1)*markerSeparation
	b.ids = make([]int, 0, n)
	b.objPoints = make([][4][3]float64, 0, n)
	for row := 0; row < markersY; row++ {
		for col := 0; col < markersX; col++ {
			left := float64(col) * step
			top := b.boardHeight - float64(row)*step
			right := left + markerLength
			bottom := top - markerLength
			// Corner order matches DetectMarkers: TL, TR, BR, BL.
			b.objPoints = append(b.objPoints, [4][3]float64{
				{left, top, 0},
				{right, top, 0},
				{right, bottom, 0},
				{left, bottom, 0},
			})
			b.ids = append(b.ids, firstMarker+row*markersX+col)
		}
	}
	return b
}

// MarkersX returns the number of markers along the board's horizontal axis.
func (b *GridBoard) MarkersX() int { return b.markersX }

// MarkersY returns the number of markers along the board's vertical axis.
func (b *GridBoard) MarkersY() int { return b.markersY }

// MarkerLength returns each marker's physical side length in world units.
func (b *GridBoard) MarkerLength() float64 { return b.markerLength }

// MarkerSeparation returns the gap between adjacent markers in world units.
func (b *GridBoard) MarkerSeparation() float64 { return b.markerSep }

// Dictionary returns the dictionary the board's markers are drawn from.
func (b *GridBoard) Dictionary() *Dictionary { return b.dict }

// Ids returns the marker identifiers in board reading order. The returned slice
// is a copy and may be modified freely.
func (b *GridBoard) Ids() []int {
	out := make([]int, len(b.ids))
	copy(out, b.ids)
	return out
}

// ObjectPoints returns, for every board marker in reading order, its four corner
// coordinates in the board frame (Z=0), ordered TL, TR, BR, BL to match the
// corners produced by [DetectMarkers]. The returned slice is a copy.
func (b *GridBoard) ObjectPoints() [][4][3]float64 {
	out := make([][4][3]float64, len(b.objPoints))
	copy(out, b.objPoints)
	return out
}

// objectCornersForID returns the board-frame corners of the marker with the
// given identifier, or ok=false when the board does not contain that id.
func (b *GridBoard) objectCornersForID(id int) ([4][3]float64, bool) {
	for i, mid := range b.ids {
		if mid == id {
			return b.objPoints[i], true
		}
	}
	return [4][3]float64{}, false
}

// Draw renders the board to a fresh single-channel [cv.Mat] of the given pixel
// size with a white margin marginPixels wide on every side. The board is scaled
// uniformly to fit inside the margins so its markers stay square, and centred in
// the image. See [DrawPlanarBoard], which this method wraps.
func (b *GridBoard) Draw(widthPixels, heightPixels, marginPixels int) *cv.Mat {
	return DrawPlanarBoard(b, widthPixels, heightPixels, marginPixels)
}

// DrawPlanarBoard renders board into a new single-channel image of the requested
// pixel dimensions, on a white background, with a margin marginPixels wide on
// every side. The board is scaled by a single factor (so markers remain square)
// to fit the area inside the margins and is centred. Each marker is drawn with
// its correct identifier, black border and cell pattern.
//
// It panics if the dimensions are not positive, the margin does not leave room
// for the board, or the resulting per-marker pixel size is too small to hold a
// marker.
func DrawPlanarBoard(board *GridBoard, widthPixels, heightPixels, marginPixels int) *cv.Mat {
	if board == nil {
		panic("aruco: DrawPlanarBoard nil board")
	}
	if widthPixels <= 0 || heightPixels <= 0 {
		panic("aruco: DrawPlanarBoard requires positive dimensions")
	}
	availW := float64(widthPixels - 2*marginPixels)
	availH := float64(heightPixels - 2*marginPixels)
	if availW <= 0 || availH <= 0 {
		panic("aruco: DrawPlanarBoard margin leaves no room for the board")
	}
	scale := math.Min(availW/board.boardWidth, availH/board.boardHeight)
	usedW := board.boardWidth * scale
	usedH := board.boardHeight * scale
	offX := (float64(widthPixels) - usedW) / 2
	offY := (float64(heightPixels) - usedH) / 2

	sidePx := int(math.Round(board.markerLength * scale))
	cells := board.dict.bitsPerSide + 2
	if sidePx < cells {
		panic(fmt.Sprintf("aruco: DrawPlanarBoard marker renders to %d px, too small for a %d-cell marker", sidePx, cells))
	}

	canvas := cv.NewMat(heightPixels, widthPixels, 1)
	canvas.SetTo(255)
	for i, id := range board.ids {
		left := board.objPoints[i][0][0] // board X of the marker's left edge
		top := board.objPoints[i][0][1]  // board Y of the marker's top edge
		px := int(math.Round(offX + left*scale))
		py := int(math.Round(offY + (board.boardHeight-top)*scale))
		GenerateMarker(board.dict, id, sidePx).CopyTo(canvas, py, px)
	}
	return canvas
}
