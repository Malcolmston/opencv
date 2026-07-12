package aruco

import (
	"fmt"
	"math"

	cv "github.com/malcolmston/opencv"
)

// CharucoBoard is a chessboard whose white squares each carry an ArUco marker,
// the analogue of OpenCV's cv::aruco::CharucoBoard. It combines the two
// techniques: the markers make the board identifiable and give a coarse pose
// even under occlusion, while the black/white chessboard corners can be
// localised to subpixel accuracy for a precise pose and for camera calibration.
//
// The board lies on its Z=0 plane with the origin at the bottom-left corner,
// +X right and +Y up. Squares are indexed by column (0..SquaresX-1, left to
// right) and row (0..SquaresY-1, top to bottom); the top-left square is black
// and colours alternate. Markers fill the white squares in reading order with
// identifiers 0, 1, 2, ... Interior chessboard corners are numbered row by row
// from the top-left interior corner. The zero value is not usable; build a board
// with [NewCharucoBoard].
type CharucoBoard struct {
	squaresX, squaresY      int
	squareLength            float64
	markerLength            float64
	dict                    *Dictionary
	boardWidth, boardHeight float64

	markerIds []int
	markerObj [][4][3]float64
	// squareOfMarker[k] is the (col,row) of the white square holding marker k.
	squareOfMarker [][2]int

	// chessObj[id] is the board-frame coordinate of interior corner id.
	chessObj [][3]float64
}

// NewCharucoBoard builds a squaresX-by-squaresY ChArUco board. squareLength is
// the chessboard square's physical side length and markerLength is the ArUco
// marker's side length (which must be smaller, so a white quiet ring surrounds
// each marker), both in world units. dictionary must hold at least as many
// markers as the board has white squares.
//
// It panics if the board is smaller than 2x2, the lengths are non-positive,
// markerLength is not smaller than squareLength, the dictionary is nil, or the
// dictionary has too few markers.
func NewCharucoBoard(squaresX, squaresY int, squareLength, markerLength float64, dictionary *Dictionary) *CharucoBoard {
	if squaresX < 2 || squaresY < 2 {
		panic(fmt.Sprintf("aruco: NewCharucoBoard requires at least 2x2 squares, got %dx%d", squaresX, squaresY))
	}
	if squareLength <= 0 || markerLength <= 0 {
		panic("aruco: NewCharucoBoard requires positive lengths")
	}
	if markerLength >= squareLength {
		panic("aruco: NewCharucoBoard requires markerLength < squareLength")
	}
	if dictionary == nil {
		panic("aruco: NewCharucoBoard nil dictionary")
	}
	b := &CharucoBoard{
		squaresX:     squaresX,
		squaresY:     squaresY,
		squareLength: squareLength,
		markerLength: markerLength,
		dict:         dictionary,
		boardWidth:   float64(squaresX) * squareLength,
		boardHeight:  float64(squaresY) * squareLength,
	}
	// Markers fill the white squares (where col+row is odd) in reading order.
	half := markerLength / 2
	for row := 0; row < squaresY; row++ {
		for col := 0; col < squaresX; col++ {
			if (col+row)%2 == 0 {
				continue // black square
			}
			cx := (float64(col) + 0.5) * squareLength
			cy := b.boardHeight - (float64(row)+0.5)*squareLength
			b.markerObj = append(b.markerObj, [4][3]float64{
				{cx - half, cy + half, 0},
				{cx + half, cy + half, 0},
				{cx + half, cy - half, 0},
				{cx - half, cy - half, 0},
			})
			b.squareOfMarker = append(b.squareOfMarker, [2]int{col, row})
		}
	}
	if dictionary.Size() < len(b.markerObj) {
		panic(fmt.Sprintf("aruco: NewCharucoBoard needs %d markers but dictionary has %d", len(b.markerObj), dictionary.Size()))
	}
	b.markerIds = make([]int, len(b.markerObj))
	for i := range b.markerIds {
		b.markerIds[i] = i
	}
	// Interior chessboard corners, numbered row by row from the top-left.
	for j := 1; j < squaresY; j++ {
		for i := 1; i < squaresX; i++ {
			x := float64(i) * squareLength
			y := b.boardHeight - float64(j)*squareLength
			b.chessObj = append(b.chessObj, [3]float64{x, y, 0})
		}
	}
	return b
}

// SquaresX returns the number of chessboard squares along the horizontal axis.
func (b *CharucoBoard) SquaresX() int { return b.squaresX }

// SquaresY returns the number of chessboard squares along the vertical axis.
func (b *CharucoBoard) SquaresY() int { return b.squaresY }

// SquareLength returns a chessboard square's physical side length.
func (b *CharucoBoard) SquareLength() float64 { return b.squareLength }

// MarkerLength returns the embedded ArUco marker's physical side length.
func (b *CharucoBoard) MarkerLength() float64 { return b.markerLength }

// Dictionary returns the dictionary the board's markers are drawn from.
func (b *CharucoBoard) Dictionary() *Dictionary { return b.dict }

// Ids returns the marker identifiers embedded in the board's white squares, in
// reading order. The returned slice is a copy.
func (b *CharucoBoard) Ids() []int {
	out := make([]int, len(b.markerIds))
	copy(out, b.markerIds)
	return out
}

// ChessboardCorners returns the board-frame coordinates (Z=0) of every interior
// chessboard corner, indexed by corner identifier. There are
// (SquaresX-1)*(SquaresY-1) of them. The returned slice is a copy.
func (b *CharucoBoard) ChessboardCorners() [][3]float64 {
	out := make([][3]float64, len(b.chessObj))
	copy(out, b.chessObj)
	return out
}

// chessboardCornerForID returns the board coordinate of interior corner id and
// whether id is in range.
func (b *CharucoBoard) chessboardCornerForID(id int) ([3]float64, bool) {
	if id < 0 || id >= len(b.chessObj) {
		return [3]float64{}, false
	}
	return b.chessObj[id], true
}

// markerObjectCornersForID returns the board corners of the marker with the
// given id (its position in the board's white-square reading order).
func (b *CharucoBoard) markerObjectCornersForID(id int) ([4][3]float64, bool) {
	if id < 0 || id >= len(b.markerObj) {
		return [4][3]float64{}, false
	}
	return b.markerObj[id], true
}

// Draw renders the board to a fresh single-channel image, wrapping
// [DrawCharucoBoard].
func (b *CharucoBoard) Draw(widthPixels, heightPixels, marginPixels int) *cv.Mat {
	return DrawCharucoBoard(b, widthPixels, heightPixels, marginPixels)
}

// DrawCharucoBoard renders board into a new single-channel image of the given
// pixel dimensions, on a white background, with a margin marginPixels wide on
// every side. The board is scaled uniformly to fit inside the margins and
// centred: black squares are filled, and each white square carries its ArUco
// marker, centred with a white quiet ring.
//
// It panics if the dimensions are not positive, the margin leaves no room, or a
// marker would render too small to be read.
func DrawCharucoBoard(board *CharucoBoard, widthPixels, heightPixels, marginPixels int) *cv.Mat {
	if board == nil {
		panic("aruco: DrawCharucoBoard nil board")
	}
	if widthPixels <= 0 || heightPixels <= 0 {
		panic("aruco: DrawCharucoBoard requires positive dimensions")
	}
	availW := float64(widthPixels - 2*marginPixels)
	availH := float64(heightPixels - 2*marginPixels)
	if availW <= 0 || availH <= 0 {
		panic("aruco: DrawCharucoBoard margin leaves no room for the board")
	}
	scale := math.Min(availW/board.boardWidth, availH/board.boardHeight)
	offX := (float64(widthPixels) - board.boardWidth*scale) / 2
	offY := (float64(heightPixels) - board.boardHeight*scale) / 2

	sidePx := int(math.Round(board.markerLength * scale))
	cells := board.dict.bitsPerSide + 2
	if sidePx < cells {
		panic(fmt.Sprintf("aruco: DrawCharucoBoard marker renders to %d px, too small for a %d-cell marker", sidePx, cells))
	}

	canvas := cv.NewMat(heightPixels, widthPixels, 1)
	canvas.SetTo(255)
	black := cv.NewScalar(0)
	// Fill the black squares.
	for row := 0; row < board.squaresY; row++ {
		for col := 0; col < board.squaresX; col++ {
			if (col+row)%2 != 0 {
				continue
			}
			x0 := int(math.Round(offX + float64(col)*board.squareLength*scale))
			y0 := int(math.Round(offY + float64(row)*board.squareLength*scale))
			x1 := int(math.Round(offX + float64(col+1)*board.squareLength*scale))
			y1 := int(math.Round(offY + float64(row+1)*board.squareLength*scale))
			cv.Rectangle(canvas, cv.Point{X: x0, Y: y0}, cv.Point{X: x1 - 1, Y: y1 - 1}, black, cv.Filled)
		}
	}
	// Paste markers into the white squares.
	for k, id := range board.markerIds {
		obj := board.markerObj[k]
		left := obj[0][0] // board X of marker left edge
		top := obj[0][1]  // board Y of marker top edge
		px := int(math.Round(offX + left*scale))
		py := int(math.Round(offY + (board.boardHeight-top)*scale))
		GenerateMarker(board.dict, id, sidePx).CopyTo(canvas, py, px)
	}
	return canvas
}

// InterpolateCornersCharuco recovers the ChArUco chessboard corners that are
// pinned down by the detected markers. markerCorners and markerIds are the
// parallel slices from [DetectMarkers] run with board's dictionary; image is the
// picture they were found in (used to reject corners projected outside its
// bounds and may be nil to skip that check).
//
// From the detected markers it fits the board-to-image homography and projects
// every interior chessboard corner through it, returning the resulting subpixel
// image points charucoCorners together with their corner identifiers
// charucoIds. Because the board is planar, the projection is exact under an
// ideal pinhole camera, giving corner locations more precise than any raw marker
// corner. At least one board marker must be detected; otherwise both results are
// nil.
func InterpolateCornersCharuco(markerCorners [][4]cv.Point, markerIds []int, image *cv.Mat, board *CharucoBoard) (charucoCorners [][2]float64, charucoIds []int) {
	if board == nil {
		return nil, nil
	}
	var src, dst [][2]float64
	for i, id := range markerIds {
		if i >= len(markerCorners) {
			break
		}
		obj, ok := board.markerObjectCornersForID(id)
		if !ok {
			continue
		}
		for j := 0; j < 4; j++ {
			src = append(src, [2]float64{obj[j][0], obj[j][1]})
			dst = append(dst, [2]float64{float64(markerCorners[i][j].X), float64(markerCorners[i][j].Y)})
		}
	}
	if len(src) < 4 {
		return nil, nil
	}
	h, ok := homographyFromPoints(src, dst)
	if !ok {
		return nil, nil
	}
	for id, obj := range board.chessObj {
		u, v, ok := applyH(h, obj[0], obj[1])
		if !ok {
			continue
		}
		if image != nil {
			if u < 0 || v < 0 || u > float64(image.Cols-1) || v > float64(image.Rows-1) {
				continue
			}
		}
		charucoCorners = append(charucoCorners, [2]float64{u, v})
		charucoIds = append(charucoIds, id)
	}
	if len(charucoCorners) == 0 {
		return nil, nil
	}
	return charucoCorners, charucoIds
}
