# aruco — provides square fiducial marker generation and detection on top

[![Go Reference](https://pkg.go.dev/badge/github.com/malcolmston/opencv/aruco.svg)](https://pkg.go.dev/github.com/malcolmston/opencv/aruco)

Package aruco provides square fiducial marker generation and detection on top
of the standard-library-only OpenCV port github.com/malcolmston/opencv
(imported here as cv). It mirrors a useful subset of OpenCV's aruco module:
predefined marker dictionaries, marker image synthesis, robust detection from
a natural image, drawing of detections, planar marker boards, ChArUco boards
and diamonds, tunable detection, and pose estimation for single markers and
whole boards (with optional lens-distortion handling) plus ChArUco camera
calibration.

An ArUco marker is a square grid of black and white cells surrounded by a
solid black border (a one-cell quiet ring). The inner grid encodes an
identifier: a `Dictionary` with N-by-N inner cells is called a "N x N"
dictionary. This package ships predefined dictionaries `Dict4x4` and `Dict5x5`
(via `GetPredefinedDictionary`) and the larger 6x6 families `Dict6x6` and
`Dict6x6Small` (via `GetPredefinedDictionary6x6`); each holds a
deterministically generated family of markers whose bit patterns are mutually
well separated in Hamming distance, both from one another and from their own
90/180/270-degree rotations. That separation is what lets detection recover an
identifier, and the marker's orientation, from a noisy reading. Build a
bespoke family of any size with `GenerateCustomDictionary`.

## Install

```sh
go get github.com/malcolmston/opencv@v0.8.0
```

```go
import "github.com/malcolmston/opencv/aruco"
```

## Usage

This is the package's own `ExampleDetectMarkers`, so it compiles and its output is
asserted on every `go test ./aruco/`.

```go
dict := aruco.GetPredefinedDictionary(aruco.Dict4x4)

	canvas := cv.NewMat(140, 140, 1)
	canvas.SetTo(255)
	aruco.GenerateMarker(dict, 9, 84).CopyTo(canvas, 30, 30)

	corners, ids := aruco.DetectMarkers(canvas, dict)
	fmt.Println("ids:", ids)
	fmt.Println("corners:", len(corners))
```

```
ids: [9]
corners: 1
```

## Exported surface

### Functions

| Function | What it does |
| --- | --- |
| `func CalibrateCameraCharuco(allCharucoCorners [][][2]float64, allCharucoIds [][]int, board *CharucoBoard, imageWidth, imageHeight int) (k [3][3]float64, reprojErr float64, ok bool)` | CalibrateCameraCharuco estimates a pinhole camera's intrinsic matrix from several views of a `CharucoBoard`, the analogue of OpenCV's… |
| `func CornerSubPix(image *cv.Mat, corners [][2]float64, winSize, maxIter int, epsilon float64) [][2]float64` | CornerSubPix refines each of the given image corners to subpixel accuracy by the standard iterative method (OpenCV's cv::cornerSubPix): around each… |
| `func DetectCharucoDiamond(image *cv.Mat, markerCorners [][4]cv.Point, markerIds []int, dict *Dictionary, squareLength, markerLength float64) (diamondCorners [][4]cv.Point, diamondIds [][4]int)` | DetectCharucoDiamond groups already-detected ArUco markers into ChArUco diamonds, the analogue of OpenCV's cv::aruco::detectCharucoDiamond. |
| `func DetectMarkers(img *cv.Mat, dict *Dictionary) (corners [][4]cv.Point, ids []int)` | DetectMarkers finds every marker from dict that appears in img and returns, for each detection, its four image corners and its identifier. |
| `func DetectMarkersWithParams(img *cv.Mat, dict *Dictionary, params DetectorParameters) (corners [][4]cv.Point, ids []int)` | DetectMarkersWithParams is a configurable variant of `DetectMarkers` driven by params. |
| `func DrawAxis(image *cv.Mat, k [3][3]float64, dist []float64, rvec, tvec [3]float64, length float64) *cv.Mat` | DrawAxis returns a three-channel copy of image with the object frame's coordinate axes drawn from its origin, the analogue of OpenCV's… |
| `func DrawCharucoBoard(board *CharucoBoard, widthPixels, heightPixels, marginPixels int) *cv.Mat` | DrawCharucoBoard renders board into a new single-channel image of the given pixel dimensions, on a white background, with a margin marginPixels wide… |
| `func DrawCharucoDiamond(dict *Dictionary, ids [4]int, squareLength, markerLength int) *cv.Mat` | DrawCharucoDiamond renders a ChArUco "diamond": a 3x3 chessboard carrying four ArUco markers in its white squares, the analogue of OpenCV's… |
| `func DrawDetectedCornersCharuco(image *cv.Mat, charucoCorners [][2]float64, charucoIds []int) *cv.Mat` | DrawDetectedCornersCharuco returns a three-channel copy of image with every interpolated ChArUco chessboard corner marked, the analogue of OpenCV's… |
| `func DrawDetectedDiamonds(image *cv.Mat, diamondCorners [][4]cv.Point, diamondIds [][4]int) *cv.Mat` | DrawDetectedDiamonds returns a three-channel copy of image with every detected ChArUco diamond outlined, the analogue of OpenCV's… |
| `func DrawDetectedMarkers(img *cv.Mat, corners [][4]cv.Point, ids []int) *cv.Mat` | DrawDetectedMarkers returns a three-channel copy of img with every detection overlaid: each marker's four edges are drawn as a green quadrilateral,… |
| `func DrawPlanarBoard(board *GridBoard, widthPixels, heightPixels, marginPixels int) *cv.Mat` | DrawPlanarBoard renders board into a new single-channel image of the requested pixel dimensions, on a white background, with a margin marginPixels… |
| `func EstimatePoseBoard(corners [][4]cv.Point, ids []int, board *GridBoard, k [3][3]float64, dist []float64) (rvec, tvec [3]float64, markersUsed int)` | EstimatePoseBoard recovers a single camera pose for a whole `GridBoard` from however many of its markers were detected. |
| `func EstimatePoseCharucoBoard(charucoCorners [][2]float64, charucoIds []int, board *CharucoBoard, k [3][3]float64, dist []float64) (rvec, tvec [3]float64, ok bool)` | EstimatePoseCharucoBoard recovers the camera pose from interpolated ChArUco chessboard corners (as returned by `InterpolateCornersCharuco`). |
| `func EstimatePoseSingleMarkers(corners [][4]cv.Point, markerLength float64, k [3][3]float64, dist []float64) (rvecs, tvecs [][3]float64)` | EstimatePoseSingleMarkers recovers an approximate camera pose for each detected marker. |
| `func EstimatePoseSingleMarkersWithDistortion(corners [][4]cv.Point, markerLength float64, k [3][3]float64, dist []float64) (rvecs, tvecs [][3]float64)` | EstimatePoseSingleMarkersWithDistortion is a distortion-aware companion to `EstimatePoseSingleMarkers`. |
| `func GenerateMarker(dict *Dictionary, id, sidePixels int) *cv.Mat` | GenerateMarker renders the marker with the given id from dict into a new single-channel `cv.Mat` whose side is sidePixels pixels. |
| `func GetBoardObjectAndImagePoints(board *GridBoard, corners [][4]cv.Point, ids []int) (objPoints [][3]float64, imgPoints [][2]float64)` | GetBoardObjectAndImagePoints pairs a `GridBoard`'s known corner coordinates with the image corners that were actually detected, the analogue of… |
| `func InterpolateCornersCharuco(markerCorners [][4]cv.Point, markerIds []int, image *cv.Mat, board *CharucoBoard) (charucoCorners [][2]float64, charucoIds []int)` | InterpolateCornersCharuco recovers the ChArUco chessboard corners that are pinned down by the detected markers. |
| `func ProjectPoints(objectPoints [][3]float64, rvec, tvec [3]float64, k [3][3]float64, dist []float64) (imagePoints [][2]float64, ok []bool)` | ProjectPoints projects 3D object points into the image, the analogue of OpenCV's cv::projectPoints. |
| `func RefineDetectedMarkers(image *cv.Mat, board *GridBoard, detectedCorners [][4]cv.Point, detectedIds []int) (corners [][4]cv.Point, ids []int, recovered int)` | RefineDetectedMarkers recovers board markers that the initial detection missed. |
| `func UndistortImagePoints(points [][2]float64, k [3][3]float64, dist []float64) [][2]float64` | UndistortImagePoints removes lens distortion from a set of pixel observations given the camera intrinsic matrix k and Brown-Conrady distortion… |

### Types

| Type | What it is |
| --- | --- |
| `CharucoBoard` | CharucoBoard is a chessboard whose white squares each carry an ArUco marker, the analogue of OpenCV's cv::aruco::CharucoBoard. |
| `CornerRefinementMethod` | CornerRefinementMethod selects how `DetectMarkersWithParams` refines the detected marker corners. |
| `DetectorParameters` | DetectorParameters tunes the marker-detection pipeline of `DetectMarkersWithParams`. |
| `Dictionary` | Dictionary is a family of markers that share a fixed inner grid size. |
| `GridBoard` | GridBoard is a planar array of ArUco markers laid out on a regular grid, the analogue of OpenCV's cv::aruco::GridBoard. |
| `PredefinedDictionaryName` | PredefinedDictionaryName selects one of the dictionaries shipped with the package for `GetPredefinedDictionary`. |

<details>
<summary><code>CharucoBoard</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewCharucoBoard(squaresX, squaresY int, squareLength, markerLength float64, dictionary *Dictionary) *CharucoBoard` | NewCharucoBoard builds a squaresX-by-squaresY ChArUco board. |
| `func (b *CharucoBoard) ChessboardCorners() [][3]float64` | ChessboardCorners returns the board-frame coordinates (Z=0) of every interior chessboard corner, indexed by corner identifier. |
| `func (b *CharucoBoard) Dictionary() *Dictionary` | Dictionary returns the dictionary the board's markers are drawn from. |
| `func (b *CharucoBoard) Draw(widthPixels, heightPixels, marginPixels int) *cv.Mat` | Draw renders the board to a fresh single-channel image, wrapping `DrawCharucoBoard`. |
| `func (b *CharucoBoard) Ids() []int` | Ids returns the marker identifiers embedded in the board's white squares, in reading order. |
| `func (b *CharucoBoard) MarkerLength() float64` | MarkerLength returns the embedded ArUco marker's physical side length. |
| `func (b *CharucoBoard) SquareLength() float64` | SquareLength returns a chessboard square's physical side length. |
| `func (b *CharucoBoard) SquaresX() int` | SquaresX returns the number of chessboard squares along the horizontal axis. |
| `func (b *CharucoBoard) SquaresY() int` | SquaresY returns the number of chessboard squares along the vertical axis. |

</details>

<details>
<summary><code>DetectorParameters</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func DefaultDetectorParameters() DetectorParameters` | DefaultDetectorParameters returns detector parameters equivalent to the package's built-in `DetectMarkers` behaviour: a moderate adaptive-threshold… |

</details>

<details>
<summary><code>Dictionary</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func GenerateCustomDictionary(bitsPerSide, count, minDistance int, seed uint64) *Dictionary` | GenerateCustomDictionary deterministically builds a fresh dictionary of up to count markers with a bitsPerSide-by-bitsPerSide inner grid, where… |
| `func GetPredefinedDictionary(name PredefinedDictionaryName) *Dictionary` | GetPredefinedDictionary returns the requested predefined dictionary. |
| `func GetPredefinedDictionary6x6(name PredefinedDictionaryName) *Dictionary` | GetPredefinedDictionary6x6 returns one of the 6x6 predefined dictionaries (`Dict6x6` or `Dict6x6Small`). |
| `func (d *Dictionary) BitsPerSide() int` | BitsPerSide returns the number of inner cells along one side of a marker in this dictionary. |
| `func (d *Dictionary) GetDistanceToId(bits []byte, id int) (distance int, ok bool)` | GetDistanceToId returns the smallest Hamming distance, in cells, between the given bit reading and any 90-degree rotation of the marker with the… |
| `func (d *Dictionary) IdentifyMarker(bits []byte) (id, rotation int, ok bool)` | IdentifyMarker matches a bit reading against every marker in the dictionary under all four rotations, mirroring OpenCV's… |
| `func (d *Dictionary) MarkerBits(id int) (bits []byte, ok bool)` | MarkerBits returns the canonical (unrotated) inner bit grid of the marker with the given id as a flat BitsPerSide()*BitsPerSide() slice in row-major… |
| `func (d *Dictionary) Size() int` | Size returns the number of distinct markers (identifiers) in the dictionary. |
| `func (d *Dictionary) Tolerance() int` | Tolerance returns the maximum Hamming distance, in cells, at which a cell reading is still accepted as a match for a dictionary marker. |

</details>

<details>
<summary><code>GridBoard</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewGridBoard(markersX, markersY int, markerLength, markerSeparation float64, dictionary *Dictionary, firstMarker int) *GridBoard` | NewGridBoard builds a markersX-by-markersY grid board. |
| `func (b *GridBoard) Dictionary() *Dictionary` | Dictionary returns the dictionary the board's markers are drawn from. |
| `func (b *GridBoard) Draw(widthPixels, heightPixels, marginPixels int) *cv.Mat` | Draw renders the board to a fresh single-channel `cv.Mat` of the given pixel size with a white margin marginPixels wide on every side. |
| `func (b *GridBoard) Ids() []int` | Ids returns the marker identifiers in board reading order. |
| `func (b *GridBoard) MarkerLength() float64` | MarkerLength returns each marker's physical side length in world units. |
| `func (b *GridBoard) MarkerSeparation() float64` | MarkerSeparation returns the gap between adjacent markers in world units. |
| `func (b *GridBoard) MarkersX() int` | MarkersX returns the number of markers along the board's horizontal axis. |
| `func (b *GridBoard) MarkersY() int` | MarkersY returns the number of markers along the board's vertical axis. |
| `func (b *GridBoard) ObjectPoints() [][4][3]float64` | ObjectPoints returns, for every board marker in reading order, its four corner coordinates in the board frame (Z=0), ordered TL, TR, BR, BL to match… |

</details>

Full signatures, doc comments and every runnable example are on
[pkg.go.dev](https://pkg.go.dev/github.com/malcolmston/opencv/aruco).

## License

MIT, as part of [`github.com/malcolmston/opencv`](..).
An independent re-implementation, not affiliated with or endorsed by the
original project.
