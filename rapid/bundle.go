package rapid

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// grayAt returns the intensity of pixel (x, y) of img, averaging channels for
// multi-channel images and clamping coordinates to the border (BORDER_REPLICATE).
func grayAt(img *cv.Mat, x, y int) float64 {
	if x < 0 {
		x = 0
	} else if x >= img.Cols {
		x = img.Cols - 1
	}
	if y < 0 {
		y = 0
	} else if y >= img.Rows {
		y = img.Rows - 1
	}
	if img.Channels == 1 {
		return float64(img.At(y, x, 0))
	}
	var sum float64
	for c := 0; c < img.Channels; c++ {
		sum += float64(img.At(y, x, c))
	}
	return sum / float64(img.Channels)
}

// ExtractLineBundle reads a one-dimensional intensity profile of length
// 2*length+1 along each control point's search normal into a bundle image. Row i
// of the returned single-channel bundle corresponds to control point i; column
// length is centred on the control point, columns below it lie on the negative
// side of the normal and columns above on the positive side.
//
// srcLocations records, for every bundle sample, the integer pixel coordinate it
// was read from, so that a found bundle column can be converted back to an image
// point by [ConvertCorrespondencies].
func ExtractLineBundle(length int, ctl []ControlPoint, img *cv.Mat) (bundle *cv.Mat, srcLocations [][]cv.Point) {
	width := 2*length + 1
	rows := len(ctl)
	if rows == 0 || length <= 0 {
		return nil, nil
	}
	bundle = cv.NewMat(rows, width, 1)
	srcLocations = make([][]cv.Point, rows)
	for i, cp := range ctl {
		locs := make([]cv.Point, width)
		for j := 0; j < width; j++ {
			off := float64(j - length)
			x := int(math.Round(cp.Image.X + off*cp.Normal.X))
			y := int(math.Round(cp.Image.Y + off*cp.Normal.Y))
			locs[j] = cv.Point{X: x, Y: y}
			bundle.Set(i, j, 0, clampToByte(grayAt(img, x, y)))
		}
		srcLocations[i] = locs
	}
	return bundle, srcLocations
}

// clampToByte rounds v to the nearest byte, clamping to [0, 255].
func clampToByte(v float64) uint8 {
	v = math.Round(v)
	if v < 0 {
		return 0
	}
	if v > 255 {
		return 255
	}
	return uint8(v)
}

// FindCorrespondencies locates, for every row of the bundle, the column of the
// strongest intensity edge. The edge response is the magnitude of a central
// difference along the row; the returned cols[i] is the argmax column and
// response[i] its magnitude. Rows shorter than three samples yield a centred
// column with zero response.
func FindCorrespondencies(bundle *cv.Mat) (cols []int, response []float64) {
	if bundle == nil || bundle.Rows == 0 {
		return nil, nil
	}
	w := bundle.Cols
	cols = make([]int, bundle.Rows)
	response = make([]float64, bundle.Rows)
	for i := 0; i < bundle.Rows; i++ {
		best := -1.0
		bestCol := w / 2
		for j := 0; j < w; j++ {
			var g float64
			switch {
			case j == 0:
				g = math.Abs(float64(bundle.At(i, 1, 0)) - float64(bundle.At(i, 0, 0)))
			case j == w-1:
				g = math.Abs(float64(bundle.At(i, w-1, 0)) - float64(bundle.At(i, w-2, 0)))
			default:
				g = math.Abs(float64(bundle.At(i, j+1, 0)) - float64(bundle.At(i, j-1, 0)))
			}
			if g > best {
				best = g
				bestCol = j
			}
		}
		cols[i] = bestCol
		response[i] = best
	}
	return cols, response
}

// Correspondence pairs a matched 2D image edge point with its 3D model point
// and the search normal along which the match was found.
type Correspondence struct {
	// Image is the located edge point in the image.
	Image Point2f
	// Object is the corresponding 3D model point.
	Object [3]float64
	// Normal is the unit search normal used for this control point.
	Normal Point2f
}

// ConvertCorrespondencies converts found bundle columns into
// [Correspondence] pairs. For each control point it looks up the pixel that the
// found column maps to (via srcLocations) and keeps it when the edge response is
// at least minResponse. The returned mask marks which input control points were
// accepted, in input order.
func ConvertCorrespondencies(cols []int, response []float64, srcLocations [][]cv.Point, ctl []ControlPoint, minResponse float64) (matches []Correspondence, mask []bool) {
	n := len(ctl)
	mask = make([]bool, n)
	for i := 0; i < n; i++ {
		if i >= len(cols) || i >= len(srcLocations) {
			break
		}
		if response[i] < minResponse {
			continue
		}
		c := cols[i]
		if c < 0 || c >= len(srcLocations[i]) {
			continue
		}
		loc := srcLocations[i][c]
		mask[i] = true
		matches = append(matches, Correspondence{
			Image:  Point2f{X: float64(loc.X), Y: float64(loc.Y)},
			Object: ctl[i].Object,
			Normal: ctl[i].Normal,
		})
	}
	return matches, mask
}
