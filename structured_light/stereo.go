package structured_light

// StereoMatch is one camera-to-camera correspondence recovered by decoding the
// same projector patterns in two cameras: the left and right camera pixels that
// both saw the same projector pixel. Such matches are the input to stereo
// triangulation, and unlike raw block matching they are dense and unambiguous
// because the projector coordinate is a global label.
type StereoMatch struct {
	// LeftX, LeftY is the pixel in the left camera.
	LeftX, LeftY int
	// RightX, RightY is the pixel in the right camera.
	RightX, RightY int
	// Col, Row is the shared projector coordinate.
	Col, Row int
}

// StereoDecode converts two independently-decoded correspondence fields (from
// the left and right cameras viewing the same projector sequence) into a list of
// left↔right pixel matches that share a projector coordinate. It builds an index
// from projector (Col,Row) to the first valid right-camera pixel — scanning the
// right field in row-major order so the choice is deterministic — then emits a
// match for every valid left-camera pixel whose projector coordinate is present.
// The matches are returned in left-camera row-major order.
//
// This is the projector-as-common-reference form of two-camera structured light:
// the two cameras need not be rectified and no window search is performed.
func StereoDecode(left, right *Decoded) []StereoMatch {
	index := make(map[[2]int][2]int, right.Rows*right.Cols)
	for y := 0; y < right.Rows; y++ {
		for x := 0; x < right.Cols; x++ {
			i := y*right.Cols + x
			if !right.Mask[i] {
				continue
			}
			key := [2]int{right.Col[i], right.Row[i]}
			if _, seen := index[key]; !seen {
				index[key] = [2]int{x, y}
			}
		}
	}

	var matches []StereoMatch
	for y := 0; y < left.Rows; y++ {
		for x := 0; x < left.Cols; x++ {
			i := y*left.Cols + x
			if !left.Mask[i] {
				continue
			}
			key := [2]int{left.Col[i], left.Row[i]}
			if rp, ok := index[key]; ok {
				matches = append(matches, StereoMatch{
					LeftX: x, LeftY: y,
					RightX: rp[0], RightY: rp[1],
					Col: key[0], Row: key[1],
				})
			}
		}
	}
	return matches
}

// TriangulateStereo reconstructs world points from [StereoMatch]es using two
// calibrated camera projection matrices. Each match's left and right pixels are
// the two views passed to [TriangulatePoint]. The returned [PointCloud] records
// each point together with its left-camera pixel.
func TriangulateStereo(matches []StereoMatch, leftCam, rightCam CameraMatrix) *PointCloud {
	pc := &PointCloud{}
	for _, m := range matches {
		p := TriangulatePoint(leftCam, rightCam, float64(m.LeftX), float64(m.LeftY), float64(m.RightX), float64(m.RightY))
		pc.Points = append(pc.Points, p)
		pc.PixelX = append(pc.PixelX, m.LeftX)
		pc.PixelY = append(pc.PixelY, m.LeftY)
	}
	return pc
}
