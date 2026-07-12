package structured_light

// CameraMatrix is a 3×4 pinhole projection matrix P mapping a homogeneous world
// point [X Y Z 1]ᵀ to homogeneous image coordinates P·[X Y Z 1]ᵀ. Build one from
// intrinsics and a pose with [NewPinhole].
type CameraMatrix struct {
	// P is the row-major 3×4 projection matrix.
	P [3][4]float64
}

// NewPinhole assembles a [CameraMatrix] as P = K·[R|t], where K is the 3×3
// intrinsic matrix, R the 3×3 world-to-camera rotation, and t the translation.
// The projector in a structured-light rig is modelled as a second pinhole
// "camera" with its own K, R, t.
func NewPinhole(k, r [3][3]float64, t [3]float64) CameraMatrix {
	var rt [3][4]float64
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			rt[i][j] = r[i][j]
		}
		rt[i][3] = t[i]
	}
	var p [3][4]float64
	for i := 0; i < 3; i++ {
		for j := 0; j < 4; j++ {
			var s float64
			for l := 0; l < 3; l++ {
				s += k[i][l] * rt[l][j]
			}
			p[i][j] = s
		}
	}
	return CameraMatrix{P: p}
}

// Project maps a world point to its pixel coordinates (u, v) under this camera,
// performing the perspective divide. The result is undefined for points on the
// camera plane (zero homogeneous depth).
func (c CameraMatrix) Project(x [3]float64) (u, v float64) {
	var h [3]float64
	for i := 0; i < 3; i++ {
		h[i] = c.P[i][0]*x[0] + c.P[i][1]*x[1] + c.P[i][2]*x[2] + c.P[i][3]
	}
	return h[0] / h[2], h[1] / h[2]
}

// PointCloud is the result of triangulating a decoded correspondence field. The
// three slices are parallel: Points[i] is the reconstructed 3-D world point seen
// at camera pixel (PixelX[i], PixelY[i]).
type PointCloud struct {
	// Points holds the reconstructed world coordinates.
	Points [][3]float64
	// PixelX and PixelY are the source camera pixel for each point.
	PixelX []int
	PixelY []int
}

// Len returns the number of reconstructed points.
func (pc *PointCloud) Len() int { return len(pc.Points) }

// TriangulatePoint reconstructs the world point observed at camera pixel
// (camX, camY) and projector pixel (projCol, projRow) from the two projection
// matrices, using the linear (DLT) method: each 2-D observation contributes two
// rows to a homogeneous system A·X=0, and X is the unit null vector of A found
// as the smallest-eigenvalue eigenvector of AᵀA. The homogeneous result is
// dehomogenized to Euclidean (X, Y, Z). Coordinates may be fractional, so this
// works directly with sub-pixel phase-derived correspondences.
func TriangulatePoint(cam, proj CameraMatrix, camX, camY, projCol, projRow float64) [3]float64 {
	rows := [4][4]float64{
		rowMinus(cam.P[2], cam.P[0], camX),
		rowMinus(cam.P[2], cam.P[1], camY),
		rowMinus(proj.P[2], proj.P[0], projCol),
		rowMinus(proj.P[2], proj.P[1], projRow),
	}
	// Normal matrix AᵀA (4×4 symmetric).
	ata := make([][]float64, 4)
	for i := 0; i < 4; i++ {
		ata[i] = make([]float64, 4)
	}
	for i := 0; i < 4; i++ {
		for j := 0; j < 4; j++ {
			var s float64
			for r := 0; r < 4; r++ {
				s += rows[r][i] * rows[r][j]
			}
			ata[i][j] = s
		}
	}
	x := smallestEigenvector(ata)
	w := x[3]
	if w == 0 {
		w = 1e-12
	}
	return [3]float64{x[0] / w, x[1] / w, x[2] / w}
}

// rowMinus returns the DLT row  coord·base − sub  for a 4-vector observation.
func rowMinus(base, sub [4]float64, coord float64) [4]float64 {
	var out [4]float64
	for i := 0; i < 4; i++ {
		out[i] = coord*base[i] - sub[i]
	}
	return out
}

// Triangulate reconstructs a [PointCloud] from a Gray-code/phase decoding by
// triangulating every valid camera pixel against its decoded projector pixel.
// The camera pixel position (x, y) and the decoded projector coordinate
// (Col, Row) supply the two views for [TriangulatePoint]. Pixels that are
// invalid in dec.Mask are skipped. cam and proj are the calibrated projection
// matrices of the camera and projector.
func Triangulate(dec *Decoded, cam, proj CameraMatrix) *PointCloud {
	pc := &PointCloud{}
	for y := 0; y < dec.Rows; y++ {
		for x := 0; x < dec.Cols; x++ {
			i := y*dec.Cols + x
			if !dec.Mask[i] {
				continue
			}
			p := TriangulatePoint(cam, proj, float64(x), float64(y), float64(dec.Col[i]), float64(dec.Row[i]))
			pc.Points = append(pc.Points, p)
			pc.PixelX = append(pc.PixelX, x)
			pc.PixelY = append(pc.PixelY, y)
		}
	}
	return pc
}
