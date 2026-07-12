package imgprocx

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// cornerMaxIters bounds the sub-pixel refinement iterations per corner.
const cornerMaxIters = 20

// cornerEps is the squared-displacement convergence threshold (in pixels²) for
// [CornerSubPix]: iteration stops once a corner moves less than this.
const cornerEps = 1e-6

// CornerSubPix refines integer corner locations to sub-pixel accuracy, mirroring
// cv2.cornerSubPix. img may be single- or three-channel (it is converted to
// luminance); corners holds approximate corner positions (for example from a
// corner detector); and winSize is the half-side of the square search window in
// pixels, so a winSize of 5 examines an 11×11 neighbourhood. winSize must be
// positive.
//
// For each corner it iterates the standard gradient-orthogonality update: the
// refined position q is the point minimising the Gaussian-weighted sum of
// squared dot products between the local image gradient and the vector from q to
// each neighbour, i.e. the solution of
//
//	(Σ wᵢ gᵢgᵢᵀ) q = Σ wᵢ gᵢgᵢᵀ pᵢ
//
// where gᵢ is the gradient and pᵢ the position of neighbour i. Each corner is
// refined independently and the results are returned in the same order as the
// input as [Point2f] values. Corners whose neighbourhood has a degenerate
// gradient system are returned unchanged (at their integer position).
func CornerSubPix(img *cv.Mat, corners []cv.Point, winSize int) []Point2f {
	if winSize <= 0 {
		panic("imgprocx: CornerSubPix requires winSize > 0")
	}
	gray, rows, cols := toGrayPlane(img)
	// Precompute Gaussian weights over the window; sigma spans the window.
	side := 2*winSize + 1
	weights := make([]float64, side*side)
	sigma := float64(winSize) / 2.0
	if sigma <= 0 {
		sigma = 1
	}
	twoSig2 := 2 * sigma * sigma
	wi := 0
	for dy := -winSize; dy <= winSize; dy++ {
		for dx := -winSize; dx <= winSize; dx++ {
			weights[wi] = math.Exp(-float64(dx*dx+dy*dy) / twoSig2)
			wi++
		}
	}
	at := func(y, x int) float64 {
		if y < 0 {
			y = 0
		} else if y >= rows {
			y = rows - 1
		}
		if x < 0 {
			x = 0
		} else if x >= cols {
			x = cols - 1
		}
		return gray[y*cols+x]
	}

	out := make([]Point2f, len(corners))
	for ci, corner := range corners {
		cx := float64(corner.X)
		cy := float64(corner.Y)
		for iter := 0; iter < cornerMaxIters; iter++ {
			var a11, a12, a22, b1, b2 float64
			wi := 0
			for dy := -winSize; dy <= winSize; dy++ {
				for dx := -winSize; dx <= winSize; dx++ {
					w := weights[wi]
					wi++
					px := int(math.Round(cx)) + dx
					py := int(math.Round(cy)) + dy
					// Central-difference gradient at the neighbour.
					gx := 0.5 * (at(py, px+1) - at(py, px-1))
					gy := 0.5 * (at(py+1, px) - at(py-1, px))
					gxx := gx * gx * w
					gxy := gx * gy * w
					gyy := gy * gy * w
					a11 += gxx
					a12 += gxy
					a22 += gyy
					b1 += gxx*float64(px) + gxy*float64(py)
					b2 += gxy*float64(px) + gyy*float64(py)
				}
			}
			det := a11*a22 - a12*a12
			if math.Abs(det) < 1e-12 {
				break
			}
			nx := (a22*b1 - a12*b2) / det
			ny := (a11*b2 - a12*b1) / det
			ddx := nx - cx
			ddy := ny - cy
			cx, cy = nx, ny
			if ddx*ddx+ddy*ddy < cornerEps {
				break
			}
		}
		out[ci] = Point2f{X: cx, Y: cy}
	}
	return out
}
