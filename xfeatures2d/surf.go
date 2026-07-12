package xfeatures2d

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// SURF is a lightweight Speeded-Up Robust Features detector and descriptor,
// following the principles of OpenCV's cv::xfeatures2d::SURF without using the
// patented reference implementation.
//
// Detection is a Fast-Hessian: the determinant of an approximate Hessian, built
// from box filters that stand in for the second-order Gaussian derivatives and
// evaluated in constant time through an integral image, is computed over a range
// of filter sizes (scales); scale-space maxima above HessianThreshold become
// keypoints. Description follows SURF-64: an orientation is assigned from Haar
// wavelet responses, and a 4×4 grid of subregions each contributes the sums of
// the (rotated) Haar responses and their absolute values, giving a 64-element
// vector that is L2 normalised. Descriptors are compared with the [L2Distance].
type SURF struct {
	// HessianThreshold is the minimum determinant-of-Hessian response for a
	// keypoint.
	HessianThreshold float64
	// FilterSizes lists the box-filter sizes (odd, multiples of 6 plus 3) used
	// as scales, from fine to coarse.
	FilterSizes []int
	// Upright, when true, skips orientation assignment (U-SURF): descriptors are
	// computed on an axis-aligned grid.
	Upright bool
}

// NewSURF returns a SURF detector/descriptor with the given Hessian threshold
// and the standard first-octave filter sizes.
func NewSURF(hessianThreshold float64) *SURF {
	return &SURF{
		HessianThreshold: hessianThreshold,
		FilterSizes:      []int{9, 15, 21, 27, 33, 39, 45, 51},
	}
}

// DescriptorSize returns the number of floats in each descriptor (64).
func (s *SURF) DescriptorSize() int { return 64 }

// boxIntegral sums src over the box whose top-left sample is (row, col) and that
// spans height rows and width cols, with clamping to the image.
func boxIntegral(it *integral, row, col, height, width int) float64 {
	return float64(it.sum(col, row, col+width-1, row+height-1))
}

// hessianResponse returns the normalised determinant of the approximate Hessian
// at (x, y) for a box filter of the given size.
func hessianResponse(it *integral, x, y, size int) float64 {
	l := size / 3 // lobe size
	b := size / 2
	w := size
	inv := 1.0 / float64(w*w)

	dxx := boxIntegral(it, y-l+1, x-b, 2*l-1, w) -
		boxIntegral(it, y-l+1, x-l/2, 2*l-1, l)*3
	dyy := boxIntegral(it, y-b, x-l+1, w, 2*l-1) -
		boxIntegral(it, y-l/2, x-l+1, l, 2*l-1)*3
	dxy := boxIntegral(it, y-l, x+1, l, l) +
		boxIntegral(it, y+1, x-l, l, l) -
		boxIntegral(it, y-l, x-l, l, l) -
		boxIntegral(it, y+1, x+1, l, l)

	dxx *= inv
	dyy *= inv
	dxy *= inv
	return dxx*dyy - 0.81*dxy*dxy
}

// Detect finds Fast-Hessian keypoints in img. Each keypoint's Response is the
// Hessian determinant, Size is the SURF scale diameter and Angle is -1 (set by
// Compute). img may be single- or three-channel; a colour image is converted to
// gray.
func (s *SURF) Detect(img *cv.Mat) []KeyPoint {
	gray := toGray(img)
	rows, cols := gray.Rows, gray.Cols
	it := newIntegral(gray)
	sizes := s.FilterSizes
	ns := len(sizes)

	resp := make([][]float64, ns)
	for k, size := range sizes {
		resp[k] = make([]float64, rows*cols)
		border := size
		for y := border; y < rows-border; y++ {
			for x := border; x < cols-border; x++ {
				resp[k][y*cols+x] = hessianResponse(it, x, y, size)
			}
		}
	}

	var kps []KeyPoint
	for k := 1; k < ns-1; k++ {
		size := sizes[k]
		border := sizes[k+1]
		for y := border; y < rows-border; y++ {
			for x := border; x < cols-border; x++ {
				v := resp[k][y*cols+x]
				if v < s.HessianThreshold {
					continue
				}
				if !surfIsMax(resp, k, x, y, cols, rows, v) {
					continue
				}
				scale := 1.2 * float64(size) / 9.0
				kps = append(kps, KeyPoint{
					Pt:       cv.Point{X: x, Y: y},
					Size:     2 * scale,
					Angle:    -1,
					Response: v,
				})
			}
		}
	}
	return kps
}

// surfIsMax reports whether resp at (x,y,k) is a strict maximum in the 3×3×3
// scale-space neighbourhood.
func surfIsMax(resp [][]float64, k, x, y, cols, rows int, val float64) bool {
	for dk := -1; dk <= 1; dk++ {
		kk := k + dk
		if kk < 0 || kk >= len(resp) {
			continue
		}
		for dy := -1; dy <= 1; dy++ {
			ny := y + dy
			if ny < 0 || ny >= rows {
				continue
			}
			for dx := -1; dx <= 1; dx++ {
				nx := x + dx
				if nx < 0 || nx >= cols {
					continue
				}
				if dk == 0 && dx == 0 && dy == 0 {
					continue
				}
				if resp[kk][ny*cols+nx] >= val {
					return false
				}
			}
		}
	}
	return true
}

// haarX returns the horizontal Haar wavelet response of size 2s at (x, y).
func haarX(it *integral, x, y, s int) float64 {
	left := float64(it.sum(x-s, y-s, x-1, y+s-1))
	right := float64(it.sum(x, y-s, x+s-1, y+s-1))
	return right - left
}

// haarY returns the vertical Haar wavelet response of size 2s at (x, y).
func haarY(it *integral, x, y, s int) float64 {
	top := float64(it.sum(x-s, y-s, x+s-1, y-1))
	bottom := float64(it.sum(x-s, y, x+s-1, y+s-1))
	return bottom - top
}

// orientation assigns a SURF orientation (radians) at (x, y) with the given
// scale by the sliding-window method over Haar responses on a circular grid.
func (s *SURF) orientation(it *integral, x, y int, scale float64) float64 {
	wave := int(math.Round(2 * scale))
	if wave < 1 {
		wave = 1
	}
	type resp struct{ dx, dy, ang float64 }
	var samples []resp
	for i := -6; i <= 6; i++ {
		for j := -6; j <= 6; j++ {
			if i*i+j*j > 36 {
				continue
			}
			px := x + int(math.Round(float64(i)*scale))
			py := y + int(math.Round(float64(j)*scale))
			g := math.Exp(-float64(i*i+j*j) / (2 * 2.5 * 2.5))
			dx := g * haarX(it, px, py, wave)
			dy := g * haarY(it, px, py, wave)
			samples = append(samples, resp{dx, dy, math.Atan2(dy, dx)})
		}
	}
	best := 0.0
	bestAng := 0.0
	for a := 0.0; a < 2*math.Pi; a += 0.15 {
		lo := a
		hi := a + math.Pi/3
		var sx, sy float64
		for _, r := range samples {
			ang := r.ang
			in := false
			if hi < 2*math.Pi {
				in = ang >= lo && ang < hi
			} else {
				in = ang >= lo || ang < hi-2*math.Pi
			}
			if in {
				sx += r.dx
				sy += r.dy
			}
		}
		mag := sx*sx + sy*sy
		if mag > best {
			best = mag
			bestAng = math.Atan2(sy, sx)
		}
	}
	return bestAng
}

// Compute describes each keypoint of img and returns the keypoints (with Angle
// set unless Upright) together with their 64-element float descriptors, L2
// normalised. Sampling uses border replication, so no keypoint is dropped. img
// may be single- or three-channel; a colour image is converted to gray.
func (s *SURF) Compute(img *cv.Mat, keypoints []KeyPoint) ([]KeyPoint, [][]float64) {
	gray := toGray(img)
	it := newIntegral(gray)
	out := make([]KeyPoint, len(keypoints))
	descs := make([][]float64, len(keypoints))

	for k, kp := range keypoints {
		scale := kp.Size / 2
		if scale <= 0 {
			scale = 1.2
		}
		x, y := kp.Pt.X, kp.Pt.Y
		ori := 0.0
		if !s.Upright {
			ori = s.orientation(it, x, y, scale)
		}
		co, si := math.Cos(ori), math.Sin(ori)
		wave := int(math.Round(2 * scale))
		if wave < 1 {
			wave = 1
		}
		desc := make([]float64, 64)
		d := 0
		// 4×4 subregions spanning [-10,10] scale units.
		for a := 0; a < 4; a++ {
			for b := 0; b < 4; b++ {
				var dxs, dys, adxs, adys float64
				for pi := 0; pi < 5; pi++ {
					for pj := 0; pj < 5; pj++ {
						lx := float64(b*5+pj) - 10 + 0.5
						ly := float64(a*5+pi) - 10 + 0.5
						sx := float64(x) + (lx*co-ly*si)*scale
						sy := float64(y) + (lx*si+ly*co)*scale
						g := math.Exp(-(lx*lx + ly*ly) / (2 * 3.3 * 3.3))
						rx := haarX(it, int(math.Round(sx)), int(math.Round(sy)), wave)
						ry := haarY(it, int(math.Round(sx)), int(math.Round(sy)), wave)
						// Rotate responses into the keypoint frame.
						rrx := g * (-rx*si + ry*co)
						rry := g * (rx*co + ry*si)
						dxs += rrx
						dys += rry
						adxs += math.Abs(rrx)
						adys += math.Abs(rry)
					}
				}
				desc[d] = dxs
				desc[d+1] = dys
				desc[d+2] = adxs
				desc[d+3] = adys
				d += 4
			}
		}
		// L2 normalise.
		var norm float64
		for _, v := range desc {
			norm += v * v
		}
		norm = math.Sqrt(norm)
		if norm > 1e-12 {
			for i := range desc {
				desc[i] /= norm
			}
		}
		deg := ori * 180 / math.Pi
		if deg < 0 {
			deg += 360
		}
		if !s.Upright {
			kp.Angle = deg
		}
		out[k] = kp
		descs[k] = desc
	}
	return out, descs
}

// DetectAndCompute detects Fast-Hessian keypoints and describes them, returning
// the keypoints and their descriptors. img may be single- or three-channel.
func (s *SURF) DetectAndCompute(img *cv.Mat) ([]KeyPoint, [][]float64) {
	return s.Compute(img, s.Detect(img))
}
