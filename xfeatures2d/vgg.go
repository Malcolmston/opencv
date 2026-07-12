package xfeatures2d

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// VGG computes a weight-free gradient-pooling descriptor in the spirit of
// OpenCV's cv::xfeatures2d::VGG.
//
// The original VGG descriptor (Simonyan et al.) applies a discriminatively
// learned linear projection to a dense pooling of image gradients. This port
// keeps the geometric front end — a log-polar arrangement of spatial pooling
// regions, each accumulating a soft-binned gradient-orientation histogram — but
// omits the learned projection matrix (documented as the weight-free
// approximation, so no trained tables are embedded). The layout is a GLOH-like
// grid: a central region plus RadialBins concentric rings each split into
// AngularBins sectors, every region holding an OrientationBins histogram. The
// pattern is rotated to the keypoint orientation, giving rotation invariance,
// and the descriptor is SIFT-style L2 normalised, clipped and renormalised. The
// result is a real-valued descriptor compared with the [L2Distance].
type VGG struct {
	// Radius is the radius of the pooling region in pixels.
	Radius float64
	// RadialBins is the number of concentric rings (excluding the centre).
	RadialBins int
	// AngularBins is the number of angular sectors per ring.
	AngularBins int
	// OrientationBins is the number of gradient-orientation histogram bins per
	// spatial region.
	OrientationBins int
}

// NewVGG returns a VGG descriptor with a GLOH-like default configuration
// (radius 20, 2 rings, 8 sectors, 8 orientation bins), a (1+2*8)*8 =
// 136-dimensional descriptor.
func NewVGG() *VGG {
	return &VGG{Radius: 20, RadialBins: 2, AngularBins: 8, OrientationBins: 8}
}

// spatialRegions returns the number of spatial pooling regions (centre + rings).
func (v *VGG) spatialRegions() int { return 1 + v.RadialBins*v.AngularBins }

// DescriptorSize returns the number of floats in each descriptor.
func (v *VGG) DescriptorSize() int { return v.spatialRegions() * v.OrientationBins }

// Compute describes each keypoint of img and returns the keypoints (with Angle
// set to the intensity-centroid orientation) together with their float
// descriptors. Sampling uses border replication, so no keypoint is dropped. img
// may be single- or three-channel; a colour image is converted to gray.
func (v *VGG) Compute(img *cv.Mat, keypoints []KeyPoint) ([]KeyPoint, [][]float64) {
	gray := toGray(img)
	rows, cols := gray.Rows, gray.Cols
	gx, gy := gradientMaps(gray)
	radius := v.Radius
	ob := v.OrientationBins
	descLen := v.DescriptorSize()

	out := make([]KeyPoint, len(keypoints))
	descs := make([][]float64, len(keypoints))

	for k, kp := range keypoints {
		cx, cy := kp.Pt.X, kp.Pt.Y
		angle := intensityCentroidAngle(gray, cx, cy, int(radius))
		ca, sa := math.Cos(angle), math.Sin(angle)
		desc := make([]float64, descLen)
		ir := int(math.Ceil(radius))
		for dy := -ir; dy <= ir; dy++ {
			for dx := -ir; dx <= ir; dx++ {
				dist := math.Hypot(float64(dx), float64(dy))
				if dist > radius {
					continue
				}
				px, py := cx+dx, cy+dy
				if px < 0 || px >= cols || py < 0 || py >= rows {
					continue
				}
				// Gradient in the keypoint-rotated frame.
				gvx := gx[py*cols+px]
				gvy := gy[py*cols+px]
				mag := math.Hypot(gvx, gvy)
				if mag < 1e-9 {
					continue
				}
				gori := math.Atan2(gvy, gvx) - angle
				// Rotated spatial position.
				rx := float64(dx)*ca + float64(dy)*sa
				ry := -float64(dx)*sa + float64(dy)*ca
				region := v.regionIndex(rx, ry, dist/radius)
				bin := orientBin(gori, ob)
				// Gaussian spatial weighting towards the centre.
				w := math.Exp(-(dist * dist) / (0.5 * radius * radius))
				desc[region*ob+bin] += mag * w
			}
		}
		normalizeSIFT(desc)
		deg := angle * 180 / math.Pi
		if deg < 0 {
			deg += 360
		}
		kp.Angle = deg
		out[k] = kp
		descs[k] = desc
	}
	return out, descs
}

// regionIndex maps a rotated spatial offset (rx, ry) with normalised radius
// rNorm in [0,1] to a log-polar region index: 0 for the central disc, then one
// index per (ring, sector).
func (v *VGG) regionIndex(rx, ry, rNorm float64) int {
	// Central disc: innermost fraction of the radius.
	if rNorm < 1.0/float64(v.RadialBins+1) {
		return 0
	}
	ring := int(rNorm * float64(v.RadialBins))
	if ring >= v.RadialBins {
		ring = v.RadialBins - 1
	}
	theta := math.Atan2(ry, rx)
	if theta < 0 {
		theta += 2 * math.Pi
	}
	sector := int(theta / (2 * math.Pi) * float64(v.AngularBins))
	if sector >= v.AngularBins {
		sector = v.AngularBins - 1
	}
	return 1 + ring*v.AngularBins + sector
}

// orientBin maps an orientation in radians to a histogram bin in [0, bins).
func orientBin(theta float64, bins int) int {
	for theta < 0 {
		theta += 2 * math.Pi
	}
	for theta >= 2*math.Pi {
		theta -= 2 * math.Pi
	}
	b := int(theta / (2 * math.Pi) * float64(bins))
	if b >= bins {
		b = bins - 1
	}
	return b
}

// normalizeSIFT applies the SIFT normalisation: L2 normalise, clip large
// components to 0.2, and L2 normalise again.
func normalizeSIFT(desc []float64) {
	l2 := func() float64 {
		var s float64
		for _, v := range desc {
			s += v * v
		}
		return math.Sqrt(s)
	}
	n := l2()
	if n < 1e-12 {
		return
	}
	for i := range desc {
		desc[i] /= n
		if desc[i] > 0.2 {
			desc[i] = 0.2
		}
	}
	n = l2()
	if n < 1e-12 {
		return
	}
	for i := range desc {
		desc[i] /= n
	}
}
