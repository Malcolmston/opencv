package imghash2

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// cmSize is the side length the image is scaled to before moments are taken.
const cmSize = 64

// ColorMomentHash implements a colour-moment hash in the spirit of OpenCV's
// cv::img_hash::ColorMomentHash. The image is scaled to a fixed square and
// expanded into six single-channel planes — the red, green and blue planes and
// the hue, saturation and value planes of its HSV form. For each plane the
// seven Hu moment invariants are computed and log-scaled, giving a
// 42-dimensional [FloatHash] compared by L1 distance.
//
// Hu moments are invariant to translation, scale and rotation, so the
// descriptor keys on the distribution of colour and brightness rather than on
// exact pixel positions. Identical images compare as 0; a blurred or
// brightness-shifted copy stays close, while a structurally different image is
// far. The zero value is ready to use; [NewColorMomentHash] is provided for
// symmetry.
type ColorMomentHash struct{}

// NewColorMomentHash returns a ready-to-use [ColorMomentHash].
func NewColorMomentHash() ColorMomentHash { return ColorMomentHash{} }

// Name returns the identifier "colormoment".
func (ColorMomentHash) Name() string { return "colormoment" }

// ComputeFloat returns the 42-value colour-moment descriptor of img.
func (ColorMomentHash) ComputeFloat(img *cv.Mat) FloatHash {
	requireImage(img, "ColorMomentHash.ComputeFloat")

	var rgb *cv.Mat
	switch img.Channels {
	case 3:
		rgb = img
	case 1:
		rgb = cv.CvtColor(img, cv.ColorGray2RGB)
	default:
		rgb = cv.CvtColor(ToGray(img), cv.ColorGray2RGB)
	}
	rgb = cv.Resize(rgb, cmSize, cmSize, cv.InterLinear)
	hsv := cv.CvtColor(rgb, cv.ColorRGB2HSV)

	feats := make(FloatHash, 0, 42)
	for c := 0; c < 3; c++ {
		hu := HuMoments(rgb, c)
		feats = append(feats, logScaleHu(hu)...)
	}
	for c := 0; c < 3; c++ {
		hu := HuMoments(hsv, c)
		feats = append(feats, logScaleHu(hu)...)
	}
	return feats
}

// ColorMoment is a convenience wrapper returning the [ColorMomentHash]
// descriptor of img.
func ColorMoment(img *cv.Mat) FloatHash { return ColorMomentHash{}.ComputeFloat(img) }

// HuMoments returns the seven Hu moment invariants of channel c of img, treating
// that channel's samples as an intensity image. The invariants are functions of
// the scale-normalised central moments and are invariant to translation, scale
// and rotation, which makes them a compact shape descriptor. They are returned
// in their raw (not log-scaled) form; [ColorMomentHash] log-scales them
// internally. It panics if c is out of range.
func HuMoments(img *cv.Mat, c int) [7]float64 {
	if c < 0 || c >= img.Channels {
		panic("imghash2: HuMoments channel out of range")
	}
	var m00, m10, m01, m11, m20, m02, m30, m03, m21, m12 float64
	ch := img.Channels
	for y := 0; y < img.Rows; y++ {
		fy := float64(y)
		for x := 0; x < img.Cols; x++ {
			v := float64(img.Data[(y*img.Cols+x)*ch+c])
			if v == 0 {
				continue
			}
			fx := float64(x)
			m00 += v
			m10 += fx * v
			m01 += fy * v
			m11 += fx * fy * v
			m20 += fx * fx * v
			m02 += fy * fy * v
			m30 += fx * fx * fx * v
			m03 += fy * fy * fy * v
			m21 += fx * fx * fy * v
			m12 += fx * fy * fy * v
		}
	}
	var hu [7]float64
	if m00 == 0 {
		return hu
	}
	xbar := m10 / m00
	ybar := m01 / m00

	mu20 := m20 - xbar*m10
	mu02 := m02 - ybar*m01
	mu11 := m11 - xbar*m01
	mu30 := m30 - 3*xbar*m20 + 2*xbar*xbar*m10
	mu03 := m03 - 3*ybar*m02 + 2*ybar*ybar*m01
	mu21 := m21 - 2*xbar*m11 - ybar*m20 + 2*xbar*xbar*m01
	mu12 := m12 - 2*ybar*m11 - xbar*m02 + 2*ybar*ybar*m10

	inv2 := 1 / (m00 * m00)
	inv25 := 1 / (m00 * m00 * math.Sqrt(m00))
	n20 := mu20 * inv2
	n02 := mu02 * inv2
	n11 := mu11 * inv2
	n30 := mu30 * inv25
	n03 := mu03 * inv25
	n21 := mu21 * inv25
	n12 := mu12 * inv25

	hu[0] = n20 + n02
	hu[1] = (n20-n02)*(n20-n02) + 4*n11*n11
	hu[2] = (n30-3*n12)*(n30-3*n12) + (3*n21-n03)*(3*n21-n03)
	hu[3] = (n30+n12)*(n30+n12) + (n21+n03)*(n21+n03)
	hu[4] = (n30-3*n12)*(n30+n12)*((n30+n12)*(n30+n12)-3*(n21+n03)*(n21+n03)) +
		(3*n21-n03)*(n21+n03)*(3*(n30+n12)*(n30+n12)-(n21+n03)*(n21+n03))
	hu[5] = (n20-n02)*((n30+n12)*(n30+n12)-(n21+n03)*(n21+n03)) +
		4*n11*(n30+n12)*(n21+n03)
	hu[6] = (3*n21-n03)*(n30+n12)*((n30+n12)*(n30+n12)-3*(n21+n03)*(n21+n03)) -
		(n30-3*n12)*(n21+n03)*(3*(n30+n12)*(n30+n12)-(n21+n03)*(n21+n03))
	return hu
}

// logScaleHu log-scales the Hu invariants to compress their wide dynamic range,
// matching OpenCV: each value h becomes -sign(h)·log10(|h|), with zeros left as
// zero.
func logScaleHu(hu [7]float64) []float64 {
	out := make([]float64, 7)
	for i, h := range hu {
		if h == 0 {
			continue
		}
		out[i] = math.Copysign(math.Log10(math.Abs(h)), h) * -1
	}
	return out
}
