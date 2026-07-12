package xfeatures2d

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// DAISY computes the DAISY dense gradient-histogram descriptor, a port of
// OpenCV's cv::xfeatures2d::DAISY.
//
// DAISY generalises SIFT/GLOH with a computationally efficient layout. The image
// gradient is decomposed into a set of orientation layers (the positive response
// of the gradient onto each of H directions); every layer is then smoothed with
// a series of Gaussians of increasing width. The descriptor of a keypoint
// concatenates, for the centre and for T points on each of Q concentric rings,
// the H smoothed orientation responses at that location, each per-histogram L2
// normalised. The result is a real-valued descriptor compared with the
// [L2Distance].
type DAISY struct {
	// Radius is the radius of the outermost ring in pixels.
	Radius float64
	// RingCount is the number of concentric rings (Q).
	RingCount int
	// PointsPerRing is the number of sample points on each ring (T).
	PointsPerRing int
	// Orientations is the number of gradient orientation layers (H).
	Orientations int
}

// NewDAISY returns a DAISY extractor with the common default configuration
// (radius 15, 3 rings, 8 points per ring, 8 orientations), yielding a
// (3*8+1)*8 = 200-dimensional descriptor.
func NewDAISY() *DAISY {
	return &DAISY{Radius: 15, RingCount: 3, PointsPerRing: 8, Orientations: 8}
}

// DescriptorSize returns the number of floats in each descriptor.
func (d *DAISY) DescriptorSize() int {
	return (d.RingCount*d.PointsPerRing + 1) * d.Orientations
}

// blurFloat applies a separable Gaussian blur of the given sigma to a row-major
// float map with border replication.
func blurFloat(src []float64, rows, cols int, sigma float64) []float64 {
	if sigma <= 0 {
		out := make([]float64, len(src))
		copy(out, src)
		return out
	}
	ksize := gaussianKSize(sigma)
	kernel := cv.GaussianKernel1D(ksize, sigma)
	half := ksize / 2
	tmp := make([]float64, rows*cols)
	// Horizontal pass.
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			var acc float64
			for t := -half; t <= half; t++ {
				sx := x + t
				if sx < 0 {
					sx = 0
				} else if sx >= cols {
					sx = cols - 1
				}
				acc += kernel[t+half] * src[y*cols+sx]
			}
			tmp[y*cols+x] = acc
		}
	}
	// Vertical pass.
	out := make([]float64, rows*cols)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			var acc float64
			for t := -half; t <= half; t++ {
				sy := y + t
				if sy < 0 {
					sy = 0
				} else if sy >= rows {
					sy = rows - 1
				}
				acc += kernel[t+half] * tmp[sy*cols+x]
			}
			out[y*cols+x] = acc
		}
	}
	return out
}

// buildLayers builds, for each ring level (0 = centre, then one per ring), the H
// Gaussian-smoothed orientation layers. layers[level][h] is a row-major map.
func (d *DAISY) buildLayers(gray *cv.Mat) [][][]float64 {
	rows, cols := gray.Rows, gray.Cols
	gx, gy := gradientMaps(gray)
	H := d.Orientations
	// Base orientation layers: positive projection of the gradient.
	base := make([][]float64, H)
	for h := 0; h < H; h++ {
		theta := 2 * math.Pi * float64(h) / float64(H)
		ct, st := math.Cos(theta), math.Sin(theta)
		layer := make([]float64, rows*cols)
		for i := range layer {
			v := gx[i]*ct + gy[i]*st
			if v > 0 {
				layer[i] = v
			}
		}
		base[h] = layer
	}
	// One smoothing level per ring plus the centre; sigma grows with ring index.
	levels := d.RingCount + 1
	layers := make([][][]float64, levels)
	for lvl := 0; lvl < levels; lvl++ {
		sigma := d.Radius * float64(lvl+1) / float64(2*levels)
		layers[lvl] = make([][]float64, H)
		for h := 0; h < H; h++ {
			layers[lvl][h] = blurFloat(base[h], rows, cols, sigma)
		}
	}
	return layers
}

// sampleHistogram reads the H smoothed orientation responses at fractional
// location (x, y) of the given smoothing level and appends the per-histogram
// L2-normalised values to dst.
func (d *DAISY) sampleHistogram(layers [][][]float64, level int, x, y float64, rows, cols int, dst []float64) {
	H := d.Orientations
	x0 := int(math.Floor(x))
	y0 := int(math.Floor(y))
	fx := x - float64(x0)
	fy := y - float64(y0)
	clampX := func(v int) int {
		if v < 0 {
			return 0
		}
		if v >= cols {
			return cols - 1
		}
		return v
	}
	clampY := func(v int) int {
		if v < 0 {
			return 0
		}
		if v >= rows {
			return rows - 1
		}
		return v
	}
	xa, xb := clampX(x0), clampX(x0+1)
	ya, yb := clampY(y0), clampY(y0+1)
	var norm float64
	for h := 0; h < H; h++ {
		lay := layers[level][h]
		v00 := lay[ya*cols+xa]
		v10 := lay[ya*cols+xb]
		v01 := lay[yb*cols+xa]
		v11 := lay[yb*cols+xb]
		top := v00*(1-fx) + v10*fx
		bot := v01*(1-fx) + v11*fx
		val := top*(1-fy) + bot*fy
		dst[h] = val
		norm += val * val
	}
	norm = math.Sqrt(norm)
	if norm > 1e-12 {
		for h := 0; h < H; h++ {
			dst[h] /= norm
		}
	}
}

// Compute describes each keypoint of img and returns the keypoints unchanged
// together with their float descriptors (one []float64 of length DescriptorSize
// per keypoint). Sampling uses border replication, so no keypoint is dropped.
// img may be single- or three-channel; a colour image is converted to gray.
func (d *DAISY) Compute(img *cv.Mat, keypoints []KeyPoint) ([]KeyPoint, [][]float64) {
	gray := toGray(img)
	rows, cols := gray.Rows, gray.Cols
	layers := d.buildLayers(gray)
	H := d.Orientations
	out := make([]KeyPoint, len(keypoints))
	descs := make([][]float64, len(keypoints))
	hist := make([]float64, H)

	for k, kp := range keypoints {
		fx := float64(kp.Pt.X)
		fy := float64(kp.Pt.Y)
		desc := make([]float64, d.DescriptorSize())
		pos := 0
		// Centre uses the finest smoothing level (0).
		d.sampleHistogram(layers, 0, fx, fy, rows, cols, hist)
		copy(desc[pos:pos+H], hist)
		pos += H
		for r := 0; r < d.RingCount; r++ {
			radius := d.Radius * float64(r+1) / float64(d.RingCount)
			level := r + 1
			for tIdx := 0; tIdx < d.PointsPerRing; tIdx++ {
				theta := 2 * math.Pi * float64(tIdx) / float64(d.PointsPerRing)
				sx := fx + radius*math.Cos(theta)
				sy := fy + radius*math.Sin(theta)
				d.sampleHistogram(layers, level, sx, sy, rows, cols, hist)
				copy(desc[pos:pos+H], hist)
				pos += H
			}
		}
		out[k] = kp
		descs[k] = desc
	}
	return out, descs
}
