package hdr

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// bilateralPlane applies an edge-preserving bilateral filter to a plane. Each
// output sample is a weighted average of its neighbours where the weight is the
// product of a spatial Gaussian (sigmaSpace, in pixels) and a range Gaussian
// (sigmaColor, in sample units), so smoothing does not cross strong edges.
// Borders are handled by mirror reflection. The window radius is 2*sigmaSpace.
func bilateralPlane(p *plane, sigmaSpace, sigmaColor float64) *plane {
	if sigmaSpace <= 0 {
		sigmaSpace = 1
	}
	if sigmaColor <= 0 {
		sigmaColor = 1
	}
	radius := int(math.Ceil(2 * sigmaSpace))
	if radius < 1 {
		radius = 1
	}
	// Precompute the separable spatial weights.
	spatial := make([]float64, 2*radius+1)
	for i := -radius; i <= radius; i++ {
		spatial[i+radius] = math.Exp(-float64(i*i) / (2 * sigmaSpace * sigmaSpace))
	}
	invColor := 1.0 / (2 * sigmaColor * sigmaColor)
	out := newPlane(p.rows, p.cols)
	for y := 0; y < p.rows; y++ {
		for x := 0; x < p.cols; x++ {
			center := p.at(y, x)
			var sum, wsum float64
			for dy := -radius; dy <= radius; dy++ {
				sw := spatial[dy+radius]
				for dx := -radius; dx <= radius; dx++ {
					v := p.atReflect(y+dy, x+dx)
					diff := v - center
					w := sw * spatial[dx+radius] * math.Exp(-diff*diff*invColor)
					sum += w * v
					wsum += w
				}
			}
			if wsum > 0 {
				out.set(y, x, sum/wsum)
			} else {
				out.set(y, x, center)
			}
		}
	}
	return out
}

// matToPlanes01 splits an 8-bit Mat into per-channel float planes in [0,1].
func matToPlanes01(m *cv.Mat) []*plane {
	return toFloatPlanes(m)
}

// planesToMat01 recombines [0,1] float planes into an 8-bit interleaved Mat.
func planesToMat01(planes []*plane) *cv.Mat {
	ch := len(planes)
	out := cv.NewMat(planes[0].rows, planes[0].cols, ch)
	total := planes[0].rows * planes[0].cols
	for c := 0; c < ch; c++ {
		for i := 0; i < total; i++ {
			out.Data[i*ch+c] = clamp8(clamp01(planes[c].data[i]) * 255)
		}
	}
	return out
}

// EdgePreservingFilter smooths an 8-bit image while keeping edges crisp, the
// analogue of OpenCV's edgePreservingFilter. It runs a bilateral filter on each
// channel independently. sigmaSpace is the spatial extent in pixels;
// sigmaColor, in [0,1] intensity units, controls how large an intensity
// difference is treated as an edge. Non-positive arguments select mild
// defaults.
func EdgePreservingFilter(m *cv.Mat, sigmaSpace, sigmaColor float64) *cv.Mat {
	if m == nil || m.Empty() {
		panic("hdr: EdgePreservingFilter on nil or empty image")
	}
	if sigmaSpace <= 0 {
		sigmaSpace = 3
	}
	if sigmaColor <= 0 {
		sigmaColor = 0.1
	}
	planes := matToPlanes01(m)
	for c := range planes {
		planes[c] = bilateralPlane(planes[c], sigmaSpace, sigmaColor)
	}
	return planesToMat01(planes)
}

// DetailEnhance sharpens local detail while leaving large-scale tone untouched,
// the analogue of OpenCV's detailEnhance and a natural finishing step on a
// tonemapped HDR image. Each channel is split into a bilateral-filtered base and
// the residual detail; the output is base + boost*detail. sigmaSpace (pixels)
// and sigmaColor ([0,1] intensity) set the edge-preserving base; boost is the
// detail gain (1 is a no-op, values above 1 enhance). Non-positive arguments
// select defaults (sigmaSpace 3, sigmaColor 0.15, boost 2).
func DetailEnhance(m *cv.Mat, sigmaSpace, sigmaColor, boost float64) *cv.Mat {
	if m == nil || m.Empty() {
		panic("hdr: DetailEnhance on nil or empty image")
	}
	if sigmaSpace <= 0 {
		sigmaSpace = 3
	}
	if sigmaColor <= 0 {
		sigmaColor = 0.15
	}
	if boost <= 0 {
		boost = 2
	}
	planes := matToPlanes01(m)
	for c := range planes {
		base := bilateralPlane(planes[c], sigmaSpace, sigmaColor)
		res := planes[c]
		for i := range res.data {
			res.data[i] = base.data[i] + boost*(res.data[i]-base.data[i])
		}
	}
	return planesToMat01(planes)
}
